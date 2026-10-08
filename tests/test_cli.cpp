#include <sys/wait.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "check.hpp"
#include "openv6/json.hpp"

namespace fs = std::filesystem;

namespace {

std::string read_text(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

struct Run {
    int status;
    std::string output;
};

Run run(const std::string& arguments, const fs::path& scratch) {
    fs::path log = scratch / "cli.log";
    int raw = std::system((std::string(OPENV6_RENDER_CLI) + " " + arguments + " > " + log.string() + " 2>&1").c_str());
    return {WIFEXITED(raw) ? WEXITSTATUS(raw) : -1, read_text(log)};
}

}

int main() {
    const char* oracle = std::getenv("OPENV6_ORACLE");
    if (!oracle) {
        std::puts("OPENV6_ORACLE unset; skipping");
        return 77;
    }
    fs::path root(oracle), scores = root.parent_path() / "scores";
    fs::path scratch = fs::temp_directory_path() / ("openv6-cli-test-" + std::to_string(std::rand()));
    fs::create_directories(scratch);

    auto meta = openv6::json::Document::parse(read_text(root / "a_ja_short" / "meta.json"));
    std::string package(meta->root().get_property("package").get_string());
    std::string score = (scores / "a_ja_short.json").string(), wav = (scratch / "out.wav").string();

    auto ok = run(package + " " + score + " " + wav + " --f0 " + (scratch / "f0.json").string() + " --positions " + (scratch / "positions.json").string(), scratch);
    CHECK_MSG(ok.status == 0, ok.output);
    CHECK_MSG(ok.output.find("real time") != std::string::npos, ok.output);
    CHECK_MSG(read_text(wav) == read_text(root / "a_ja_short" / "out.wav"), "CLI WAV differs from the oracle's WAV");
    CHECK(read_text(scratch / "f0.json").starts_with("{\"frameMs\":5,\"pitches\":["));
    auto positions = openv6::json::Document::parse(read_text(scratch / "positions.json"));
    auto expected = openv6::json::Document::parse(read_text(root / "a_ja_short" / "positions.json"));
    CHECK(positions->root().array_length() == expected->root().array_length() && positions->root()[0][0].get_int32() == expected->root()[0][0].get_int32());

    std::ofstream(scratch / "lyrics.json") << R"({"tempoBpm":120,"languageId":0,"notes":[
        {"startTick":0,"durationTick":480,"pitch":60,"lyric":"あ"},{"startTick":480,"durationTick":480,"pitch":62,"lyric":"さ"},
        {"startTick":960,"durationTick":480,"pitch":64,"lyric":"+"},{"startTick":1440,"durationTick":480,"pitch":67,"lyric":"R"},
        {"startTick":1920,"durationTick":480,"pitch":65,"lyric":"か"}]})";
    auto lyrics = run(package + " " + (scratch / "lyrics.json").string() + " " + (scratch / "lyrics.wav").string(), scratch);
    CHECK_MSG(lyrics.status == 0, lyrics.output);
    CHECK(fs::file_size(scratch / "lyrics.wav") > 44 + 4 * 44100);

    auto phonemes = run(package + " " + (scratch / "lyrics.json").string() + " --no-audio --phonemes " + (scratch / "phonemes.json").string() + " --positions " + (scratch / "lyrics-positions.json").string(), scratch);
    CHECK_MSG(phonemes.status == 0, phonemes.output);
    auto phoneme_rows = openv6::json::Document::parse(read_text(scratch / "phonemes.json"));
    auto position_rows = openv6::json::Document::parse(read_text(scratch / "lyrics-positions.json"));
    CHECK(phoneme_rows->root().array_length() == 4 && position_rows->root().array_length() == 4);
    CHECK(phoneme_rows->root()[0].get_property("startTick").get_int32() == 0 && phoneme_rows->root()[3].get_property("startTick").get_int32() == 1920);
    CHECK(!phoneme_rows->root()[0].get_property("phonemes").get_string().empty());

    auto speakers = run(package + " --list-speakers", scratch);
    CHECK_MSG(speakers.status == 0 && speakers.output.find("0: ") == 0, speakers.output);

    std::ofstream(scratch / "bad.json") << R"({"tempoBpm":120,"languageId":0,"notes":[{"startTick":0,"durationTick":480,"pitch":60,"lyric":"kka"}]})";
    auto bad = run(package + " " + (scratch / "bad.json").string() + " " + wav, scratch);
    CHECK_MSG(bad.status == 1 && bad.output.find("error: Note 1") == 0, bad.output);
    std::ofstream(scratch / "hint.json") << R"({"tempoBpm":120,"languageId":0,"notes":[{"startTick":0,"durationTick":480,"pitch":60,"lyric":"kka","phoneticHint":"k a"}]})";
    auto hint = run(package + " " + (scratch / "hint.json").string() + " " + wav, scratch);
    CHECK_MSG(hint.status == 0, hint.output);

    auto note_json = [](const std::string& extra) {
        return R"({"tempoBpm":120,"languageId":0,"notes":[{"startTick":0,"durationTick":1920,"pitch":60,"phonemes":"a")" + extra + "}]}";
    };
    std::ofstream(scratch / "plain.json") << note_json("");
    std::ofstream(scratch / "shaped.json") << note_json(R"(,"aiParams":{"pitchFine":1,"amplitudeWhole":0.9},"vibrato":{"depth":64,"rate":64,"duration":960,"ai":false})");
    auto plain = run(package + " " + (scratch / "plain.json").string() + " " + wav + " --f0 " + (scratch / "plain-f0.json").string(), scratch);
    auto shaped = run(package + " " + (scratch / "shaped.json").string() + " " + (scratch / "shaped.wav").string() + " --f0 " + (scratch / "shaped-f0.json").string(), scratch);
    CHECK_MSG(plain.status == 0 && shaped.status == 0, plain.output + shaped.output);
    CHECK(read_text(scratch / "plain-f0.json") != read_text(scratch / "shaped-f0.json"));
    CHECK(read_text(wav) != read_text(scratch / "shaped.wav"));
    std::ofstream(scratch / "range.json") << note_json(R"(,"aiParams":{"pitchFine":1.5})");
    auto range = run(package + " " + (scratch / "range.json").string() + " " + wav, scratch);
    CHECK_MSG(range.status == 1 && range.output.find("aiParams.pitchFine") != std::string::npos, range.output);

    auto rejected = [&](const std::string& body, const std::string& message) {
        std::ofstream(scratch / "invalid.json") << R"({"tempoBpm":120,"languageId":0,)" << body << "}";
        auto result = run(package + " " + (scratch / "invalid.json").string() + " " + wav, scratch);
        CHECK_MSG(result.status == 1 && result.output.find(message) != std::string::npos, body + ": " + result.output);
    };
    const std::string note = R"("notes":[{"startTick":0,"durationTick":480,"pitch":60,"phonemes":"a"}])";
    rejected(R"("notes":[{"startTick":4000000000000,"durationTick":480,"pitch":60,"phonemes":"a"}])", "exceeds the native tick range");
    rejected(R"("notes":[{"startTick":4611686018427387904,"durationTick":4611686018427387904,"pitch":60,"phonemes":"a"}])", "exceeds the native tick range");
    rejected(note + R"(,"pitch":[0,0],"pitchTicks":[0,100000000000])", "pitchTicks exceed the native tick range");
    rejected(note + R"(,"controllers":[{"name":"Dynamics","events":[{"tick":100000000000,"value":64}]}])", "exceed the native tick range");
    rejected(note + R"(,"breaths":[{"startTick":0,"durationTick":9223372036854775807}])", "breath exceeds the native tick range");
    rejected(R"("notes":[{"startTick":0,"durationTick":480,"pitch":60,"phonemes":"  ","lyric":"x"}])", "has no phonemes");
    auto speaker = run(package + " " + score + " " + wav + " --npidx 255", scratch);
    CHECK_MSG(speaker.status == 1 && speaker.output.find("Speaker row 255") != std::string::npos, speaker.output);

    std::ofstream(scratch / "dynamics.json") << R"({"tempoBpm":120,"languageId":0,"notes":[{"startTick":0,"durationTick":1920,"pitch":60,"phonemes":"a"}],
        "controllers":[{"name":"Dynamics","events":[{"tick":0,"value":20},{"tick":960,"value":127}]}]})";
    auto dynamics = run(package + " " + (scratch / "dynamics.json").string() + " " + (scratch / "dynamics.wav").string(), scratch);
    CHECK_MSG(dynamics.status == 0, dynamics.output);
    auto neutral = run(package + " " + (scratch / "plain.json").string() + " " + (scratch / "neutral.wav").string(), scratch);
    CHECK_MSG(neutral.status == 0, neutral.output);
    CHECK(read_text(scratch / "dynamics.wav") != read_text(scratch / "neutral.wav"));

    auto missing = run((scratch / "nowhere").string() + " " + score + " " + wav, scratch);
    CHECK_MSG(missing.status == 1 && missing.output.find("error:") == 0, missing.output);
    auto usage = run(package + " " + score + " " + wav + " --bogus", scratch);
    CHECK_MSG(usage.status == 2 && usage.output.find("usage:") != std::string::npos, usage.output);
    auto malformed = run(package + " " + (scratch / "cli.log").string() + " " + wav, scratch);
    CHECK_MSG(malformed.status == 1 && malformed.output.find("error:") == 0, malformed.output);

    fs::remove_all(scratch);
    if (g_failures) std::fprintf(stderr, "%d failures\n", g_failures);
    return g_failures ? 1 : 0;
}
