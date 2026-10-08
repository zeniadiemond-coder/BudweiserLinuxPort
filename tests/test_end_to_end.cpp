#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numbers>

#include "audio_support.hpp"
#include "check.hpp"
#include "openv6/fft.hpp"
#include "openv6/frame_driver.hpp"
#include "openv6/local_voice.hpp"
#include "openv6/phonemizer.hpp"
#include "openv6/speex_resampler.hpp"
#include "openv6/synthesizer.hpp"
#include "openv6/vsm_pitch.hpp"
#include "openv6/voice_package.hpp"
#include "timing_support.hpp"

using namespace openv6;
using namespace timing_support;
using namespace audio_support;
namespace fs = std::filesystem;

namespace {

constexpr int AsaNpIdx = 0xa5a5, AsaParamExp = 1200;
constexpr double MaxBandDistance = 0.65;
constexpr const char* AsaBank = "BLKEA452HDSKXLD7";

double since(std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); }

std::unique_ptr<json::Document> read_json(const fs::path& path) { return json::Document::parse(read_text(path)); }

struct Difference {
    size_t mismatches = 0;
    float max = 0;
    bool same_size = true;
};

Difference compare_floats(std::span<const float> a, std::span<const float> b) {
    Difference d;
    d.same_size = a.size() == b.size();
    for (size_t i = 0; i < std::min(a.size(), b.size()); i++) {
        if (std::memcmp(&a[i], &b[i], 4) == 0) continue;
        d.mismatches++;
        d.max = std::max(d.max, std::fabs(a[i] - b[i]));
    }
    return d;
}

fs::path score_path(const fs::path& scores, const std::string& name) {
    static const StringMap<std::string> Aliases = {
        {"f_en_allen", "f_en_asa"}, {"f_en_allen_npidx257", "f_en_asa"}, {"a_ja_allen_cross", "a_ja_short"},
        {"a_ja_asahi", "a_ja_short"}, {"a_ja_asa_npidx257", "a_ja_short"},
    };
    auto alias = Aliases.find(name);
    return scores / ((alias != Aliases.end() ? alias->second : name) + ".json");
}

bool same_notes(const V6Note& a, const V6Note& b) {
    return a.start == b.start && a.end == b.end && a.pitch == b.pitch && a.phonemes == b.phonemes && a.meter == b.meter &&
           a.position == b.position && a.flag == b.flag && a.language == b.language && a.phoneme_ticks == b.phoneme_ticks;
}

struct OracleTotals {
    size_t cases = 0, inexact = 0;
    float worst_samples = 0, worst_pitch = 0;
    double cpp_seconds = 0, csharp_seconds = 0;
};

void oracle_case(const fs::path& dir, const fs::path& scores, OracleTotals& totals) {
    std::string name = dir.filename().string();
    auto meta_doc = read_json(dir / "meta.json");
    json::Value meta = meta_doc->root();
    auto package = load_voice_package(std::string(meta.get_property("package").get_string()));
    std::optional<int> np_idx;
    if (meta.get_property("npIdx").kind() == json::Kind::Number) np_idx = meta.get_property("npIdx").get_int32();
    auto start = std::chrono::steady_clock::now();
    V6LocalVoice voice(package.model_directory(), package.registration, package.config.language_id, np_idx);
    double load_seconds = since(start);

    auto score_doc = read_json(score_path(scores, name));
    V6Score score = parse_score(score_doc->root());
    if (score.controllers) std::erase_if(*score.controllers, [](const V6Controller& c) { return c.name == "Dynamics"; });
    double tick = 60.0 / (score.tempoBpm * 480);

    auto notes_doc = read_json(dir / "notes.json");
    auto expected_notes = oracle_notes(notes_doc->root());
    auto built = build_notes(score, tick);
    bool notes_same = built.notes.size() == expected_notes.size();
    for (size_t i = 0; notes_same && i < expected_notes.size(); i++) notes_same = same_notes(built.notes[i], expected_notes[i]);
    CHECK_MSG(notes_same, name + ": notes differ from notes.json");

    start = std::chrono::steady_clock::now();
    auto rendered = voice.render(score);
    double seconds = since(start);
    auto samples = compare_floats(rendered.samples, read_floats(dir / "samples.f32"));
    auto pitch = compare_floats(rendered.pitch.pitches, read_floats(dir / "pitch.f32"));
    CHECK_MSG(samples.same_size && samples.mismatches == 0, name + ": samples differ (" + std::to_string(samples.mismatches) + " values, max " + std::to_string(samples.max) + ")");
    CHECK_MSG(pitch.same_size && pitch.mismatches == 0, name + ": pitch differs (" + std::to_string(pitch.mismatches) + " values, max " + std::to_string(pitch.max) + ")");

    auto positions_doc = read_json(dir / "positions.json");
    auto positions = voice.phoneme_positions(score);
    bool positions_same = positions.size() == positions_doc->root().array_length();
    for (size_t i = 0; positions_same && i < positions.size(); i++) {
        json::Value row = positions_doc->root()[i];
        positions_same = row.array_length() == positions[i].size();
        for (size_t k = 0; positions_same && k < positions[i].size(); k++) positions_same = row[k].get_int32() == positions[i][k];
    }
    CHECK_MSG(positions_same, name + ": phoneme positions differ from positions.json");

    double csharp = meta.get_property("renderSeconds").get_double();
    std::printf("%-32s samples %s (%zu, max %.3g)  pitch %s  positions %s  notes %s   %.2fs vs C# %.2fs (%.2fx), load %.2fs\n", name.c_str(),
                samples.mismatches == 0 && samples.same_size ? "exact" : "DIFFER", samples.mismatches, samples.max,
                pitch.mismatches == 0 && pitch.same_size ? "exact" : "DIFFER", positions_same ? "exact" : "DIFFER", notes_same ? "exact" : "DIFFER", seconds,
                csharp, csharp / seconds, load_seconds);
    totals.cases++;
    if (samples.mismatches || !samples.same_size || pitch.mismatches || !pitch.same_size || !positions_same || !notes_same) totals.inexact++;
    totals.worst_samples = std::max(totals.worst_samples, samples.max);
    totals.worst_pitch = std::max(totals.worst_pitch, pitch.max);
    totals.cpp_seconds += seconds;
    totals.csharp_seconds += csharp;
}

std::optional<V6PitchEdit> capture_pitch_edit(json::Value score) {
    json::Value pitch;
    if (!score.try_get_property("pitch", pitch)) return std::nullopt;
    double tick = capture_tick(score), origin = score.get_property("origin").get_double();
    V6PitchEdit edit{{}, {}, tick, origin};
    for (json::Value p : pitch.enumerate_array()) {
        edit.ticks.push_back(static_cast<int64_t>(round_even((p[0].get_double() - origin) / tick)));
        edit.midi_cents.push_back(p[1].get_single());
    }
    return edit;
}

std::shared_ptr<ModelFile> model(const fs::path& path) { return std::make_shared<ModelFile>(ModelFile::load(path.string())); }

void driver_matches_capture(const fs::path& dir) {
    auto doc = read_json(dir / "score.json");
    auto timed = capture_phonemes(doc->root().get_property("timed"));
    auto gate = frame_driver::gate(timed, 3220, false);
    constexpr int frames = 2891;
    auto mixed = frame_driver::mix(
        read_floats(dir / "ch0.f32"), read_floats(dir / "ch1.f32"), frames, [](int) { return 0.f; }, [](int) { return 1.f; },
        [&](int f) { return gate[f]; }, [](int) { return 1.f; });
    std::vector<float> part(static_cast<size_t>(round_even(16.1 * 32000)));
    std::copy(mixed.begin(), mixed.end(), part.begin());
    auto audio = speex_resampler::resample(part, 32000, 44100, static_cast<int>(round_even(16.1 * 44100)));
    auto pcm = read_pcm16(dir / "morning.wav");
    float worst = 0;
    for (size_t i = 0; i < pcm.size(); i++) worst = std::max(worst, std::fabs(audio[i] - pcm[i]));
    CHECK_MSG(worst < 2.5f / 32768, "mixed and resampled audio differs from the native WAV by " + std::to_string(worst));
    std::printf("driver mix + resample vs native WAV: max %.3g (limit %.3g)\n", worst, 2.5f / 32768);
}

float f0_error(std::span<const float> got, std::span<const float> expected, bool saw_scale = false) {
    float worst = 0;
    for (size_t i = 0; i < std::min(got.size(), expected.size()); i++) {
        float value = got[i];
        bool gap = saw_scale ? value < 0 : value == vsm_pitch::Gap;
        if (gap != (expected[i] == vsm_pitch::Gap)) return INFINITY;
        if (!gap) worst = std::max(worst, std::fabs((saw_scale ? value * 100 - 6900 : value) - expected[i]));
    }
    return worst;
}

void end_to_end_matches_capture(const std::string& bundle, const fs::path& dir) {
    auto doc = read_json(dir / "score.json");
    json::Value score = doc->root();
    auto notes = capture_notes(score);
    V6Synthesizer synth(model(dir / "asa.vtb2"), model(dir / "asa.vpit"), std::make_shared<VtmgModel>(VtmgModel::load((dir / "VOCALOID6.vtmg").string())),
                        AsaNpIdx, AsaParamExp);
    auto timed = synth.time(notes, score.get_property("condition").get_int32(), capture_tick(score));
    std::vector<TimeSpan> breaths;
    json::Value value;
    if (score.try_get_property("breaths", value) && value.get_boolean()) breaths.push_back({-1.0, notes.back().end + 1});
    int exhalation = score.try_get_property("exhalation", value) ? value.get_int32() : 5;
    auto edit = capture_pitch_edit(score);
    auto start = std::chrono::steady_clock::now();
    auto result = synth.render(notes, timed, {}, edit ? &*edit : nullptr, breaths, exhalation);
    double seconds = since(start);
    auto native_f0 = read_floats(dir / "part_f0.f32");
    float f0 = f0_error(result.f0, native_f0);
    CHECK_MSG(f0 < 5.f, bundle + ": end-to-end f0 differs by " + std::to_string(f0) + " cents");
    auto pcm = read_pcm16(dir / "out.wav");
    CHECK_MSG(pcm.size() == result.samples.size(), bundle + ": length " + std::to_string(result.samples.size()) + " vs " + std::to_string(pcm.size()));
    double distance = band_distance(result.samples, pcm);
    CHECK_MSG(distance < MaxBandDistance, bundle + ": band spectra differ by " + std::to_string(distance) + " dB (median)");
    std::printf("%-28s f0 %.3g cents, band distance %.3f dB, %.2fs\n", bundle.c_str(), f0, distance, seconds);
}

fs::path voice_dir;

V6LocalVoice asa_voice() { return V6LocalVoice(voice_dir.string(), V6Registration::read(voice_dir.string(), AsaBank), 0); }

void local_voice_matches_capture(const std::string& bundle, const fs::path& dir) {
    V6Score score = parse_score(read_text(dir / "phrase.json"));
    auto phrase_doc = read_json(dir / "phrase.json");
    json::Value ignored;
    if (phrase_doc->root().try_get_property("breathMode", ignored)) score.breaths = std::vector<V6Breath>{{-1000, 1'000'000}};
    auto result = asa_voice().render(score);
    auto native_f0 = read_floats(dir / "part_f0.f32");
    CHECK_MSG(native_f0.size() == result.pitch.pitches.size(), bundle + ": pitch length");
    float f0 = f0_error(result.pitch.pitches, native_f0, true);
    CHECK_MSG(f0 < 5.f, bundle + ": f0 differs by " + std::to_string(f0) + " cents");
    auto pcm = read_pcm16(dir / "out.wav");
    CHECK_MSG(pcm.size() == result.samples.size(), bundle + ": length");
    double distance = band_distance(result.samples, pcm);
    CHECK_MSG(distance < MaxBandDistance, bundle + ": band spectra differ by " + std::to_string(distance) + " dB (median)");
    std::printf("local voice %-28s f0 %.3g cents, band distance %.3f dB\n", bundle.c_str(), f0, distance);
}

int64_t native_tick(double ms) { return static_cast<int64_t>(round_away(ms * 0.96)); }

void phoneme_positions_match_native_phonemizer(const fs::path& timing_file) {
    auto voice = asa_voice();
    auto doc = read_json(timing_file);
    for (auto member : doc->root().enumerate_object()) {
        V6Score score{};
        score.tempoBpm = 120;
        std::vector<std::vector<int>> expected;
        json::Value groups = member.value.get_property("request"), response = member.value.get_property("response");
        for (size_t g = 0; g < groups.array_length(); g++) {
            json::Value spans = groups[g].get_property("notes"), results = response[g];
            for (size_t n = 0; n < spans.array_length(); n++) {
                int64_t start = native_tick(spans[n].get_property("startMs").get_double());
                int64_t end = native_tick(spans[n].get_property("endMs").get_double());
                score.notes.push_back({start, end - start, spans[n].get_property("pitch").get_int32(), "",
                                       std::string(results[n].get_property("Phonemes").get_string()), groups[g].get_property("languageId").get_int32(), std::nullopt, std::nullopt, {}, std::nullopt});
                std::vector<int> row;
                for (json::Value p : results[n].get_property("Positions").enumerate_array()) row.push_back(p.get_int32());
                expected.push_back(std::move(row));
            }
        }
        auto positions = voice.phoneme_positions(score);
        CHECK_MSG(positions == expected, std::string(member.name) + ": phoneme positions differ from the native phonemizer");
    }
    std::puts("phoneme positions vs native phonemizer: checked");
}

void long_silence_note_keeps_one_position() {
    V6Score score{};
    score.tempoBpm = 120;
    score.notes = {{480, 480, 60, "", "k a", std::nullopt, std::nullopt, std::nullopt, {}, std::nullopt},
                   {960, 1920, 60, "", "Sil", std::nullopt, std::nullopt, std::nullopt, {}, std::nullopt},
                   {2880, 480, 60, "", "k a", std::nullopt, std::nullopt, std::nullopt, {}, std::nullopt}};
    auto positions = asa_voice().phoneme_positions(score);
    CHECK(positions[1].size() == 2);
    CHECK(positions[2][0] == positions[1][1] - 1920);
}

std::string native_directory() {
    const char* native = std::getenv("OPENV6_NATIVE");
    return native ? native : OPENV6_NATIVE_DIR;
}

void phonemizer_matches_native_phonemizer(const fs::path& timing_file) {
    auto doc = read_json(timing_file);
    std::string native = native_directory();
    bool english = fs::exists(fs::path(native) / "G2PAManager_ENG.dll");
    for (auto member : doc->root().enumerate_object()) {
        std::string name(member.name);
        if (name.starts_with("en") && !english) {
            std::printf("phonemizer %s: skipped (no English dictionary in %s)\n", name.c_str(), native.c_str());
            continue;
        }
        json::Value request = member.value.get_property("request"), response = member.value.get_property("response");
        V6Score score{};
        score.tempoBpm = 120;
        std::vector<std::string> expected;
        for (size_t g = 0; g < request.array_length(); g++) {
            json::Value group = request[g];
            size_t n = 0;
            for (json::Value note : group.get_property("notes").enumerate_array()) {
                int64_t start = native_tick(note.get_property("startMs").get_double()), end = native_tick(note.get_property("endMs").get_double());
                std::string lyric = n == 0 ? std::string(group.get_property("lyric").get_string()) : note.get_property("extend").get_boolean() ? "+~" : "+";
                V6ScoreNote converted{start, end - start, note.get_property("pitch").get_int32(), lyric, "", group.get_property("languageId").get_int32(),
                                      std::nullopt, std::nullopt, {}, std::nullopt};
                score.notes.push_back(std::move(converted));
                expected.push_back(std::string(response[g][n].get_property("Phonemes").get_string()));
                n++;
            }
        }
        phonemizer::phonemize_score(score, native);
        bool same = score.notes.size() == expected.size();
        for (size_t i = 0; same && i < expected.size(); i++) same = score.notes[i].phonemes == expected[i];
        CHECK_MSG(same, name + ": phonemes differ from the native phonemizer");
        std::printf("phonemizer %s: %s the native phonemizer\n", name.c_str(), same ? "matches" : "DIFFERS from");
    }
}

void lyric_only_scores_match_phonemes(const fs::path& scores) {
    std::string native = native_directory();
    for (const char* name : {"a_ja_short", "cap_extend", "e_extend_longrest", "d_breaths"}) {
        auto doc = read_json(scores / (std::string(name) + ".json"));
        V6Score expected = parse_score(doc->root());
        V6Score score = expected;
        for (auto& note : score.notes) {
            if (note.phonemes == "-") note.lyric = "+";
            note.phonemes.clear();
        }
        phonemizer::phonemize_score(score, native);
        bool same = score.notes.size() == expected.notes.size();
        for (size_t i = 0; same && i < expected.notes.size(); i++)
            same = score.notes[i].phonemes == expected.notes[i].phonemes && score.notes[i].languageId == expected.languageId;
        CHECK_MSG(same, std::string(name) + ": phonemes from lyrics differ from the score's");
        std::printf("lyric-only %s: phonemes %s\n", name, same ? "match" : "DIFFER");
    }

    V6Score score{};
    score.tempoBpm = 120;
    score.languageId = 0;
    auto note = [](int64_t start, const char* lyric) { return V6ScoreNote{start, 480, 60, lyric, "", std::nullopt, std::nullopt, std::nullopt, {}, std::nullopt}; };
    score.notes = {note(0, "あ"), note(480, "R"), note(960, "br"), note(1440, "か"), note(1920, "+"), note(2400, "き_C4")};
    phonemizer::phonemize_score(score, native);
    CHECK(score.notes.size() == 4);
    CHECK(score.notes[0].phonemes == "a" && score.notes[1].phonemes == "k a" && score.notes[2].phonemes == "-" && score.notes[3].phonemes == "k' i");
    CHECK(score.breaths && score.breaths->size() == 1 && (*score.breaths)[0].startTick == 960);
    V6Score bad = score;
    bad.notes = {note(0, "kka")};
    bool threw = false;
    try {
        phonemizer::phonemize_score(bad, native);
    } catch (const InvalidDataError& e) {
        threw = std::string(e.what()).find("Note 1") != std::string::npos;
    }
    CHECK_MSG(threw, "an unconvertible lyric must throw naming the note");
}

void clean_lyric_cases() {
    CHECK(phonemizer::clean_lyric(0, "a か") == "か");
    CHECK(phonemizer::clean_lyric(0, "- か") == "か");
    CHECK(phonemizer::clean_lyric(0, "か_C4") == "か");
    CHECK(phonemizer::clean_lyric(0, "か強") == "か");
    CHECK(phonemizer::clean_lyric(1, "hello_C#4") == "hello");
    CHECK(phonemizer::clean_lyric(1, "hello↑") == "hello");
    CHECK(!phonemizer::clean_lyric(0, "R"));
    CHECK(!phonemizer::clean_lyric(0, "っ"));
    CHECK(!phonemizer::clean_lyric(1, "  "));
    CHECK(phonemizer::is_rest(0, std::nullopt, "sil"));
    CHECK(!phonemizer::is_rest(0, "a", "sil"));
    CHECK(phonemizer::is_breath(" Br "));
    CHECK(phonemizer::is_vowel_extension("+~") && phonemizer::is_vowel_extension("+*") && !phonemizer::is_vowel_extension("+"));
}

}

int main() {
    clean_lyric_cases();
    const char* reference = reference_dir();
    const char* oracle = std::getenv("OPENV6_ORACLE");
    if (!reference && !oracle) {
        std::puts("OPENV6_REFERENCE and OPENV6_ORACLE unset; skipping");
        return g_failures ? 1 : 77;
    }
    if (reference) {
        fs::path root = reference;
        voice_dir = root / "voice-asa";
        driver_matches_capture(root / "reference");
        for (const char* bundle : {"reference-morning-pitch", "reference-english", "reference-chinese", "reference-edits", "reference-timing-edits",
                                   "reference-breath", "reference-longrest-breath", "reference-extend", "reference-english-extend",
                                   "reference-rest-extend", "reference-exhale9"})
            end_to_end_matches_capture(bundle, root / bundle);
        for (const char* bundle : {"reference-morning-pitch", "reference-english", "reference-chinese", "reference-edits", "reference-rest-edit",
                                   "reference-longrest-breath", "reference-extend", "reference-english-extend", "reference-rest-extend",
                                   "reference-lead-ja1", "reference-lead-en1", "reference-lead-zh_ext"})
            local_voice_matches_capture(bundle, root / bundle);
        phoneme_positions_match_native_phonemizer(root / "phonemize-timing.json");
        long_silence_note_keeps_one_position();
        phonemizer_matches_native_phonemizer(root / "phonemize-timing.json");
    }
    if (oracle) {
        OracleTotals totals;
        fs::path root(oracle);
        std::vector<fs::path> cases;
        for (const auto& entry : fs::directory_iterator(root))
            if (fs::exists(entry.path() / "samples.f32")) cases.push_back(entry.path());
        std::sort(cases.begin(), cases.end());
        for (const auto& dir : cases) {
            try {
                oracle_case(dir, root.parent_path() / "scores", totals);
            } catch (const std::exception& e) {
                CHECK_MSG(false, dir.filename().string() + ": " + e.what());
            }
        }
        lyric_only_scores_match_phonemes(root.parent_path() / "scores");
        std::printf("oracle: %zu cases, %zu not bit-exact, max sample error %.3g, max pitch error %.3g; total %.2fs vs C# %.2fs\n", totals.cases,
                    totals.inexact, totals.worst_samples, totals.worst_pitch, totals.cpp_seconds, totals.csharp_seconds);
    }
    if (g_failures) std::fprintf(stderr, "%d failures\n", g_failures);
    return g_failures ? 1 : 0;
}
