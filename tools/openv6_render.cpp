#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "openv6/local_voice.hpp"
#include "openv6/phonemizer.hpp"
#include "openv6/speakers.hpp"
#include "openv6/voice_package.hpp"

using namespace openv6;

namespace {

constexpr const char* Usage =
    "usage: openv6_render <voice-package> <score.json> <out.wav> [options]\n"
    "       openv6_render <voice-package> <score.json> --no-audio [options]\n"
    "       openv6_render <voice-package> --list-speakers\n"
    "\n"
    "Renders a VOCALOID6 voice package singing a score to a 44.1 kHz mono 32-bit float WAV.\n"
    "Notes without \"phonemes\" are converted from their \"lyric\" (JA/EN/ZH; ZH as pinyin).\n"
    "\n"
    "Per-note fields, all optional (a note is {startTick, durationTick, pitch, lyric, ...}):\n"
    "  phoneticHint         phonemes to use instead of the lyric's G2P, for notes without \"phonemes\"\n"
    "  aiParams             VOCALOID6's AI note parameters, each 0..1 with 0.5 neutral: pitchFine (+-100 cents),\n"
    "                       pitchDriftStart, pitchDriftEnd (0 pulls the note edge onto the note pitch, 1 doubles the drift),\n"
    "                       pitchScalingCenter, pitchScalingOrigin (scale the pitch deviation, the latter the whole note),\n"
    "                       pitchTransitionStart, pitchTransitionEnd (where the drift gives way), amplitudeWhole,\n"
    "                       amplitudeStart, amplitudeEnd (loudness ramps), vibratoLeadingDepth, vibratoFollowingDepth\n"
    "                       (vibrato strength over the first and second half)\n"
    "  vibrato              {depth 0..127 (0 = none, default), rate 0..127 (default 50), duration in ticks from the note\n"
    "                       end backwards (default 2/3 of the note), ai true|false (default true: cycles follow the\n"
    "                       pitch model)}\n"
    "\n"
    "Score-level fields, all optional:\n"
    "  controllers          [{name, events: [{tick, value}, ...]}, ...], ticks strictly increasing, each value held until the next\n"
    "                       event. Names: Expression (0..100, 50 neutral), Dynamics (0..127, 64 neutral, 127 is +7.65 dB),\n"
    "                       S5Air (-100..100) and S5Character (-64..63), both 0 neutral\n"
    "\n"
    "options:\n"
    "  --npidx N            override the voice's NPIdx (timbre row | pitch row << 8)\n"
    "  --speaker ROW        render with timbre row ROW (see --list-speakers)\n"
    "  --f0 FILE            write the part's f0 as JSON {\"frameMs\":5,\"pitches\":[MIDI note, -1 unvoiced]}\n"
    "  --positions FILE     write VSM's automatic phoneme positions per note (ticks from the note start) as JSON\n"
    "  --native-dir DIR     directory holding G2PAManager_ENG.dll, for English lyrics (default: $OPENV6_NATIVE, else the bundled native/)\n"
    "  --phonemes FILE      write the phonemes of each note that remains after phonemizing (rests and breaths drop out) as\n"
    "                       JSON [{\"startTick\":N,\"phonemes\":\"...\"}, ...], in the order of --positions' rows\n"
    "  --no-audio           do not render; <out.wav> is omitted (--positions and --phonemes still work)\n"
    "  --list-speakers      print the voice's speaker rows and exit\n";

struct Options {
    std::string package, score, output, f0, positions, phonemes, native_dir;
    std::optional<int> np_idx, speaker;
    bool list_speakers = false, no_audio = false;
};

struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

int integer(const std::string& option, const std::string& value) {
    try {
        size_t used;
        int result = std::stoi(value, &used);
        if (used == value.size()) return result;
    } catch (const std::exception&) {
    }
    throw UsageError(option + " expects an integer, got '" + value + "'.");
}

Options parse_arguments(int argc, char** argv) {
    Options options;
    const char* native = std::getenv("OPENV6_NATIVE");
    options.native_dir = native ? native : OPENV6_NATIVE_DIR;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw UsageError(arg + " needs a value.");
            return argv[++i];
        };
        if (arg == "--f0") options.f0 = value();
        else if (arg == "--positions") options.positions = value();
        else if (arg == "--phonemes") options.phonemes = value();
        else if (arg == "--no-audio") options.no_audio = true;
        else if (arg == "--native-dir") options.native_dir = value();
        else if (arg == "--npidx") options.np_idx = integer(arg, value());
        else if (arg == "--speaker") options.speaker = integer(arg, value());
        else if (arg == "--list-speakers") options.list_speakers = true;
        else if (arg.starts_with("--")) throw UsageError("Unknown option " + arg + ".");
        else positional.push_back(arg);
    }
    size_t expected = options.list_speakers ? 1 : options.no_audio ? 2 : 3;
    if (positional.size() != expected)
        throw UsageError(options.list_speakers ? "--list-speakers takes only the voice package."
                         : options.no_audio    ? "Expected <voice-package> <score.json> with --no-audio."
                                               : "Expected <voice-package> <score.json> <out.wav>.");
    if (options.no_audio && !options.f0.empty()) throw UsageError("--f0 needs the audio render.");
    options.package = positional[0];
    if (!options.list_speakers) {
        options.score = positional[1];
        if (!options.no_audio) options.output = positional[2];
    }
    if (options.np_idx && options.speaker) throw UsageError("--npidx and --speaker are exclusive.");
    return options;
}

void write_text(const std::string& path, const std::string& text) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw InvalidDataError("Cannot write " + path + ".");
    stream << text;
}

void write_wav(const std::string& path, const std::vector<float>& samples, int rate) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw InvalidDataError("Cannot write " + path + ".");
    uint32_t bytes = static_cast<uint32_t>(samples.size() * 4);
    auto u32 = [&](uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16 = [&](uint16_t v) { out.write(reinterpret_cast<const char*>(&v), 2); };
    out.write("RIFF", 4);
    u32(36 + bytes);
    out.write("WAVEfmt ", 8);
    u32(16);
    u16(3);
    u16(1);
    u32(rate);
    u32(rate * 4);
    u16(4);
    u16(32);
    out.write("data", 4);
    u32(bytes);
    out.write(reinterpret_cast<const char*>(samples.data()), bytes);
    if (!out) throw InvalidDataError("Cannot write " + path + ".");
}

std::string number_text(double v) {
    char buffer[40];
    std::snprintf(buffer, sizeof buffer, "%.9g", v);
    return buffer;
}

std::string json_string(const std::string& value) {
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') result += std::string("\\") + static_cast<char>(c);
        else if (c < 0x20) result += " ";
        else result += static_cast<char>(c);
    }
    return result + "\"";
}

void write_positions(const std::string& path, const std::vector<std::vector<int>>& positions) {
    std::string text = "[";
    for (size_t i = 0; i < positions.size(); i++) {
        text += (i ? ",\n[" : "[");
        for (size_t k = 0; k < positions[i].size(); k++) text += (k ? "," : "") + std::to_string(positions[i][k]);
        text += "]";
    }
    write_text(path, text + "]\n");
}

void print_speakers(const VoicePackage& package) {
    auto rows = speakers::available(package.registration);
    if (rows.empty()) {
        std::printf("No speaker table for timbre model '%s'; the voice renders with its own NPIdx (%d).\n", package.registration.timbre_model.c_str(),
                    package.registration.np_idx);
        return;
    }
    for (const auto& row : rows) std::printf("%s\n", row.name.c_str());
}

int run(const Options& options) {
    auto package = load_voice_package(options.package);
    if (options.list_speakers) {
        print_speakers(package);
        return 0;
    }
    std::optional<int> np_idx = options.np_idx;
    if (options.speaker) np_idx = speakers::np_idx(package.registration, *options.speaker);

    V6Score score = parse_score(read_text(options.score));
    std::ranges::stable_sort(score.notes, {}, &V6ScoreNote::startTick);
    validate_score(score, false);
    phonemizer::phonemize_score(score, options.native_dir);

    if (!options.phonemes.empty()) {
        std::string text = "[";
        for (size_t i = 0; i < score.notes.size(); i++)
            text += (i ? ",\n" : "") + std::string("{\"startTick\":") + std::to_string(score.notes[i].startTick) + ",\"phonemes\":\"" + score.notes[i].phonemes + "\"}";
        write_text(options.phonemes, text + "]\n");
    }
    if (options.no_audio && options.positions.empty()) return 0;

    auto start = std::chrono::steady_clock::now();
    V6LocalVoice voice(package.model_directory(), package.registration, package.config.language_id, np_idx);
    auto loaded = std::chrono::steady_clock::now();
    if (options.no_audio) {
        write_positions(options.positions, voice.phoneme_positions(score));
        return 0;
    }
    auto rendered = voice.render(score);
    auto done = std::chrono::steady_clock::now();
    write_wav(options.output, rendered.samples, V6Synthesizer::SampleRate);

    if (!options.f0.empty()) {
        std::string text = "{\"frameMs\":5,\"pitches\":[";
        for (size_t i = 0; i < rendered.pitch.pitches.size(); i++) text += (i ? "," : "") + number_text(rendered.pitch.pitches[i]);
        write_text(options.f0, text + "]}\n");
    }
    if (!options.positions.empty()) write_positions(options.positions, voice.phoneme_positions(score));

    double load_seconds = std::chrono::duration<double>(loaded - start).count();
    double render_seconds = std::chrono::duration<double>(done - loaded).count();
    double audio_seconds = static_cast<double>(rendered.samples.size()) / V6Synthesizer::SampleRate;
    std::printf("%s: %.2f s of audio, loaded in %.2f s, rendered in %.2f s (%.1fx real time)\n", options.output.c_str(), audio_seconds, load_seconds,
                render_seconds, audio_seconds / render_seconds);
    return 0;
}

}

int main(int argc, char** argv) {
    try {
        Options options = parse_arguments(argc, argv);
        return run(options);
    } catch (const UsageError& e) {
        std::fprintf(stderr, "error: %s\n\n%s", e.what(), Usage);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
