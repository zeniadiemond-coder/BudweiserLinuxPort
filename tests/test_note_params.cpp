#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

#include "audio_support.hpp"
#include "check.hpp"
#include "openv6/frame_driver.hpp"
#include "openv6/local_voice.hpp"
#include "openv6/vsm_pitch.hpp"

using namespace openv6;
using namespace audio_support;
namespace fs = std::filesystem;

namespace {

constexpr float NoteStageTolerance = 1e-3f, ExpressionTolerance = 1e-6f, F0Tolerance = 2e-3f, EndToEndCents = 5.f;
constexpr double MaxBandDistance = 0.8;
constexpr int AsaParamExpOffset = -1800;
constexpr const char* AsaBank = "BLKEA452HDSKXLD7";

struct Record {
    std::string name;
    int phase;
    std::string label;
    std::string data;
};

std::string trimmed(const char* text, size_t size) {
    std::string s(text, size);
    s.erase(s.find_last_not_of(std::string(" \0", 2)) + 1);
    return s;
}

std::vector<Record> read_records(const fs::path& path) {
    std::string bytes = timing_support::read_text(path);
    std::vector<Record> records;
    for (size_t o = 0; o + 28 <= bytes.size();) {
        Record r;
        int size;
        std::memcpy(&r.phase, &bytes[o + 8], 4);
        std::memcpy(&size, &bytes[o + 24], 4);
        r.name = trimmed(&bytes[o], 8);
        r.label = trimmed(&bytes[o + 12], 12);
        r.data = bytes.substr(o + 28, size);
        records.push_back(std::move(r));
        o += 28 + size;
    }
    return records;
}

template <class T>
T at(const std::string& data, size_t offset) {
    T value;
    std::memcpy(&value, data.data() + offset, sizeof value);
    return value;
}

std::vector<float> floats(const std::string& data) {
    std::vector<float> result(data.size() / 4);
    std::memcpy(result.data(), data.data(), result.size() * 4);
    return result;
}

const Record& find(const std::vector<Record>& records, const std::string& name, const std::string& label, int occurrence = 0) {
    for (const auto& r : records)
        if (r.name == name && r.label == label && occurrence-- == 0) return r;
    throw std::runtime_error("no record " + name + "/" + label);
}

struct NoteRecord {
    int lo, hi, first200, last200, midi, depth, rate;
    bool ai_vibrato;
    int64_t vibrato_start;
    float offset;
    AiNoteExpression expression;
};

std::vector<NoteRecord> note_records(const std::vector<Record>& vsm) {
    const std::string& data = find(vsm, "850b0", "notes").data;
    std::vector<NoteRecord> notes;
    for (size_t o = 0; o + 0xa0 <= data.size(); o += 0xa0) {
        NoteRecord n;
        n.lo = static_cast<int>(at<int64_t>(data, o + 0x18));
        n.hi = static_cast<int>(at<int64_t>(data, o + 0x20));
        n.first200 = static_cast<int>(at<int64_t>(data, o + 0x28));
        n.last200 = static_cast<int>(at<int64_t>(data, o + 0x30));
        n.depth = at<int>(data, o + 0x38);
        n.rate = at<int>(data, o + 0x3c);
        n.ai_vibrato = at<uint8_t>(data, o + 0x40) != 0;
        n.vibrato_start = at<int64_t>(data, o + 0x48);
        n.midi = at<int>(data, o + 0x50);
        n.offset = at<float>(data, o + 0x54);
        std::memcpy(&n.expression, data.data() + o + 0x58, sizeof n.expression);
        notes.push_back(n);
    }
    return notes;
}

struct Worst {
    float value = 0;
    void add(float v) { value = std::max(value, v); }
};

void note_stage_matches_capture(const std::string& name, const fs::path& dir, Worst& worst) {
    auto vsm = read_records(dir / "vsm.bin");
    AiNoteExpression expression;
    int note = 0, start = 0, depth = 0, rate = 0;
    bool ai = false;
    std::vector<float> pairs;
    int calls = 0;
    float worst_pitch = 0, worst_vibrato = 0;
    for (const auto& r : vsm) {
        if (r.name != "f6380") continue;
        if (r.label == "args") {
            note = at<int>(r.data, 0);
            start = static_cast<int>(at<int64_t>(r.data, 4));
        } else if (r.label == "pairs") {
            pairs = floats(r.data);
        } else if (r.label == "vibrato") {
            depth = at<int>(r.data, 0);
            rate = at<int>(r.data, 4);
            ai = at<uint8_t>(r.data, 8) != 0;
        } else if (r.label == "expression") {
            std::memcpy(&expression, r.data.data(), sizeof expression);
        } else if (r.label == "output") {
            auto reference = floats(r.data);
            size_t n = pairs.size() / 2;
            std::vector<float> pitch(n), residual(n);
            for (size_t i = 0; i < n; i++) {
                pitch[i] = pairs[2 * i];
                residual[i] = pairs[2 * i + 1];
            }
            auto curve = vsm_pitch::note_pitch(pitch, residual, note, expression);
            std::vector<float> with_vibrato = curve;
            vsm_pitch::add_vibrato(with_vibrato, residual, depth, rate, ai, start, expression);
            CHECK(reference.size() == n * 3);
            for (size_t i = 0; i < n; i++) {
                worst_pitch = std::max(worst_pitch, std::fabs(curve[i] - reference[3 * i]));
                worst_vibrato = std::max(worst_vibrato, std::fabs(with_vibrato[i] - reference[3 * i + 1]));
            }
            calls++;
        }
    }
    std::printf("note stage %-22s %2d notes: pitch %.3g, with vibrato %.3g cents\n", name.c_str(), calls, worst_pitch, worst_vibrato);
    CHECK_MSG(calls > 0, name + ": no f6380 calls");
    CHECK_MSG(worst_pitch < NoteStageTolerance && worst_vibrato < NoteStageTolerance, name + ": note stage differs from the native one");
    worst.add(std::max(worst_pitch, worst_vibrato));
}

void expression_matches_capture(const std::string& name, const fs::path& dir, Worst& worst) {
    auto vsm = read_records(dir / "vsm.bin");
    auto notes = note_records(vsm);
    auto reference = floats(find(vsm, "882c0", "vecb").data);
    std::vector<frame_driver::AmplitudeSpan> spans;
    for (const auto& n : notes)
        spans.push_back({n.first200, n.last200, n.expression.amplitude_whole, n.expression.amplitude_start, n.expression.amplitude_end});
    auto ramps = frame_driver::expression_ramps(spans, static_cast<int>(reference.size()));
    float error = 0;
    for (size_t f = 0; f < reference.size(); f++)
        error = std::max(error, std::fabs(frame_driver::control(50, AsaParamExpOffset, ramps[f]) - reference[f]));
    std::printf("expression %-22s %zu frames: max error %.3g\n", name.c_str(), reference.size(), error);
    CHECK_MSG(error < ExpressionTolerance, name + ": Expression curve differs from the native one");
    worst.add(error);
}

void f0_matches_capture_given_native_pitch_model(const std::string& name, const fs::path& dir, Worst& worst) {
    auto vsm = read_records(dir / "vsm.bin");
    auto calls = read_records(dir / "calls.bin");
    std::vector<float> pitch, residual;
    for (const auto& r : calls) {
        if (r.name != "2db29822" || r.phase != 1 || r.label != "output") continue;
        pitch.push_back(at<float>(r.data, 0));
        residual.push_back(at<float>(r.data, 4));
    }
    std::vector<float> reference;
    for (const auto& r : vsm)
        if (r.name == "850b0" && r.label == "output") reference = floats(r.data);
    std::vector<vsm_pitch::PitchNote> notes;
    for (const auto& n : note_records(vsm))
        notes.push_back({n.lo, n.hi, n.midi, n.offset, n.expression, n.depth, n.rate, n.ai_vibrato, static_cast<int>(n.vibrato_start)});
    auto f0 = vsm_pitch::f0_curve(pitch, residual, notes, static_cast<int>(reference.size()));
    float error = 0;
    size_t compared = 0, gap_mismatches = 0;
    for (size_t i = 0; i < reference.size(); i++) {
        bool gap = f0[i] == vsm_pitch::Gap;
        if (gap != (reference[i] == vsm_pitch::Gap)) gap_mismatches++;
        if (gap || reference[i] == vsm_pitch::Gap) continue;
        error = std::max(error, std::fabs(f0[i] - reference[i]));
        compared++;
    }
    std::printf("f0 %-30s %zu frames: max error %.3g cents, %zu gap mismatches\n", name.c_str(), compared, error, gap_mismatches);
    CHECK_MSG(error < F0Tolerance && gap_mismatches == 0, name + ": f0 differs from the native one");
    worst.add(error);
}

V6Score phrase_score(const fs::path& phrase) {
    std::string text = timing_support::read_text(phrase);
    V6Score score = parse_score(text);
    auto doc = json::Document::parse(text);
    size_t i = 0;
    for (auto n : doc->root().get_property("notes").enumerate_array()) {
        auto& note = score.notes[i++];
        json::Value value;
        if (n.try_get_property("aiExp", value)) {
            size_t k = 0;
            for (auto [field, member] : ai_param_names()) note.aiParams.*member = value[k++].get_single();
        }
        if (n.try_get_property("vibratoDepth", value)) {
            V6ScoreVibrato vibrato;
            vibrato.depth = value.get_int32();
            vibrato.rate = n.get_property("vibratoRate").get_int32();
            vibrato.durationTick = n.get_property("vibratoDurationTick").get_int64();
            vibrato.ai = n.get_property("aiVibrato").get_boolean();
            note.vibrato = vibrato;
        }
    }
    return score;
}

void end_to_end_matches_capture(const std::string& name, const std::string& phrase, const fs::path& root, Worst& worst_f0) {
    fs::path voice_dir = root / "voice-asa";
    V6LocalVoice voice(voice_dir.string(), V6Registration::read(voice_dir.string(), AsaBank), 0);
    auto result = voice.render(phrase_score(root.parent_path() / "renderer" / "phrases" / (phrase + ".json")));
    std::vector<float> reference;
    if (fs::exists(root / name / "vsm.bin"))
        for (const auto& r : read_records(root / name / "vsm.bin"))
            if (r.name == "850b0" && r.label == "output") reference = floats(r.data);
    CHECK_MSG(reference.empty() || reference.size() == result.pitch.pitches.size(), name + ": f0 length");
    float f0_error = 0;
    for (size_t i = 0; i < std::min(reference.size(), result.pitch.pitches.size()); i++) {
        bool gap = result.pitch.pitches[i] < 0;
        if (gap != (reference[i] == vsm_pitch::Gap)) {
            f0_error = INFINITY;
            break;
        }
        if (!gap) f0_error = std::max(f0_error, std::fabs(result.pitch.pitches[i] * 100 - 6900 - reference[i]));
    }
    auto pcm = read_pcm16(root / name / "out.wav");
    CHECK_MSG(pcm.size() == result.samples.size(), name + ": length " + std::to_string(result.samples.size()) + " vs " + std::to_string(pcm.size()));
    double distance = band_distance(result.samples, pcm);
    std::printf("end to end %-22s f0 %.3g cents%s, band distance %.3f dB\n", name.c_str(), f0_error, reference.empty() ? " (no native f0)" : "", distance);
    CHECK_MSG(f0_error < EndToEndCents, name + ": end-to-end f0 differs by " + std::to_string(f0_error) + " cents");
    CHECK_MSG(distance < MaxBandDistance, name + ": band spectra differ by " + std::to_string(distance) + " dB (median)");
    worst_f0.add(f0_error);
}

void note_clamp_and_defaults() {
    V6Score score{};
    score.tempoBpm = 120;
    for (auto [start, pitch] : {std::pair{0, 100}, {480, 10}, {960, 60}}) {
        V6ScoreNote note{};
        note.startTick = start;
        note.durationTick = 480;
        note.pitch = pitch;
        note.phonemes = "a";
        score.notes.push_back(note);
    }
    auto notes = build_notes(score, 60.0 / (120 * 480)).notes;
    std::vector<float> pitches, offsets;
    for (const auto& n : notes)
        if (!n.rest()) {
            pitches.push_back(*n.pitch);
            offsets.push_back(n.pitch_offset);
        }
    CHECK((pitches == std::vector<float>{2400, -4500, -900}));
    CHECK((offsets == std::vector<float>{-700, 1400, 0}));
    CHECK(notes[1].vibrato_depth == 0 && notes[1].ai_vibrato);
}

}

int main() {
    note_clamp_and_defaults();
    const char* reference = reference_dir();
    if (!reference) return g_failures ? 1 : 77;
    fs::path root = reference;
    Worst stage, expression, f0, end_to_end;
    for (const char* name : {"ai-pitch", "ai-vibrato", "ai-vibrato-resid", "ai-vibrato-resid2", "ai-vibrato-resid3", "ai-vibrato-resid4",
                             "ai-morning-random2", "ai-extreme", "ai-amplitude"})
        note_stage_matches_capture(name, root / name, stage);
    for (const char* name : {"ai-amplitude", "ai-morning-random2", "ai-pitch"}) expression_matches_capture(name, root / name, expression);
    for (const char* name : {"ai-pitch", "ai-extreme", "ai-vibrato", "ai-morning-random2"})
        f0_matches_capture_given_native_pitch_model(name, root / name, f0);
    for (auto [name, phrase] : std::map<std::string, std::string>{
             {"ai-pitch", "ai-pitch"}, {"ai-vibrato", "ai-vibrato"}, {"ai-amplitude", "ai-amplitude"}, {"ai-amplitude-ctl", "ai-amplitude-ctl"},
             {"ai-extreme", "ai-extreme"}, {"ai-morning-random2", "ai-morning-random"}, {"classic-base", "ai-base"}, {"classic-vel", "ai-vel"},
             {"classic-cexp", "ai-cexp"}, {"classic-cons", "ai-cons"}, {"classic-hmm", "ai-hmm"},
             {"ctl-Dynamics", "morning-Dynamics"}})
        end_to_end_matches_capture(name, phrase, root, end_to_end);
    std::printf("max errors: note stage %.3g cents, Expression %.3g, f0 from native pitch %.3g cents, end-to-end f0 %.3g cents\n", stage.value,
                expression.value, f0.value, end_to_end.value);
    if (g_failures) return 1;
    std::puts("test_note_params ok");
    return 0;
}
