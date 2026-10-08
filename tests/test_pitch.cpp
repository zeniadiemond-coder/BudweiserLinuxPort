#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <sstream>

#include "check.hpp"
#include "openv6/model_file.hpp"
#include "openv6/pitch_model.hpp"
#include "openv6/synthesizer.hpp"
#include "openv6/timing.hpp"
#include "openv6/vpit_features.hpp"
#include "openv6/vsm_pitch.hpp"
#include "openv6/vtmg_model.hpp"
#include "timing_support.hpp"

using namespace openv6;
namespace fs = std::filesystem;

namespace {

constexpr float PitchNoise = 0.1f;

std::vector<float> read_floats(const fs::path& path) {
    auto bytes = read_file(path.string());
    std::vector<float> result(bytes.size() / 4);
    std::memcpy(result.data(), bytes.data(), result.size() * 4);
    return result;
}

std::unique_ptr<json::Document> read_json(const fs::path& path) {
    auto bytes = read_file(path.string());
    return json::Document::parse(std::string(bytes.begin(), bytes.end()));
}

float max_error(std::span<const float> a, std::span<const float> b) {
    CHECK(a.size() == b.size());
    float worst = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); i++) worst = std::max(worst, std::fabs(a[i] - b[i]));
    return worst;
}

float f0_error(std::span<const float> a, std::span<const float> b) {
    CHECK(a.size() == b.size());
    float worst = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); i++) {
        bool gap_a = a[i] == vsm_pitch::Gap, gap_b = b[i] == vsm_pitch::Gap;
        if (gap_a != gap_b) return INFINITY;
        if (!gap_a) worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

std::vector<float> slice(std::span<const float> v, size_t index, size_t width) { return {v.begin() + index * width, v.begin() + (index + 1) * width}; }

double since(std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); }


void test_vsm_pitch_unit() {
    std::vector<float> ramp(40);
    for (size_t i = 0; i < ramp.size(); i++) ramp[i] = static_cast<float>(i);
    auto same = vsm_pitch::resample(ramp, 1.0f);
    CHECK(same.size() == ramp.size() && max_error(same, ramp) == 0);
    auto half = vsm_pitch::resample(ramp, 0.5f);
    CHECK(half.size() == 80);
    CHECK(std::fabs(half[21] - 10.5f) < 1e-4f);

    std::vector<vsm_pitch::KeyPoint> points = {{0, 100.f}, {10, 300.f}, {20, 300.f}, {30, 100.f}};
    auto spline = vsm_pitch::constrained_spline(points);
    CHECK(spline.size() == 31);
    CHECK(std::fabs(spline[10] - 300.f) < 0.01f && std::fabs(spline[0] - 100.f) < 0.01f && spline[30] == 100.f);
    for (float v : spline) CHECK(v >= 100.f - 1 && v <= 300.f + 1);

    std::vector<float> flat(20, 500.f), zero(20, 0.f);
    auto note = vsm_pitch::note_pitch(flat, zero, 74, {});
    CHECK(max_error(note, flat) < 0.01f);

    V6PitchEdit edit{{0, 10, 20}, {6900.f, 7000.f, 7000.f}, 60.0 / (120 * 480), 0.0};
    auto direct = vsm_pitch::direct_pitch(edit);
    CHECK(direct.first == 0);
    CHECK(std::isnan(direct.cents.back()) && !std::isnan(direct.cents.front()));
}

void test_xorshift() {

    Xorshift128 a, b;
    double sum = 0;
    for (int i = 0; i < 1000; i++) {
        double g = a.gauss();
        CHECK(g == b.gauss());
        sum += g;
    }
    CHECK(std::fabs(sum / 1000) < 0.2);
}

std::shared_ptr<ModelFile>& pitch_model_file(std::map<std::string, std::shared_ptr<ModelFile>>& cache, const std::string& path) {
    auto& model = cache[path];
    if (!model) model = std::make_shared<ModelFile>(ModelFile::load(path));
    return model;
}

std::vector<V6Note> reference_notes(json::Value score) {
    std::vector<V6Note> notes;
    for (auto n : score.get_property("notes").enumerate_array()) {
        V6Note note;
        note.start = n.get_property("start").get_double();
        note.end = n.get_property("end").get_double();
        auto pitch = n.get_property("pitch");
        if (pitch.kind() != json::Kind::Null) note.pitch = pitch.get_single();
        for (auto p : n.get_property("phonemes").enumerate_array()) note.phonemes.emplace_back(p.get_string());
        note.meter = n.get_property("i18").get_int32();
        note.position = n.get_property("f14").get_single();
        note.flag = n.get_property("b21").get_int32() != 0;
        json::Value language;
        if (n.try_get_property("language", language)) note.language = language.get_int32();
        notes.push_back(std::move(note));
    }
    return notes;
}

void test_capture(const std::string& root) {
    fs::path dir = fs::path(root) / "reference";
    auto score_doc = read_json(dir / "score.json");
    auto score = score_doc->root();
    std::vector<V6Phoneme> phonemes;
    for (auto p : score.get_property("phonemes").enumerate_array()) {
        V6Phoneme phoneme;
        phoneme.start = p.get_property("start").get_double();
        phoneme.end = p.get_property("end").get_double();
        phoneme.name = std::string(p.get_property("name").get_string());
        phoneme.pitch = p.get_property("f10").get_single();
        phoneme.note_start = p.get_property("b14").get_int32() != 0;
        phoneme.nucleus = p.get_property("b15").get_int32() != 0;
        phoneme.cls = p.get_property("i20").get_int32();
        phonemes.push_back(std::move(phoneme));
    }
    auto model = ModelFile::load((dir / "asa.vpit").string());
    VpitFeatures features(model.root());
    int frames;
    auto rows = features.rows(phonemes, frames);
    auto native_rows = read_floats(dir / "vpit_rows.f32");
    float rows_error = max_error(rows, native_rows);
    CHECK_MSG(rows_error < 2e-4f, "VPIT rows differ by " + std::to_string(rows_error));

    constexpr int AsaNpIdx = 0xa5a5;
    PitchModel pitch_model(model.root(), std::span<const float>(native_rows).first(VpitFeatures::Width), PitchNoise, AsaNpIdx >> 8);
    auto native_out = read_floats(dir / "pitch_out.f32");
    std::vector<float> native_f0, native_residual;
    for (size_t i = 0; i < native_out.size(); i += 2) {
        native_f0.push_back(native_out[i]);
        native_residual.push_back(native_out[i + 1]);
    }
    std::vector<float> f0(frames), residual(frames);
    for (int f = 0; f < frames; f++) {
        auto step = pitch_model.step(slice(native_rows, f, VpitFeatures::Width));
        f0[f] = step.f0;
        residual[f] = step.residual;
    }
    float f0_diff = max_error(f0, native_f0);
    std::printf("capture: rows %.3g, pitch model f0 %.3g cents\n", rows_error, f0_diff);
    CHECK_MSG(f0_diff < 2.0f, "pitch model f0 differs by " + std::to_string(f0_diff));

    auto notes = reference_notes(score);
    auto vtmg = VtmgModel::load((dir / "VOCALOID6.vtmg").string());
    auto timed = timing::run(vtmg, notes, score.get_property("condition").get_int32());
    auto ranges = vsm_pitch::pitch_notes(notes, timed);
    int part_frames = static_cast<int>(round_even(notes.back().end * 200));
    auto curve = vsm_pitch::f0_curve(native_f0, native_residual, ranges, part_frames);
    auto frame_scalars = read_floats(dir / "frame_scalars.f32");
    std::vector<float> native_curve;
    for (size_t i = 0; i < frame_scalars.size(); i += 4) native_curve.push_back(frame_scalars[i]);
    float curve_diff = max_error(std::span<const float>(curve).first(frames), std::span<const float>(native_curve).first(frames));
    std::printf("capture: VSM post-processing %.3g cents\n", curve_diff);
    CHECK_MSG(curve_diff < 1e-3f, "VSM pitch post-processing differs by " + std::to_string(curve_diff));
}

struct BankInfo {
    std::string pitch_model_path;
    int np_idx;
};

BankInfo bank_info(json::Value meta) {
    std::string package(meta.get_property("package").get_string()), models(meta.get_property("models").get_string());
    std::string bank_id(meta.get_property("bank_id").get_string());
    auto bank = read_json(fs::path(package) / "bank.json");
    auto component = bank->root().get_property("Components\\" + bank_id);
    std::string pitch_name(component.get_property("PModelName").get_string());
    fs::path pitch = fs::path(models) / (pitch_name + ".vpit");
    if (!fs::exists(pitch)) pitch = fs::path(models) / "VOCALOID6.vpit";
    BankInfo info{pitch.string(), 0};
    json::Value np_idx;
    if (component.try_get_property("NPIdx", np_idx)) info.np_idx = np_idx.get_int32();
    json::Value override_idx;
    if (meta.try_get_property("npIdx", override_idx) && override_idx.kind() == json::Kind::Number) info.np_idx = override_idx.get_int32();
    return info;
}

void test_oracle(const fs::path& root) {
    std::map<std::string, std::shared_ptr<ModelFile>> models;
    float worst_pitch = 0, worst_residual = 0, worst_f0 = 0;
    double pitch_seconds = 0, longest_seconds = 0;
    std::string longest_case;
    for (auto& entry : fs::directory_iterator(root)) {
        fs::path dir = entry.path();
        if (!fs::exists(dir / "sections.json")) continue;
        std::string name = dir.filename().string();
        auto meta_doc = read_json(dir / "meta.json");
        auto meta = meta_doc->root();
        auto bank = bank_info(meta);
        V6Synthesizer synth(nullptr, pitch_model_file(models, bank.pitch_model_path), nullptr, bank.np_idx);

        auto sections_doc = read_json(dir / "sections.json");
        std::vector<std::vector<V6Phoneme>> sections;
        for (auto s : sections_doc->root().enumerate_array()) {
            std::vector<V6Phoneme> section;
            for (auto p : s.get_property("phonemes").enumerate_array()) {
                V6Phoneme ph;
                ph.start = p.get_property("Start").get_double();
                ph.end = p.get_property("End").get_double();
                ph.name = std::string(p.get_property("Name").get_string());
                ph.pitch = p.get_property("Pitch").get_single();
                ph.note_start = p.get_property("NoteStart").get_boolean();
                ph.nucleus = p.get_property("Nucleus").get_boolean();
                ph.cls = p.get_property("Class").get_int32();
                ph.note = p.get_property("Note").get_int32();
                section.push_back(std::move(ph));
            }
            sections.push_back(std::move(section));
        }

        float case_pitch = 0, case_residual = 0;
        auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < sections.size(); i++) {
            auto curves = synth.pitch_curve(sections[i]);
            case_pitch = std::max(case_pitch, max_error(curves.f0, read_floats(dir / ("pitch_" + std::to_string(i) + ".f32"))));
            case_residual = std::max(case_residual, max_error(curves.residual, read_floats(dir / ("residual_" + std::to_string(i) + ".f32"))));
        }
        double seconds = since(start);
        pitch_seconds += seconds;
        if (seconds > longest_seconds) {
            longest_seconds = seconds;
            longest_case = name;
        }

        auto notes_doc = read_json(dir / "notes.json");
        auto notes = timing_support::oracle_notes(notes_doc->root());
        auto timing_doc = read_json(dir / "timing.json");
        auto timed = timing_support::oracle_phonemes(timing_doc->root());

        std::optional<V6PitchEdit> edit;
        fs::path score_path = root.parent_path() / "scores" / (name + ".json");
        if (fs::exists(score_path)) {
            auto score_doc = read_json(score_path);
            json::Value pitch, ticks;
            if (score_doc->root().try_get_property("pitch", pitch)) {
                V6PitchEdit e;
                for (auto t : score_doc->root().get_property("pitchTicks").enumerate_array()) e.ticks.push_back(t.get_int64());
                for (auto c : pitch.enumerate_array()) e.midi_cents.push_back(c.get_single());
                e.tick_seconds = meta.get_property("tickSeconds").get_double();
                e.origin_seconds = 0.5;
                edit = std::move(e);
            }
        }
        auto f0 = synth.f0(notes, timed, edit ? &*edit : nullptr);
        float case_f0 = f0_error(f0, read_floats(dir / "f0.f32"));
        std::printf("%-32s sections %zu  pitch %.3g  residual %.3g  f0 %.3g  (%.3fs)%s\n", name.c_str(), sections.size(), case_pitch,
                    case_residual, case_f0, seconds, edit ? "  [pitch edit]" : "");
        worst_pitch = std::max(worst_pitch, case_pitch);
        worst_residual = std::max(worst_residual, case_residual);
        worst_f0 = std::max(worst_f0, case_f0);
    }
    std::printf("oracle max errors: pitch %.3g, residual %.3g, f0 %.3g; pitch stage total %.2fs, longest %s %.3fs\n", worst_pitch,
                worst_residual, worst_f0, pitch_seconds, longest_case.c_str(), longest_seconds);
    CHECK(worst_pitch < 1e-3f);
    CHECK(worst_residual < 1e-3f);
    CHECK(worst_f0 < 1e-3f);
}

}

int main() {
    test_vsm_pitch_unit();
    test_xorshift();
    const char* reference = std::getenv("OPENV6_REFERENCE");
    const char* oracle = std::getenv("OPENV6_ORACLE");
    if (reference) test_capture(reference);
    if (oracle) test_oracle(oracle);
    if (g_failures) return 1;
    if (!reference && !oracle) return 77;
    std::puts("test_pitch ok");
    return 0;
}
