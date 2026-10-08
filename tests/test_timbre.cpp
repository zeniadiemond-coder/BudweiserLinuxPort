#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "check.hpp"
#include "openv6/frame_driver.hpp"
#include "openv6/frame_synth.hpp"
#include "openv6/model_file.hpp"
#include "openv6/networks.hpp"
#include "openv6/score_features.hpp"
#include "openv6/synthesizer.hpp"

using namespace openv6;
namespace fs = std::filesystem;

namespace {

constexpr int AsaNpIdx = 0xa5a5;

std::string read_text(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw InvalidDataError("cannot open " + path.string());
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::unique_ptr<json::Document> read_json(const fs::path& path) { return json::Document::parse(read_text(path)); }

std::vector<float> read_floats(const fs::path& path) {
    auto bytes = read_file(path.string());
    std::vector<float> result(bytes.size() / 4);
    std::memcpy(result.data(), bytes.data(), result.size() * 4);
    return result;
}

float max_error(std::span<const float> a, std::span<const float> b) {
    CHECK_MSG(a.size() == b.size(), "length " + std::to_string(a.size()) + " vs " + std::to_string(b.size()));
    float max = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); i++) max = std::max(max, std::fabs(a[i] - b[i]));
    return max;
}

double correlation(std::span<const float> a, std::span<const float> b) {
    double sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
        sa += a[i];
        sb += b[i];
        sab += static_cast<double>(a[i]) * b[i];
        saa += static_cast<double>(a[i]) * a[i];
        sbb += static_cast<double>(b[i]) * b[i];
    }
    double cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
    return cov / std::sqrt(va * vb);
}

long first_difference(std::span<const float> a, std::span<const float> b, float threshold) {
    for (size_t i = 0; i < std::min(a.size(), b.size()); i++) {
        if (std::fabs(a[i] - b[i]) > threshold) return static_cast<long>(i);
    }
    return -1;
}

std::vector<V6Phoneme> reference_phonemes(json::Value list) {
    std::vector<V6Phoneme> result;
    for (auto p : list.enumerate_array()) {
        result.push_back({p.get_property("start").get_double(), p.get_property("end").get_double(),
                          std::string(p.get_property("name").get_string()), p.get_property("f10").get_single(),
                          p.get_property("b14").get_int32() != 0, p.get_property("b15").get_int32() != 0,
                          p.get_property("i20").get_int32()});
    }
    return result;
}

void test_reference(const fs::path& dir) {
    auto score = read_json(dir / "score.json");
    auto phonemes = reference_phonemes(score->root().get_property("phonemes"));
    auto timbre_file = std::make_shared<ModelFile>(ModelFile::load((dir / "asa.vtb2").string()));
    const ModelFile& timbre = *timbre_file;
    V6Synthesizer synth(timbre_file, nullptr, nullptr, AsaNpIdx);

    int frames, width;
    auto rows = score_features::frame_rows(timbre.root(), phonemes, frames, width);
    auto native = read_floats(dir / "x415.f32");
    float rows_error = max_error(rows, std::span(native).subspan(63 * static_cast<size_t>(width), static_cast<size_t>(frames) * width));
    CHECK_MSG(rows_error < 1e-6f, "415 rows differ: " + std::to_string(rows_error));
    auto vec = synth.frame_vectors(phonemes, frames);
    auto native_vec = read_floats(dir / "vec.f32");
    float vec_error = max_error(vec, std::span(native_vec).first(static_cast<size_t>(frames) * 47));
    CHECK_MSG(vec_error < 1e-4f, "Frame vectors differ: " + std::to_string(vec_error));
    std::printf("reference score features: rows %.3g, vectors %.3g\n", rows_error, vec_error);

    auto scalars = read_floats(dir / "frame_scalars.f32");
    int count = static_cast<int>(scalars.size() / 4);
    std::vector<V6Frame> inputs;
    for (int f = 0; f < count; f++) {
        inputs.push_back({{native_vec.begin() + f * 47, native_vec.begin() + (f + 1) * 47}, scalars[f * 4], scalars[f * 4 + 1], scalars[f * 4 + 3]});
    }
    auto out = FrameSynth(timbre.root(), static_cast<int>(scalars[2])).run(inputs);
    double c0 = correlation(out.harmonic, read_floats(dir / "ch0.f32")), c1 = correlation(out.noise, read_floats(dir / "ch1.f32"));
    CHECK_MSG(c0 > 0.99999 && c1 > 0.99999, "Synthesis correlation " + std::to_string(c0) + " / " + std::to_string(c1));
    std::printf("reference synthesis: correlation %.9f / %.9f\n", c0, c1);
}

fs::path score_path(const fs::path& scores, const std::string& name) {
    static const StringMap<std::string> Aliases = {
        {"f_en_allen", "f_en_asa"}, {"f_en_allen_npidx257", "f_en_asa"}, {"a_ja_allen_cross", "a_ja_short"},
        {"a_ja_asahi", "a_ja_short"}, {"a_ja_asa_npidx257", "a_ja_short"},
    };
    auto alias = Aliases.find(name);
    return scores / ((alias != Aliases.end() ? alias->second : name) + ".json");
}

std::vector<int> curve(json::Value score, std::string_view name, int frames, double tick) {
    json::Value controllers;
    if (!score.try_get_property("controllers", controllers)) return {};
    for (auto c : controllers.enumerate_array()) {
        if (c.get_property("name").get_string() != name) continue;
        auto events = c.get_property("events");
        size_t count = events.array_length();
        std::vector<int> result(frames);
        long e = -1;
        auto time = [&](size_t i) { return 0.5 + events[i].get_property("tick").get_int64() * tick; };
        for (int f = 0; f < frames; f++) {
            while (e + 1 < static_cast<long>(count) && time(e + 1) <= f / 200.0) e++;
            result[f] = events[e < 0 ? 0 : e].get_property("value").get_int32();
        }
        return result;
    }
    return {};
}

struct Totals {
    float vec = 0, ch0 = 0, ch1 = 0;
    double min_c0 = 1, min_c1 = 1;
    double longest_seconds = 0;
    std::string longest;
};

void test_oracle_case(const fs::path& dir, const fs::path& scores, Totals& totals) {
    auto meta_doc = read_json(dir / "meta.json");
    auto meta = meta_doc->root();
    auto bank = read_json(fs::path(std::string(meta.get_property("package").get_string())) / "bank.json");
    std::string component = "Components\\" + std::string(meta.get_property("bank_id").get_string());
    auto bank_component = bank->root().get_property(component);
    json::Value name_value;
    if (!bank_component.try_get_property("TModelName", name_value)) name_value = bank_component.get_property("PModelName");
    auto timbre_file = std::make_shared<ModelFile>(ModelFile::load((fs::path(std::string(meta.get_property("models").get_string())) / (std::string(name_value.get_string()) + ".vtb2")).string()));
    const ModelFile& timbre = *timbre_file;
    V6Synthesizer synth(timbre_file, nullptr, nullptr, 0);

    int speaker = meta.get_property("timbreSpeaker").get_int32(), expression_k = meta.get_property("expressionK").get_int32();
    int part_frames = meta.get_property("partFrames").get_int32();
    double tick = meta.get_property("tickSeconds").get_double();
    auto score_doc = read_json(score_path(scores, dir.filename().string()));
    V6Controls controls;
    auto expression = curve(score_doc->root(), "Expression", part_frames, tick);
    auto character = curve(score_doc->root(), "S5Character", part_frames, tick);
    if (!expression.empty()) controls.expression = expression;
    if (!character.empty()) controls.character = character;

    auto f0 = read_floats(dir / "f0.f32");
    auto sections_doc = read_json(dir / "sections.json");
    auto sections = sections_doc->root();
    for (size_t i = 0; i < sections.array_length(); i++) {
        auto section = sections[i];
        int origin_frame = static_cast<int>(round_away(section.get_property("origin").get_double() * 200));
        std::vector<V6Phoneme> phonemes;
        for (auto p : section.get_property("phonemes").enumerate_array()) {
            phonemes.push_back({p.get_property("Start").get_double(), p.get_property("End").get_double(),
                                std::string(p.get_property("Name").get_string()), p.get_property("Pitch").get_single(),
                                p.get_property("NoteStart").get_boolean(), p.get_property("Nucleus").get_boolean(),
                                p.get_property("Class").get_int32(), p.get_property("Note").get_int32()});
        }
        int frames;
        auto vec = synth.frame_vectors(phonemes, frames);
        auto oracle_vec = read_floats(dir / ("vec_" + std::to_string(i) + ".f32"));
        float vec_error = max_error(vec, oracle_vec);
        totals.vec = std::max(totals.vec, vec_error);
        CHECK_MSG(vec_error < 1e-4f, dir.filename().string() + " section " + std::to_string(i) + " vectors differ: " + std::to_string(vec_error));

        auto at = [&](int f) { return std::min(f + origin_frame, part_frames - 1); };
        std::vector<V6Frame> inputs;
        for (int c = 0; c < frames + frame_driver::Latency; c++) {
            int f = std::min(c, frames - 1);
            inputs.push_back({{oracle_vec.begin() + f * 47, oracle_vec.begin() + (f + 1) * 47}, std::clamp(f0[at(f)], -6900.f, 6200.f),
                              frame_driver::control(controls.expression_at(at(f)), expression_k),
                              0.3f * frame_driver::bipolar(controls.character_at(at(f)), -64, 63)});
        }
        auto start = std::chrono::steady_clock::now();
        auto out = FrameSynth(timbre.root(), speaker).run(inputs);
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (seconds > totals.longest_seconds) {
            totals.longest_seconds = seconds;
            totals.longest = dir.filename().string() + " section " + std::to_string(i) + " (" + std::to_string(frames) + " frames)";
        }

        auto ch0 = read_floats(dir / ("ch0_" + std::to_string(i) + ".f32"));
        auto ch1 = read_floats(dir / ("ch1_" + std::to_string(i) + ".f32"));
        float e0 = max_error(out.harmonic, ch0), e1 = max_error(out.noise, ch1);
        double c0 = correlation(out.harmonic, ch0), c1 = correlation(out.noise, ch1);
        totals.ch0 = std::max(totals.ch0, e0);
        totals.ch1 = std::max(totals.ch1, e1);
        totals.min_c0 = std::min(totals.min_c0, c0);
        totals.min_c1 = std::min(totals.min_c1, c1);
        std::printf("%-32s s%zu  %5d frames  vec %.2g  ch0 %.3g (corr %.9f, first >1e-6 @%ld)  ch1 %.3g (corr %.9f, first >1e-6 @%ld)  %.2fs\n",
                    dir.filename().string().c_str(), i, frames, vec_error, e0, c0, first_difference(out.harmonic, ch0, 1e-6f), e1, c1,
                    first_difference(out.noise, ch1, 1e-6f), seconds);
        CHECK_MSG(c0 > 0.99999 && c1 > 0.99999, dir.filename().string() + " synthesis correlation " + std::to_string(c0) + " / " + std::to_string(c1));
    }
}

}

int main() {
    const char* reference = reference_dir();
    const char* oracle = std::getenv("OPENV6_ORACLE");
    if (!reference && !oracle) {
        std::puts("OPENV6_REFERENCE and OPENV6_ORACLE unset; skipping");
        return 77;
    }
    if (reference) test_reference(fs::path(reference) / "reference");
    if (oracle) {
        Totals totals;
        fs::path root(oracle);
        std::vector<fs::path> cases;
        for (const auto& entry : fs::directory_iterator(root)) {
            if (fs::exists(entry.path() / "sections.json")) cases.push_back(entry.path());
        }
        std::sort(cases.begin(), cases.end());
        for (const auto& dir : cases) test_oracle_case(dir, root.parent_path() / "scores", totals);
        std::printf("oracle: %zu cases, max error vec %.3g ch0 %.3g ch1 %.3g, min correlation %.9f / %.9f\n", cases.size(), totals.vec,
                    totals.ch0, totals.ch1, totals.min_c0, totals.min_c1);
        std::printf("longest section: %s, %.2fs\n", totals.longest.c_str(), totals.longest_seconds);
    }
    return g_failures == 0 ? 0 : 1;
}
