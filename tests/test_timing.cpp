#include <algorithm>
#include <cstdio>
#include <iostream>

#include "check.hpp"
#include "openv6/frame_driver.hpp"
#include "openv6/timing.hpp"
#include "openv6/vtmg_model.hpp"
#include "openv6/voice_package.hpp"
#include "timing_support.hpp"

using namespace openv6;
using namespace timing_support;

namespace {

constexpr int AsaParamExp = 1200;

std::unique_ptr<json::Document> load_json(const std::filesystem::path& path) { return json::Document::parse(read_text(path)); }

void timing_matches_capture(const std::filesystem::path& dir) {
    auto doc = load_json(dir / "score.json");
    json::Value score = doc->root();
    auto notes = capture_notes(score);
    int count;
    auto rows = timing::rows(notes, count);
    CHECK_MSG(max_error(rows, read_floats(dir / "timing_rows.f32")) < 1e-6f, "VTMG rows differ");
    auto vtmg = VtmgModel::load((dir / "VOCALOID6.vtmg").string());
    auto outputs = vtmg.forward(rows, count, score.get_property("condition").get_int32());
    auto native = read_floats(dir / "timing_out.f32");
    std::vector<float> expected_outputs(native.begin() + 9, native.end() - 9);
    CHECK_MSG(max_error(outputs, expected_outputs) < 2e-6f, "VTMG outputs differ");
    auto timed = timing::phonemes(notes, outputs, vtmg.consonant_weights());
    CHECK_MSG(compare(capture_phonemes(score.get_property("timed")), timed) < 1e-6, "timed phonemes differ");
    auto sections = timing::sections(timed);
    CHECK(sections.size() == 1);
    CHECK_MSG(compare(capture_phonemes(score.get_property("phonemes")), sections[0].phonemes) < 1e-6, "section phonemes differ");
}

void driver_matches_capture(const std::filesystem::path& dir) {
    auto doc = load_json(dir / "score.json");
    auto timed = capture_phonemes(doc->root().get_property("timed"));
    auto gate = frame_driver::gate(timed, 3220, false);
    CHECK_MSG(max_error(gate, read_floats(dir / "driver_vec12.f32")) < 1e-6f, "gate differs");
    std::vector<float> control(3220, frame_driver::control(50, AsaParamExp - 3000));
    CHECK_MSG(max_error(control, read_floats(dir / "driver_vecb.f32")) < 1e-6f, "control differs");
}

void score_timing_matches_capture(const std::string& bundle, const std::filesystem::path& dir) {
    auto doc = load_json(dir / "score.json");
    json::Value score = doc->root();
    auto notes = capture_notes(score);
    auto vtmg = VtmgModel::load((dir / "VOCALOID6.vtmg").string());
    auto timed = timing::run(vtmg, notes, score.get_property("condition").get_int32());
    CHECK_MSG(compare(capture_phonemes(score.get_property("timed")), timed) < 1e-6, bundle + ": timed phonemes differ");
    auto sections = timing::sections(timing::edit(timed, notes, capture_tick(score)));
    std::vector<std::vector<V6Phoneme>> expected;
    for (json::Value s : score.get_property("sections").enumerate_array()) expected.push_back(capture_phonemes(s));
    CHECK_MSG(expected.size() == sections.size(), bundle + ": section count differs");
    for (size_t i = 0; i < std::min(expected.size(), sections.size()); i++) {
        CHECK_MSG(compare(expected[i], sections[i].phonemes) < 1e-6, bundle + ": section " + std::to_string(i) + " differs");
    }
}

struct StageError {
    double timing = 0;
    double sections = 0;
    float gate = 0;
    int mismatches = 0;
};

void oracle_case(const std::filesystem::path& dir, StageError& worst) {
    std::string name = dir.filename().string();
    auto meta_doc = load_json(dir / "meta.json");
    json::Value meta = meta_doc->root();
    auto notes_doc = load_json(dir / "notes.json");
    auto notes = oracle_notes(notes_doc->root());

    auto package = load_voice_package(std::string(meta.get_property("package").get_string()));
    CHECK_MSG(package.config.bank_id == meta.get_property("bank_id").get_string(), name + ": bank id");
    CHECK_MSG(package.config.language_id == meta.get_property("language_id").get_int32(), name + ": language id");
    std::string models = package.model_directory();
    CHECK_MSG(std::filesystem::equivalent(models, std::string(meta.get_property("models").get_string())), name + ": model directory " + models);
    std::filesystem::path vtmg_path = std::filesystem::path(models) / (package.registration.timbre_model + ".vtmg");
    if (!std::filesystem::exists(vtmg_path)) vtmg_path = std::filesystem::path(models) / "VOCALOID6.vtmg";
    auto vtmg = VtmgModel::load(vtmg_path.string());

    int condition = meta.get_property("condition").get_int32();
    double tick = meta.get_property("tickSeconds").get_double();
    auto timed = timing::edit(timing::run(vtmg, notes, condition), notes, tick);

    auto timing_doc = load_json(dir / "timing.json");
    double timing_error = compare(oracle_phonemes(timing_doc->root()), timed);
    CHECK_MSG(timing_error == 0, name + ": timing differs by " + std::to_string(timing_error));
    worst.timing = std::max(worst.timing, timing_error);

    auto sections = timing::sections(timed);
    auto sections_doc = load_json(dir / "sections.json");
    double sections_error = sections_doc->root().array_length() == sections.size() ? 0 : INFINITY;
    for (size_t i = 0; i < std::min(sections.size(), sections_doc->root().array_length()); i++) {
        json::Value expected = sections_doc->root()[i];
        double origin_error = std::fabs(expected.get_property("origin").get_double() - sections[i].origin);
        sections_error = std::max({sections_error, origin_error, compare(oracle_phonemes(expected.get_property("phonemes")), sections[i].phonemes)});
    }
    CHECK_MSG(sections_error == 0, name + ": sections differ by " + std::to_string(sections_error));
    worst.sections = std::max(worst.sections, sections_error);

    int frames = meta.get_property("partFrames").get_int32();
    auto gate_error = max_error(frame_driver::gate(timed, frames, false), read_floats(dir / "gate.f32"));
    CHECK_MSG(gate_error == 0, name + ": gate differs by " + std::to_string(gate_error));
    worst.gate = std::max(worst.gate, gate_error);
    if (timing_error != 0 || sections_error != 0 || gate_error != 0) worst.mismatches++;
}

void driver_math() {
    CHECK(frame_driver::dynamics_gain(64) == 1.f);
    CHECK(frame_driver::bipolar(-100, -64, 63) == -1.f);
    CHECK(frame_driver::bipolar(63, -64, 63) == 1.f);
    CHECK(frame_driver::control(50, 0) == 0.f);
    CHECK(frame_driver::noise_gain(0) == 1.f);
}

}

int main() {
    driver_math();
    const char* oracle = std::getenv("OPENV6_ORACLE");
    const char* reference = reference_dir();
    if (!reference && !oracle) {
        std::puts("OPENV6_REFERENCE and OPENV6_ORACLE are unset");
        return g_failures ? 1 : 77;
    }
    if (reference) {
        std::filesystem::path root = reference;
        timing_matches_capture(root / "reference");
        driver_matches_capture(root / "reference");
        for (const char* bundle : {"reference-morning-pitch", "reference-english", "reference-chinese", "reference-edits", "reference-timing-edits",
                                   "reference-rest-edit", "reference-extend", "reference-english-extend", "reference-rest-extend"}) {
            score_timing_matches_capture(bundle, root / bundle);
        }
    }
    if (oracle) {
        StageError worst;
        int cases = 0;
        for (const auto& entry : std::filesystem::directory_iterator(oracle)) {
            if (!std::filesystem::exists(entry.path() / "timing.json")) continue;
            try {
                oracle_case(entry.path(), worst);
            } catch (const std::exception& e) {
                CHECK_MSG(false, entry.path().filename().string() + ": " + e.what());
            }
            cases++;
        }
        std::printf("oracle: %d cases, mismatching cases %d, max timing error %g, sections %g, gate %g\n", cases, worst.mismatches, worst.timing,
                    worst.sections, static_cast<double>(worst.gate));
    }
    if (g_failures) std::fprintf(stderr, "%d failures\n", g_failures);
    return g_failures ? 1 : 0;
}
