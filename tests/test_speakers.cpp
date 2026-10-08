#include <cstdio>
#include <filesystem>
#include <fstream>

#include "check.hpp"
#include "openv6/speakers.hpp"
#include "openv6/voice_package.hpp"
#include "timing_support.hpp"

using namespace openv6;
using timing_support::read_text;

namespace {

namespace fs = std::filesystem;

V6Registration registration(const std::string& model, int np_idx) {
    V6Registration result;
    result.timbre_model = model;
    result.np_idx = np_idx;
    return result;
}

template <class F>
bool throws_invalid(F&& f) {
    try {
        f();
    } catch (const InvalidDataError&) {
        return true;
    }
    return false;
}

void speakers_unit() {
    auto asa = speakers::available(registration("asa", 42405));
    CHECK(asa.size() == 226);
    for (size_t i = 0; i < asa.size(); i++) CHECK(asa[i].row == static_cast<int>(i));
    CHECK(asa[165].name == "165: JJF001_016 (asa)");
    CHECK(speakers::available(registration("custom_model", 0)).empty());

    CHECK(speakers::np_idx(registration("asa", 42405), 165) == 42405);
    CHECK(speakers::np_idx(registration("AI_Megpoid", 2048), 0) == 2048);
    CHECK(speakers::np_idx(registration("asa", 42405), 1) == (1 | 1 << 8));
    CHECK(speakers::np_idx(registration("Mirai_Komachi", 10537), 7) == (7 | 7 << 8));
    CHECK(speakers::np_idx(registration("Fuiro", 1542), 0) == (0 | 6 << 8));
    CHECK(throws_invalid([] { speakers::np_idx(registration("asa", 42405), 226); }));
    CHECK(throws_invalid([] { speakers::np_idx(registration("custom_model", 0), 3); }));
}

void package_unit() {
    fs::path root = fs::temp_directory_path() / "openv6-test-package";
    fs::remove_all(root);
    fs::create_directories(root / "sub");
    auto write = [&](const char* name, const std::string& text) { std::ofstream(root / "sub" / name) << text; };
    write("v6.yaml", "# comment\nformat_version: 1\nbank_id: \"bank\"   # trailing\nlanguage_id: 4\nmodel_directory: ../shared\nwine_prefix: ~/.wine\n");
    write("bank.json", R"({"Components\\bank":{"PModelName":"asa","TModelName":"timbre","NPIdx":42405}})");
    auto package = load_voice_package((root / "sub").string());
    CHECK(package.config.bank_id == "bank");
    CHECK(package.config.language_id == 4);
    CHECK(package.registration.timbre_model == "timbre");
    CHECK(package.registration.np_idx == 42405);
    CHECK(package.registration.timbre_row() == 0xa5 && package.registration.pitch_row() == 0xa5);
    CHECK(package.model_directory() == (root / "shared").string());

    write("v6.yaml", "format_version: 1\nbank_id: bank\nlanguage_id: 2\n");
    CHECK(throws_invalid([&] { load_voice_package((root / "sub").string()); }));
    write("v6.yaml", "format_version: 1\nbank_id: bank\nlanguage_id: 1\n");
    CHECK(load_voice_package((root / "sub").string()).model_directory() == (root / "sub").string());
    write("bank.json", R"({"Components\\bank":{"PModelName":"asa"}})");
    auto fallback = load_voice_package((root / "sub").string()).registration;
    CHECK(fallback.timbre_model == "asa" && fallback.np_idx == 0);
    fs::remove_all(root);
}

void oracle_speakers(const fs::path& oracle) {
    int voices = 0, mismatches = 0;
    for (const auto& file : fs::directory_iterator(oracle / "speakers")) {
        auto doc = json::Document::parse(read_text(file.path()));
        json::Value root = doc->root();
        std::string name = file.path().stem().string();
        auto package = load_voice_package(("/media/zero/T7/UTAUs/" + name));
        const auto& r = package.registration;
        bool ok = package.config.bank_id == root.get_property("bankId").get_string() &&
                  r.timbre_model == root.get_property("TimbreModel").get_string() && r.np_idx == root.get_property("NPIdx").get_int32() &&
                  r.timbre_row() == root.get_property("TimbreRow").get_int32() && r.pitch_row() == root.get_property("PitchRow").get_int32();
        auto available = speakers::available(r);
        json::Value expected = root.get_property("speakers");
        ok = ok && available.size() == expected.array_length();
        for (size_t i = 0; ok && i < available.size(); i++) {
            json::Value row = expected[i];
            ok = available[i].row == row.get_property("Row").get_int32() && available[i].name == row.get_property("Name").get_string();
            int np = 0;
            bool failed = false;
            try {
                np = speakers::np_idx(r, available[i].row);
            } catch (const InvalidDataError&) {
                failed = true;
            }
            bool expect_error = row.get_property("error").kind() == json::Kind::String;
            ok = ok && failed == expect_error && (failed || np == row.get_property("npIdx").get_int32());
        }
        voices++;
        if (!ok) {
            mismatches++;
            std::fprintf(stderr, "speakers mismatch for %s\n", name.c_str());
        }
    }
    std::printf("oracle speakers: %d voices, %d mismatches\n", voices, mismatches);
    CHECK(mismatches == 0);
}

void all_packages() {
    int loaded = 0;
    for (const auto& entry : fs::directory_iterator("/media/zero/T7/UTAUs")) {
        std::string name = entry.path().filename().string();
        if (!entry.is_directory() || !(name.ends_with("_V6") || name == "asa-v6") || !fs::exists(entry.path() / "v6.yaml")) continue;
        try {
            auto package = load_voice_package(entry.path().string());
            CHECK_MSG(fs::is_directory(package.model_directory()), name + ": model directory " + package.model_directory());
            loaded++;
        } catch (const std::exception& e) {
            CHECK_MSG(false, name + ": " + e.what());
        }
    }
    std::printf("voice packages loaded: %d\n", loaded);
}

}

int main() {
    speakers_unit();
    package_unit();
    if (const char* oracle = std::getenv("OPENV6_ORACLE")) oracle_speakers(oracle);
    if (fs::exists("/media/zero/T7/UTAUs")) all_packages();
    if (g_failures) std::fprintf(stderr, "%d failures\n", g_failures);
    return g_failures ? 1 : 0;
}
