#include <cstdio>
#include <filesystem>

#include "check.hpp"
#include "openv6/g2p.hpp"
#include "timing_support.hpp"

using namespace openv6;
using timing_support::read_text;

namespace {

using Syllables = std::optional<std::vector<std::string>>;

std::string native_directory() {
    const char* native = std::getenv("OPENV6_NATIVE");
    return native ? native : OPENV6_NATIVE_DIR;
}

bool english_available(const std::string& native) { return std::filesystem::exists(std::filesystem::path(native) / "G2PAManager_ENG.dll"); }

void check_syllables(const char* lyric, int language, const std::string& native, Syllables expected) {
    auto actual = g2p::syllables(lyric, language, native);
    CHECK_MSG(actual == expected, std::string("syllables of ") + lyric);
}

void japanese() {
    check_syllables("か", 0, "", {{"k a"}});
    check_syllables("きゃっと", 0, "", {{"k' a Sil t o"}});
    check_syllables("ウィ", 0, "", {{"w i"}});
    check_syllables("うぃぇ", 0, "", {{"w i e"}});
    check_syllables("ふァ", 0, "", {{"p\\ a"}});
    check_syllables("ヴァ", 0, "", {{"b M a"}});
    check_syllables("てぃー", 0, "", {{"t' i -"}});
    check_syllables("か　き", 0, "", {{"k a k' i"}});
    check_syllables("はー", 0, "", {{"h a", "-"}});
    check_syllables("kyaa", 0, "", {{"k' a", "a"}});
    check_syllables("Ka-", 0, "", {{"k a", "-"}});
    check_syllables("si", 0, "", {{"s i"}});
    check_syllables("nna", 0, "", {{"N\\", "a"}});
    check_syllables("xtsu", 0, "", {{"Sil"}});
    check_syllables("かa", 0, "", {{"k a a"}});
    check_syllables("kka", 0, "", std::nullopt);
    check_syllables("ゔ", 0, "", std::nullopt);
    check_syllables("ヷ", 0, "", std::nullopt);
}

void chinese() {
    check_syllables("zhuang", 4, "", {{"ts` uAN"}});
    check_syllables("Ni", 4, "", {{"n i"}});
    check_syllables("lv", 4, "", {{"l y"}});
    check_syllables("ni hao-", 4, "", {{"n i", "x AU", "-"}});
    check_syllables("-ni", 4, "", {{"- n i"}});
    check_syllables("lü", 4, "", std::nullopt);
    check_syllables("ni3", 4, "", std::nullopt);
    check_syllables("ni-hao", 4, "", std::nullopt);
}

void english(const std::string& native) {
    if (!english_available(native)) {
        std::puts("English dictionary DLL not found; skipping English cases");
        return;
    }
    check_syllables("hello", 1, native, {{"h e", "l @U"}});
    check_syllables("HeLLo", 1, native, {{"h e", "l @U"}});
    check_syllables("strengths", 1, native, {{"s t r e N k T s"}});
    check_syllables("coop", 1, native, {{"kh @U", "Q p"}});
    check_syllables("co-op-", 1, native, {{"kh @U", "Q", "- p"}});
    check_syllables("beautiful--", 1, native, {{"bh j u:", "t I", "f @", "-", "- l"}});
    check_syllables("hel--lo", 1, native, {{"h e", "-", "l @U"}});
    check_syllables("-hello", 1, native, {{"h e l @U"}});
    check_syllables("-world", 1, native, {{"- w @r l d"}});
    check_syllables("beau-tiful", 1, native, std::nullopt);
    check_syllables("blorptastic", 1, native, std::nullopt);
}

void assigns() {
    using V = std::vector<std::string>;
    CHECK(g2p::assign("かん", std::nullopt, 0, {false, true}, "") == (V{"k a", "- N\\"}));
    CHECK(g2p::assign("かん", std::nullopt, 0, {false, false}, "") == (V{"k a N\\", "-"}));
    CHECK(g2p::assign("ni hao", std::nullopt, 4, {false, false}, "") == (V{"n i", "x AU"}));
    CHECK(g2p::assign("ni hao", std::nullopt, 4, {false, true}, "") == (V{"n i x AU", "-"}));
    CHECK(g2p::assign("か", "s a", 0, {false, false}, "") == (V{"s a", "-"}));
    bool threw = false;
    try {
        g2p::assign("ゔ", std::nullopt, 0, {false}, "");
    } catch (const InvalidDataError&) {
        threw = true;
    }
    CHECK(threw);
}

void oracle_file(const std::filesystem::path& path, const std::string& native, int& entries, int& mismatches) {
    auto doc = json::Document::parse(read_text(path));
    int language = doc->root().get_property("language").get_int32();
    for (json::Value entry : doc->root().get_property("results").enumerate_array()) {
        entries++;
        std::string input(entry.get_property("g2pInput").get_string());
        std::optional<std::string> hint;
        if (entry.get_property("hint").kind() == json::Kind::String) hint = std::string(entry.get_property("hint").get_string());
        std::vector<bool> extend;
        for (json::Value e : entry.get_property("extend").enumerate_array()) extend.push_back(e.get_boolean());

        Syllables expected_syllables;
        json::Value syllables = entry.get_property("syllables");
        if (syllables.kind() == json::Kind::Array) {
            expected_syllables = std::vector<std::string>();
            for (json::Value s : syllables.enumerate_array()) expected_syllables->emplace_back(s.get_string());
        }
        std::optional<std::string> expected_error;
        if (entry.get_property("error").kind() == json::Kind::String) expected_error = std::string(entry.get_property("error").get_string());

        std::optional<std::vector<std::string>> assigned;
        std::optional<std::string> error;
        try {
            std::optional<std::string_view> view;
            if (hint) view = *hint;
            assigned = g2p::assign(input, view, language, extend, native);
        } catch (const InvalidDataError& e) {
            error = e.what();
        }
        Syllables actual_syllables = g2p::syllables(input, language, native);

        std::optional<std::vector<std::string>> expected_assigned;
        json::Value assigned_json = entry.get_property("assigned");
        if (assigned_json.kind() == json::Kind::Array) {
            expected_assigned = std::vector<std::string>();
            for (json::Value s : assigned_json.enumerate_array()) expected_assigned->emplace_back(s.get_string());
        }
        bool ok = actual_syllables == expected_syllables && assigned == expected_assigned && error == expected_error;
        if (!ok) {
            mismatches++;
            std::fprintf(stderr, "%s: mismatch for '%s'\n", path.filename().string().c_str(), input.c_str());
        }
    }
}

void oracle(const std::string& native) {
    const char* root = std::getenv("OPENV6_ORACLE");
    if (!root) return;
    int entries = 0, mismatches = 0;
    for (const auto& file : std::filesystem::directory_iterator(std::filesystem::path(root) / "g2p")) {
        oracle_file(file.path(), native, entries, mismatches);
    }
    std::printf("oracle g2p: %d entries, %d mismatches\n", entries, mismatches);
    CHECK(mismatches == 0);
}

}

int main() {
    std::string native = native_directory();
    japanese();
    chinese();
    english(native);
    assigns();
    oracle(native);
    if (g_failures) std::fprintf(stderr, "%d failures\n", g_failures);
    return g_failures ? 1 : 0;
}
