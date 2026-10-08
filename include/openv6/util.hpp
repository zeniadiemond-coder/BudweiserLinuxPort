#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace openv6 {

struct InvalidDataError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct StringHash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const { return std::hash<std::string_view>{}(s); }
};

template <class T>
using StringMap = std::unordered_map<std::string, T, StringHash, std::equal_to<>>;
using StringSet = std::unordered_set<std::string, StringHash, std::equal_to<>>;

inline double round_even(double x) { return std::nearbyint(x); }
inline float round_even(float x) { return std::nearbyintf(x); }
inline double round_away(double x) { return std::round(x); }
inline float round_away(float x) { return std::roundf(x); }

inline std::string read_text(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw InvalidDataError("Cannot read " + path.string() + ".");
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

inline std::string_view trim(std::string_view s) {
    constexpr const char* Blank = " \t\r\n\v\f";
    size_t first = s.find_first_not_of(Blank);
    if (first == std::string_view::npos) return {};
    return s.substr(first, s.find_last_not_of(Blank) - first + 1);
}

inline std::vector<std::string_view> split(std::string_view text, char separator, bool keep_empty) {
    std::vector<std::string_view> parts;
    size_t start = 0;
    while (true) {
        size_t next = text.find(separator, start);
        std::string_view part = text.substr(start, next == std::string_view::npos ? std::string_view::npos : next - start);
        if (keep_empty || !part.empty()) parts.push_back(part);
        if (next == std::string_view::npos) return parts;
        start = next + 1;
    }
}

inline float lerp(float a, float b, float t) {
    if ((a <= 0 && b >= 0) || (a >= 0 && b <= 0)) return t * b + (1 - t) * a;
    if (t == 1) return b;
    float x = std::fma(t, b - a, a);
    return (t > 1) == (b > a) ? std::max(b, x) : std::min(b, x);
}

}
