#include "openv6/score_features.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include "openv6/model_file.hpp"

namespace openv6::score_features {

namespace {

constexpr std::string_view PhonemeKey = "edc2ce94290a3b18bc7f745276643253";
constexpr std::string_view FrontendKey = "a4ba655a061e6151f80574e6088492a0";
constexpr std::string_view MinDuration = "f962293ecd429e2bdcc6b736d5fbe03f";
constexpr std::string_view MaxDuration = "52deffcc05ab8eef20105c2860706a51";
constexpr std::string_view NextKey = "3abe629e45940435ab13470c11104d50";
constexpr std::string_view PrevKey = "36cdeb254eea602bbf99240c5287b151";
constexpr std::string_view ClassesKey = "d809560074085c5b6f43bf86bd3bebd4";

const std::vector<std::vector<std::string_view>> ClassLists = {
    {"sil", "Sil", "SIL"},
    {"br", "Br", "BR"},
    {"a", "i", "u", "e", "o", "wo", "ey", "aw", "ay", "ow", "oy", "aa", "uh", "iy", "ih", "ah", "uw", "eh", "ae",
     "er", "ao", "i_n", "uei", "iE_n", "yE_r", "@N", "u@N", "@U", "ia", "i`", "@`", "i\\", "i@U", "aI", "uaI", "y_n",
     "7", "iN", "iE_r", "ua_n", "iUN", "AN", "ei", "AU", "uAN", "a_n", "iAN", "y{_n", "u@_n", "@_n", "uo", "UN", "iAU",
     "ua", "ia+1", "iE_r+1", "yE_r+1", "ua+1", "uo+1", "iAU+1", "i@U+1", "uaI+1", "uei+1", "iAN+1", "iUN+1",
     "iE_n+1", "y{_n+1", "u@N+1", "u@_n+1", "ua_n+1", "uAN+1", "y|CH", "a|CH", "i|CH", "u|CH", "o|CH"},
    {"n", "N", "m", "ny", "my", "ng"},
    {"g", "d", "b", "gy", "dy", "by", "dh"},
    {"z", "zh", "v", "z`"},
    {"j", "jh"},
    {"y", "w", "ia+0", "iE_r+0", "yE_r+0", "ua+0", "uo+0", "iAU+0", "i@U+0", "uaI+0", "uei+0", "iAN+0",
     "iUN+0", "iE_n+0", "y{_n+0", "u@N+0", "u@_n+0", "ua_n+0", "uAN+0"},
    {"r", "ry", "l"},
    {"k", "t", "p", "ky", "ty", "py", "p_h", "k_h"},
    {"s", "sh", "h", "C", "f", "hy", "hh", "th", "s\\"},
    {"ch", "ts", "ts\\", "ts`", "t_h", "ts_h", "ts`_h", "ts\\_h"},
    {"cl"},
};

const StringSet StaticNames = {"7", "@N", "@U", "@_n", "@`", "AN", "AU", "UN", "aI", "a_n", "a|CH", "ei", "i@U", "i@U+0", "i@U+1", "iAN", "iAN+0", "iAN+1", "iAU", "iAU+0", "iAU+1", "iE_n", "iE_n+0", "iE_n+1", "iE_r", "iE_r+0", "iE_r+1", "iN", "iUN", "iUN+0", "iUN+1", "i\\", "i_n", "i`", "ia", "ia+0", "ia+1", "i|CH", "k_h", "o|CH", "p_h", "s\\", "s`", "t_h", "ts\\", "ts\\_h", "ts_h", "ts`", "ts`_h", "u@N", "u@N+0", "u@N+1", "u@_n", "u@_n+0", "u@_n+1", "uAN", "uAN+0", "uAN+1", "ua", "ua+0", "ua+1", "uaI", "uaI+0", "uaI+1", "ua_n", "ua_n+0", "ua_n+1", "uei", "uei+0", "uei+1", "uo", "uo+0", "uo+1", "u|CH", "x", "yE_r", "yE_r+0", "yE_r+1", "y_n", "y{_n", "y{_n+0", "y{_n+1", "y|CH", "z`"};

const StringMap<std::string_view> Replacements = {{"i_n", "ih"}, {"uei", "ey"}, {"iE_n", "eh"}, {"yE_r", "eh"}, {"@N", "aa"}, {"u@N", "aa"}, {"@U", "aa"}, {"ia", "ih"}, {"i`", "iy"}, {"@`", "ae"}, {"i\\", "iy"}, {"i@U", "aa"}, {"aI", "ay"}, {"uaI", "ay"}, {"y_n", "iy"}, {"7", "uw"}, {"iN", "ih"}, {"iE_r", "eh"}, {"ua_n", "aa"}, {"iUN", "uw"}, {"AN", "aa"}, {"ei", "ey"}, {"AU", "aw"}, {"uAN", "aa"}, {"a_n", "aa"}, {"iAN", "aa"}, {"y{_n", "ii"}, {"u@_n", "ae"}, {"@_n", "ae"}, {"uo", "wo"}, {"UN", "M"}, {"iAU", "aa"}, {"ua", "aa"}, {"p_h", "p"}, {"ts\\", "ts"}, {"s\\", "sh"}, {"ts`", "ts"}, {"k_h", "k"}, {"z`", "z"}, {"t_h", "th"}, {"ts_h", "ts"}, {"x", "h"}, {"s`", "sh"}, {"ia+0", "y"}, {"ua+0", "w"}, {"uo+0", "w"}, {"ia+1", "ah"}, {"ua+1", "ah"}, {"uo+1", "aa"}, {"y|CH", "i"}, {"a|CH", "a"}, {"i|CH", "i"}, {"u|CH", "u"}, {"o|CH", "o"}};

float duration_feature(double duration, float lo, float hi) {
    float a = std::log(lo), b = std::log(hi);
    float x = std::log(std::max(static_cast<float>(duration), lo));
    return std::clamp((x - a + (x - a)) / (b - a) - 1.f, -1.f, 1.f);
}

std::vector<float> phoneme_rows(json::Value model, std::span<const V6Phoneme> phonemes, int& width) {
    auto cfg = model.get_property(FrontendKey);
    auto tbl = table(model);
    float lo = cfg.get_property(MinDuration).get_single(), hi = cfg.get_property(MaxDuration).get_single();
    int prev = cfg.get_property(PrevKey).get_int32(), next = cfg.get_property(NextKey).get_int32();
    int classes = cfg.get_property(ClassesKey).get_int32();
    int bas = classes + 4, block = static_cast<int>(tbl.size()) + bas;
    int n = static_cast<int>(phonemes.size()), contexts = 1 + prev + next;
    width = block * contexts;
    std::vector<float> rows(static_cast<size_t>(n) * width);
    std::vector<int> ctx(contexts);
    for (int i = 0; i < n; i++) {
        ctx[0] = i;
        for (int k = 0; k < prev; k++) ctx[1 + k] = i - prev + k;
        for (int k = 0; k < next; k++) ctx[1 + prev + k] = i + 1 + k;
        for (int c = 0; c < contexts; c++) {
            const auto& p = phonemes[std::clamp(ctx[c], 0, n - 1)];
            size_t o = static_cast<size_t>(i) * width + static_cast<size_t>(c) * block;
            rows[o] = duration_feature(p.duration(), lo, hi);
            rows[o + (p.name.starts_with('!') ? 1 : 2)] = 1;
            rows[o + bas - classes + p.cls] = 1;
            rows[o + bas + phoneme_id(tbl, p.name)] = 1;
        }
    }
    return rows;
}

}

const StringSet& nuclei() {
    static const StringSet set = {
        "ih", "eh", "ae", "aa", "ah", "uh", "ax", "iy", "uw", "ao", "er", "ey", "ay", "oy", "ow", "aw", "a", "i", "u",
        "e", "o", "wo", "N", "i_n", "uei", "iE_n", "yE_r", "@N", "u@N", "@U", "ia", "i`", "@`", "i\\", "i@U", "aI", "uaI",
        "y_n", "7", "iN", "iE_r", "ua_n", "iUN", "AN", "ei", "AU", "uAN", "a_n", "iAN", "y{_n", "u@_n", "@_n", "uo", "UN",
        "iAU", "ua", "ia+1", "iE_r+1", "yE_r+1", "ua+1", "uo+1", "iAU+1", "i@U+1", "uaI+1", "uei+1", "iAN+1", "iUN+1",
        "iE_n+1", "y{_n+1", "u@N+1", "u@_n+1", "ua_n+1", "uAN+1", "y|CH", "a|CH", "i|CH", "u|CH", "o|CH", "SIL", "Sil",
        "BR", "Br",
    };
    return set;
}

int phoneme_class(std::string_view name) {
    for (size_t i = 0; i < ClassLists.size(); i++) {
        const auto& list = ClassLists[i];
        if (std::find(list.begin(), list.end(), name) != list.end()) return static_cast<int>(i);
    }
    return 13;
}

PhonemeTable table(json::Value model) { return ModelFile::phoneme_table(model.get_property(PhonemeKey)); }

int phoneme_id(const PhonemeTable& table, std::string_view name) {
    std::string resolved;
    if (StaticNames.contains(name)) {
        if (table.contains(std::string(name))) {
            resolved = name;
        } else {
            auto it = Replacements.find(name);
            resolved = it != Replacements.end() ? std::string(it->second) : std::string(name);
        }
    } else {
        for (char c : name) {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) resolved += c;
        }
    }
    if (resolved == "SIL" || resolved == "Sil") return 0;
    if (resolved == "BR" || resolved == "Br") return 1;
    auto it = table.find(resolved);
    return it != table.end() ? it->second : 0;
}

std::vector<V6Phoneme> merge_dashes(std::span<const V6Phoneme> phonemes) {
    std::vector<V6Phoneme> result;
    for (const auto& p : phonemes) {
        if (p.name == "-" && !result.empty()) result.back().end = p.end;
        else result.push_back(p);
    }
    return result;
}

std::vector<V6Phoneme> breath_to_silence(std::span<const V6Phoneme> phonemes) {
    std::vector<V6Phoneme> result;
    for (const auto& p : phonemes) {
        if (p.name != "br") {
            result.push_back(p);
        } else if (!result.empty() && result.back().name == "sil") {
            result.back().end = p.end;
        } else {
            result.push_back(p);
            result.back().name = "sil";
        }
    }
    return result;
}

std::vector<float> frame_records(const PhonemeTable& table, std::span<const V6Phoneme> phonemes, bool notes) {
    using Record = std::array<float, RecordWidth>;
    int n = static_cast<int>(phonemes.size());
    std::vector<Record> recs(n);
    int cat = 0, pid = phoneme_id(table, phonemes[0].name);
    for (int i = 0; i < n; i++) {
        const auto& p = phonemes[i];
        bool dash = p.name == "-";
        if (!dash) pid = phoneme_id(table, p.name);
        bool b48 = notes && p.note_start, b49 = notes && p.nucleus;
        if (b49) cat = 1;
        else if (b48) cat = 0;
        else if (cat == 1 || cat == 2) cat = 2;
        float offset = static_cast<float>(p.start - static_cast<float>(static_cast<int>(p.start / Period)) * Period);
        recs[i] = {static_cast<float>(pid), b48 ? 1.f : 0.f, b49 ? 1.f : 0.f, nuclei().contains(p.name) ? 1.f : 0.f,
                   dash ? 1.f : 0.f, static_cast<float>(p.duration()), offset, notes ? p.pitch : 0.f,
                   static_cast<float>(cat), 0.f, 0.f};
    }
    int frames = static_cast<int>(std::ceil(phonemes.back().end / Period));
    Record none{};
    none[5] = 1;
    std::vector<float> output(static_cast<size_t>(frames) * 3 * RecordWidth);
    auto put = [&](int f, int ctx, const Record& values) {
        std::memcpy(&output[(static_cast<size_t>(f) * 3 + ctx) * RecordWidth], values.data(), sizeof(float) * RecordWidth);
    };
    auto field = [&](int f, int ctx, int index) -> float& { return output[(static_cast<size_t>(f) * 3 + ctx) * RecordWidth + index]; };
    int first = 0;
    for (int i = 0; i < n; i++) {
        int last = std::min(static_cast<int>(std::ceil(phonemes[i].end / Period)), frames);
        for (int k = 0; k < last - first; k++) {
            int f = first + k;
            put(f, 0, recs[i]);
            field(f, 0, 6) = k * Period + recs[i][6];
            put(f, 1, i < n - 1 ? recs[i + 1] : none);
            field(f, 1, 6) = 0;
            put(f, 2, i > 0 ? recs[i - 1] : none);
            field(f, 2, 6) = 0;
        }
        first = std::max(first, last);
    }
    for (int k = 0; k < frames - first; k++) {
        put(first + k, 0, recs[n - 1]);
        field(first + k, 0, 6) = k * Period + recs[n - 1][6];
        put(first + k, 1, none);
    }
    for (int i = 0; i < n; i++) {
        if (recs[i][8] != 1) continue;
        float total = static_cast<float>(phonemes[i].duration());
        for (int j = i + 1; j < n && recs[j][8] >= 2; j++) total += static_cast<float>(phonemes[j].duration());
        float start = static_cast<float>(phonemes[i].start);
        if (total < 1e-5f) continue;
        int a = std::clamp(static_cast<int>(start / Period), 0, frames);
        int b = std::clamp(static_cast<int>((total + start) / Period + 1.f), 0, frames);
        for (int f = a; f < b; f++) {
            field(f, 0, 10) = f * Period - start;
            field(f, 0, 9) = total;
        }
    }
    return output;
}

std::vector<float> frame_rows(json::Value model, std::span<const V6Phoneme> phonemes, int& frames, int& width) {
    int row_width;
    auto rows = phoneme_rows(model, phonemes, row_width);
    frames = static_cast<int>(phonemes.back().end / Period);
    width = row_width + 4;
    std::vector<float> output(static_cast<size_t>(frames) * width);
    int index = static_cast<int>(phonemes.size()) - 1, first = 0;
    std::vector<int> phoneme_of(frames, index);
    std::vector<float> pos(frames), dur(frames, 1.f);
    for (int i = 0; i < static_cast<int>(phonemes.size()); i++) {
        const auto& p = phonemes[i];
        int last = std::min(static_cast<int>(std::ceil(p.end / Period)), frames);
        for (int k = 0; k < last - first; k++) {
            phoneme_of[first + k] = i;
            pos[first + k] = static_cast<float>(static_cast<double>(k * Period) + (first * static_cast<double>(Period) - p.start));
            dur[first + k] = static_cast<float>(p.duration());
        }
        first = std::max(first, last);
    }
    for (int f = 0; f < frames; f++) {
        std::memcpy(&output[static_cast<size_t>(f) * width], &rows[static_cast<size_t>(phoneme_of[f]) * row_width],
                    sizeof(float) * row_width);
        float ratio = pos[f] / dur[f];
        size_t o = static_cast<size_t>(f) * width + row_width;
        output[o] = ratio;
        output[o + 1] = 1.f - ratio;
        output[o + 2] = 1.f - std::tanh(pos[f] * 10.f);
        output[o + 3] = std::tanh((pos[f] - dur[f]) * 10.f) + 1.f;
    }
    return output;
}

}
