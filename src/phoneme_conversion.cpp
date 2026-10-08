#include "openv6/phoneme_conversion.hpp"

#include <sstream>
#include <stdexcept>

#include "openv6/util.hpp"

namespace openv6::phoneme_conversion {

namespace {
StringMap<std::string> pairs(std::initializer_list<std::string_view> entries) {
    StringMap<std::string> map;
    for (auto entry : entries) {
        size_t space = entry.find(' ');
        map[std::string(entry.substr(0, space))] = std::string(entry.substr(space + 1));
    }
    return map;
}

const StringMap<std::string>& japanese() {
    static const auto map = pairs({"a a", "i i", "M u", "e e", "o o", "k k", "k' ky", "g g", "g' gy", "s s", "S sh", "t t",
        "t' ty", "d d", "d' dy", "tS ch", "ts ts", "n n", "J ny", "z z", "Z j", "dz z", "dZ j", "h h", "h\\ h", "C hy",
        "p p", "p' py", "p\\ f", "p\\' f", "b b", "b' by", "m m", "m' my", "j y", "4 r", "4' ry", "w w", "N\\ N", "- -",
        "Sil sil"});
    return map;
}

const StringMap<std::string>& japanese_before_i() {
    static const auto map = pairs({"k' k", "t' t", "J n", "C h", "m' m", "4' r", "g' g", "d' d", "b' b", "p' p"});
    return map;
}

const StringMap<std::string>& english() {
    static const auto map = pairs({"{ ae", "@ ah", "@r er", "@U ow", "aI ay", "aU aw", "b b", "bh b", "d d", "D dh", "dh d",
        "dZ jh", "e eh", "eI ey", "f f", "g g", "gh g", "h hh", "I ih", "i: iy", "j y", "k k", "kh k", "l l", "l0 l", "m m",
        "n n", "N ng", "O: ao", "OI oy", "p p", "ph p", "Q aa", "r r", "s s", "S sh", "t t", "T th", "th t", "tS ch", "U uh",
        "u: uw", "v v", "V ah", "w w", "z z", "Z zh", "- -", "Sil sil"});
    return map;
}

const StringMap<std::vector<std::string>>& english_split() {
    static const StringMap<std::vector<std::string>> map = {
        {"I@", {"ih", "r"}}, {"O@", {"ao", "r"}}, {"Q@", {"aa", "r"}}, {"U@", {"uh", "r"}}, {"e@", {"eh", "r"}},
    };
    return map;
}

const StringMap<std::string>& chinese() {
    static const StringMap<std::string> map = [] {
        StringMap<std::string> m;
        for (auto p : split("N aa ae ah ao aw ay b by ch cl d dh dy e eh er ey f g gy h hh hy ih iy j jh k ky l m my n ng ny "
                            "ow oy p py r ry s sh t th ts ty uh uw v w wo z zh 7 @N @U @_n @` AN AU UN aI a_n ei iN i\\ i_n i` "
                            "k_h p_h s\\ s` t_h ts\\ ts\\_h ts_h ts` ts`_h x y_n z` -", ' ', true))
            m[std::string(p)] = std::string(p);
        for (auto& [k, v] : pairs({"y y|CH", "a a|CH", "i i|CH", "u u|CH", "o o|CH", "Sil sil"})) m[k] = v;
        return m;
    }();
    return map;
}

const StringMap<std::vector<std::string>>& chinese_split() {
    static const StringMap<std::vector<std::string>> map = [] {
        StringMap<std::vector<std::string>> m;
        for (auto p : split("ia i@U iAN iAU iE_n iE_r iUN u@N u@_n uAN ua uaI ua_n uei uo yE_r y{_n", ' ', true))
            m[std::string(p)] = {std::string(p) + "+0", std::string(p) + "+1"};
        return m;
    }();
    return map;
}

const std::string* find(const StringMap<std::string>& map, std::string_view key) {
    auto it = map.find(key);
    return it == map.end() ? nullptr : &it->second;
}
}

std::vector<ModelPhoneme> to_model(std::string_view phonemes, int language) {
    auto tokens = split(phonemes, ' ', false);
    std::vector<ModelPhoneme> result;
    auto add = [&](std::string_view token, int i, const StringMap<std::string>& map, const StringMap<std::vector<std::string>>& parts) {
        if (auto* name = find(map, token)) {
            result.push_back({*name, i});
        } else if (auto it = parts.find(token); it != parts.end()) {
            for (const auto& part : it->second) result.push_back({part, i});
        } else {
            result.push_back({"sil", i});
        }
    };
    for (int i = 0; i < static_cast<int>(tokens.size()); i++) {
        std::string_view token = tokens[i];
        switch (language) {
            case 0: {
                std::string_view next = i + 1 < static_cast<int>(tokens.size()) ? tokens[i + 1] : "Sil";
                const std::string* plain = next == "i" ? find(japanese_before_i(), token) : nullptr;
                if (plain) {
                    result.push_back({*plain, i});
                } else {
                    const std::string* name = find(japanese(), token);
                    result.push_back({name ? *name : "sil", i});
                }
                break;
            }
            case 1:
                add(token, i, english(), english_split());
                break;
            case 4:
                add(token, i, chinese(), chinese_split());
                break;
            default:
                throw std::out_of_range("language");
        }
    }
    return result;
}

}
