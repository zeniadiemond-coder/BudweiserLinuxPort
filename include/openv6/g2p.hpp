#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "openv6/util.hpp"

namespace openv6::g2p {

std::vector<std::string> assign(std::string_view lyric, std::optional<std::string_view> hint, int language,
                                const std::vector<bool>& extend, const std::string& native_directory);

std::optional<std::vector<std::string>> syllables(std::string_view lyric, int language, const std::string& native_directory);

class EnglishDictionary {
public:
    static const EnglishDictionary& load(const std::string& native_directory);

    std::optional<std::string> find(std::string_view word) const;

private:
    explicit EnglishDictionary(const std::string& path);
    StringMap<std::string> plain_;
    StringMap<std::string> hyphenated_;
};

}
