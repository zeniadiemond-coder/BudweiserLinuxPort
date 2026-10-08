#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "openv6/local_voice.hpp"

namespace openv6::phonemizer {

std::optional<std::string> clean_lyric(int language_id, std::string_view lyric);
bool is_rest(int language_id, std::optional<std::string_view> phonetic_hint, std::string_view lyric);
bool is_breath(std::string_view lyric);
bool is_vowel_extension(std::string_view lyric);

void phonemize_score(V6Score& score, const std::string& native_directory);

}
