#pragma once

#include <span>
#include <string_view>
#include <vector>

#include "openv6/json.hpp"
#include "openv6/model_file.hpp"
#include "openv6/types.hpp"

namespace openv6::score_features {

inline constexpr float Period = 160.f / 32000.f;
inline constexpr int RecordWidth = 11;

const StringSet& nuclei();

int phoneme_class(std::string_view name);

PhonemeTable table(json::Value model);

int phoneme_id(const PhonemeTable& table, std::string_view name);

std::vector<V6Phoneme> merge_dashes(std::span<const V6Phoneme> phonemes);

std::vector<V6Phoneme> breath_to_silence(std::span<const V6Phoneme> phonemes);

std::vector<float> frame_records(const PhonemeTable& table, std::span<const V6Phoneme> phonemes, bool notes);

std::vector<float> frame_rows(json::Value model, std::span<const V6Phoneme> phonemes, int& frames, int& width);

}
