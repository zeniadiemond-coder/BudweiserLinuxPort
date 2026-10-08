#pragma once

#include <array>
#include <span>
#include <vector>

#include "openv6/types.hpp"
#include "openv6/vtmg_model.hpp"

namespace openv6::timing {

inline constexpr int Width = 26 + 4 * 14;

inline constexpr std::array<std::array<int, 5>, 3> Conditions = {{{8, 10, 15, 17, 19}, {2, 5, 3, 0, 1}, {20, 2, 8, 1, 24}}};

std::vector<float> rows(std::span<const V6Note> notes, int& rows);

std::vector<V6Phoneme> phonemes(std::span<const V6Note> notes, std::span<const float> outputs, const StringMap<float>& weights);

std::vector<V6Phoneme> run(const VtmgModel& model, std::span<const V6Note> notes, int condition);

std::vector<V6Phoneme> edit(std::span<const V6Phoneme> timed, std::span<const V6Note> notes, double tick);

struct Section {
    double origin;
    std::vector<V6Phoneme> phonemes;
};

std::vector<Section> sections(std::span<const V6Phoneme> timed);

}
