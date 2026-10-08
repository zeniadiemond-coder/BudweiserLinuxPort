#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace openv6::phoneme_conversion {

struct ModelPhoneme {
    std::string name;
    int source;
};

std::vector<ModelPhoneme> to_model(std::string_view phonemes, int language);

}
