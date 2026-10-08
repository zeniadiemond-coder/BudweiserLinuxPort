#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "openv6/util.hpp"

namespace openv6 {

class VtmgModel {
public:
    explicit VtmgModel(std::span<const uint8_t> container);
    static VtmgModel load(const std::string& path);

    int conditions() const { return conditions_; }

    const StringMap<float>& consonant_weights() const { return consonant_weights_; }

    std::vector<float> forward(std::span<const float> features, int frames, int condition) const;

private:
    StringMap<std::vector<float>> weights_;
    StringMap<float> consonant_weights_;
    int conditions_ = 0;
};

}
