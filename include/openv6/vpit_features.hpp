#pragma once

#include <memory>
#include <span>
#include <vector>

#include "openv6/json.hpp"
#include "openv6/types.hpp"

namespace openv6 {

class VpitFeatures {
public:
    static constexpr int Width = 93;

    explicit VpitFeatures(json::Value model);
    ~VpitFeatures();

    std::vector<float> rows(std::span<const V6Phoneme> phonemes, int& frames) const;

private:
    std::vector<float> encoder_rows(std::span<const V6Phoneme> phonemes, int& frames) const;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
