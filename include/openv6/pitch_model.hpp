#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "openv6/json.hpp"

namespace openv6 {

class Xorshift128 {
public:
    uint32_t next();
    double gauss(double mean = 0, double scale = 1);

private:
    uint32_t x_ = 0xa1d31f49, y_ = 0x857194d4, z_ = 0x4a82ab01, w_ = 0xe3b21af0;
};

struct PitchStep {
    float f0;
    float residual;
};

class PitchModel {
public:
    PitchModel(json::Value model, std::span<const float> first_row, float noise, int speaker);
    ~PitchModel();

    PitchStep step(std::span<const float> row);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
