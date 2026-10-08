#pragma once

#include <memory>
#include <span>
#include <vector>

#include "openv6/json.hpp"
#include "openv6/types.hpp"

namespace openv6 {

struct FrameSynthOutput {
    std::vector<float> harmonic;
    std::vector<float> noise;
};

class FrameSynth {
public:
    FrameSynth(json::Value model, int speaker);
    ~FrameSynth();

    static std::vector<float> edges(json::Value config);

    FrameSynthOutput run(std::span<const V6Frame> frames);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
