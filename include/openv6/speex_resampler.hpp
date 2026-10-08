#pragma once

#include <span>
#include <vector>

namespace openv6::speex_resampler {

std::vector<float> resample(std::span<const float> input, int in_rate, int out_rate, int output_length);

}
