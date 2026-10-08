#pragma once

#include <functional>
#include <span>
#include <vector>

#include "openv6/types.hpp"

namespace openv6::frame_driver {

inline constexpr int Hop = 160, Latency = 9;

float noise_gain(float v);

float harmonic_gain(float v);

float dynamics_gain(int dynamics, float offset_db = 0);

float bipolar(int value, int min, int max);

struct AmplitudeSpan {
    int first;
    int last;
    float whole;
    float start;
    float end;
};

std::vector<AmplitudeSpan> amplitude_spans(std::span<const V6Note> notes, std::span<const V6Phoneme> timed);

std::vector<float> expression_ramps(std::span<const AmplitudeSpan> spans, int frames);

float control(int expression, int k, float ramp = 0);

std::vector<float> gate(std::span<const V6Phoneme> timed, int frames, bool breaths, int exhalation = 5);

std::vector<float> breath_harmonics(std::span<const V6Phoneme> timed, int frames);

std::vector<float> mix(std::span<const float> ch0, std::span<const float> ch1, int frames, const std::function<float(int)>& air,
                       const std::function<float(int)>& dynamics, const std::function<float(int)>& gate,
                       const std::function<float(int)>& harmonics);

}
