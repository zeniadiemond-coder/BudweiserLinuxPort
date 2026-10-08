#pragma once

#include <limits>
#include <span>
#include <vector>

#include "openv6/types.hpp"

namespace openv6::vsm_pitch {

inline constexpr float Gap = -std::numeric_limits<float>::max();

std::vector<float> resample(std::span<const float> x, float ratio);

struct KeyPoint {
    int x;
    float v;
};

std::vector<float> constrained_spline(std::span<const KeyPoint> points);

std::vector<float> note_pitch(std::span<const float> pitch, std::span<const float> residual, int midi, const AiNoteExpression& expression);

void add_vibrato(std::span<float> pitch, std::span<const float> residual, int depth, int rate, bool ai, int start,
                 const AiNoteExpression& expression);

struct PitchNote {
    int lo;
    int hi;
    int midi;
    float pitch_offset;
    AiNoteExpression expression;
    int vibrato_depth;
    int vibrato_rate;
    bool ai_vibrato;
    int vibrato_start;
};

std::vector<PitchNote> pitch_notes(std::span<const V6Note> notes, std::span<const V6Phoneme> timed);

struct DirectPitchCurve {
    int first;
    std::vector<float> cents;
};

DirectPitchCurve direct_pitch(const V6PitchEdit& edit);

std::vector<float> f0_curve(std::span<const float> pitch, std::span<const float> residual, std::span<const PitchNote> notes,
                            int frames, const V6PitchEdit* edit = nullptr);

}
