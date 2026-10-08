#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace openv6 {

struct V6Phoneme {
    double start;
    double end;
    std::string name;
    float pitch;
    bool note_start;
    bool nucleus;
    int cls;
    int note = -1;

    double duration() const { return end - start; }
};

struct AiNoteExpression {
    float pitch_fine = 0.5f;
    float pitch_drift_start = 0.5f;
    float pitch_drift_end = 0.5f;
    float pitch_scaling_center = 0.5f;
    float pitch_scaling_origin = 0.5f;
    float pitch_transition_start = 0.5f;
    float pitch_transition_end = 0.5f;
    float amplitude_whole = 0.5f;
    float amplitude_start = 0.5f;
    float amplitude_end = 0.5f;
    float vibrato_leading_depth = 0.5f;
    float vibrato_following_depth = 0.5f;
};
static_assert(sizeof(AiNoteExpression) == 12 * sizeof(float));

struct V6Note {
    double start;
    double end;
    std::optional<float> pitch;
    std::vector<std::string> phonemes;
    int meter = 4;
    float position = 0;
    bool flag = false;
    int language = 0;
    std::optional<std::vector<std::optional<int>>> phoneme_ticks;
    float pitch_offset = 0;
    AiNoteExpression expression;
    int vibrato_depth = 0;
    int vibrato_rate = 0;
    bool ai_vibrato = true;
    double vibrato_start = 0;

    bool rest() const { return !pitch.has_value(); }
};

struct V6PitchEdit {
    std::vector<int64_t> ticks;
    std::vector<float> midi_cents;
    double tick_seconds;
    double origin_seconds;
};

struct V6Frame {
    std::vector<float> vec;
    float f0;
    float control;
    float alpha;
};

struct V6Controls {
    std::optional<std::vector<int>> expression;
    std::optional<std::vector<int>> dynamics;
    std::optional<std::vector<int>> air;
    std::optional<std::vector<int>> character;

    int expression_at(int frame) const { return at(expression, frame, 50); }
    int dynamics_at(int frame) const { return at(dynamics, frame, 64); }
    int air_at(int frame) const { return at(air, frame, 0); }
    int character_at(int frame) const { return at(character, frame, 0); }

private:
    static int at(const std::optional<std::vector<int>>& curve, int frame, int fallback) {
        if (!curve || curve->empty()) return fallback;
        return (*curve)[std::clamp(frame, 0, static_cast<int>(curve->size()) - 1)];
    }
};

struct TimeSpan {
    double start;
    double end;
};

}
