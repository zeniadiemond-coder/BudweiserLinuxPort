#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "openv6/json.hpp"
#include "openv6/speakers.hpp"
#include "openv6/synthesizer.hpp"

namespace openv6 {

struct V6ScoreVibrato {
    int depth = 0;
    int rate = 50;
    std::optional<int64_t> durationTick;
    bool ai = true;
};

struct V6ScoreNote {
    int64_t startTick;
    int64_t durationTick;
    int pitch;
    std::string lyric;
    std::string phonemes;
    std::optional<int> languageId;
    std::optional<std::vector<std::optional<int>>> phonemeTicks;
    std::optional<std::string> phoneticHint;
    AiNoteExpression aiParams;
    std::optional<V6ScoreVibrato> vibrato;
};

std::span<const std::pair<std::string_view, float AiNoteExpression::*>> ai_param_names();

struct V6ControllerEvent {
    int64_t tick;
    int value;
};

struct V6Controller {
    std::string name;
    std::vector<V6ControllerEvent> events;
};

struct V6Breath {
    int64_t startTick;
    int64_t durationTick;
};

struct V6Score {
    double tempoBpm;
    int languageId;
    std::vector<V6ScoreNote> notes;
    std::optional<std::vector<float>> pitch;
    std::optional<std::vector<int64_t>> pitchTicks;
    std::optional<std::vector<V6Controller>> controllers;
    std::optional<std::vector<V6Breath>> breaths;
};

struct V6Pitch {
    double frameMs;
    std::vector<float> pitches;
};

V6Score parse_score(json::Value root);
V6Score parse_score(std::string_view json_text);

void validate_score(const V6Score& score, bool phonemized = true);

struct ScoreNotes {
    std::vector<V6Note> notes;
    std::vector<int> index;
    std::vector<std::vector<int>> sources;
};

ScoreNotes build_notes(const V6Score& score, double tick);

struct LocalRenderResult {
    std::vector<float> samples;
    V6Pitch pitch;
};

class V6LocalVoice {
public:
    V6LocalVoice(const std::string& models, const V6Registration& registration, int language, std::optional<int> np_idx = std::nullopt);

    LocalRenderResult render(const V6Score& score) const;

    std::vector<std::vector<int>> phoneme_positions(const V6Score& score) const;

private:
    std::unique_ptr<V6Synthesizer> synthesizer_;
    int condition_;
};

}
