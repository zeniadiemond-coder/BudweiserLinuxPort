#pragma once

#include <memory>
#include <span>
#include <vector>

#include "openv6/model_file.hpp"
#include "openv6/types.hpp"
#include "openv6/vtmg_model.hpp"

namespace openv6 {

struct PitchCurveResult {
    std::vector<float> f0;
    std::vector<float> residual;
};

struct RenderResult {
    std::vector<float> samples;
    std::vector<float> f0;
};

class V6Synthesizer {
public:
    static constexpr int SampleRate = 44100;

    V6Synthesizer(std::shared_ptr<const ModelFile> timbre, std::shared_ptr<const ModelFile> pitch,
                  std::shared_ptr<const VtmgModel> timing, int np_idx, int param_exp = 3000, int param_dyn = 3000);

    std::vector<V6Phoneme> time(std::span<const V6Note> notes, int condition, double tick) const;

    std::vector<float> frame_vectors(std::span<const V6Phoneme> timed, int& frames) const;

    PitchCurveResult pitch_curve(std::span<const V6Phoneme> phonemes) const;

    std::vector<float> f0(std::span<const V6Note> notes, std::span<const V6Phoneme> timed, const V6PitchEdit* pitch_edit = nullptr) const;

    RenderResult render(std::span<const V6Note> notes, std::span<const V6Phoneme> timed, const V6Controls& controls = {},
                        const V6PitchEdit* pitch_edit = nullptr, std::span<const TimeSpan> breaths = {}, int exhalation = 5) const;

private:
    std::shared_ptr<const ModelFile> timbre_, pitch_;
    std::shared_ptr<const VtmgModel> timing_;
    int timbre_speaker_, pitch_speaker_, expression_k_;
    float dynamics_offset_;
};

}
