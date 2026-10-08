#include "openv6/synthesizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>

#include "openv6/frame_driver.hpp"
#include "openv6/frame_synth.hpp"
#include "openv6/networks.hpp"
#include "openv6/parallel.hpp"
#include "openv6/pitch_model.hpp"
#include "openv6/score_features.hpp"
#include "openv6/speex_resampler.hpp"
#include "openv6/timing.hpp"
#include "openv6/vpit_features.hpp"
#include "openv6/vsm_pitch.hpp"

namespace openv6 {

namespace {

constexpr float PitchNoise = 0.1f;

int to_frame(double seconds) { return static_cast<int>(round_away(seconds * 200)); }

int resampled(size_t samples) { return static_cast<int>(std::ceil(static_cast<double>(samples) * V6Synthesizer::SampleRate / 32000)); }

template <class F>
auto parallel_map(size_t count, F&& f) {
    std::vector<decltype(f(size_t{}))> results(count);
    parallel_for(static_cast<int>(count), [&](int begin, int end) {
        for (int i = begin; i < end; i++) results[i] = f(i);
    });
    return results;
}

struct SectionVectors {
    int origin;
    int frames;
    std::vector<float> vec;
};

struct SectionMix {
    std::vector<float> dry;
    std::vector<float> wet;
};

}

V6Synthesizer::V6Synthesizer(std::shared_ptr<const ModelFile> timbre, std::shared_ptr<const ModelFile> pitch,
                             std::shared_ptr<const VtmgModel> timing, int np_idx, int param_exp, int param_dyn)
    : timbre_(std::move(timbre)),
      pitch_(std::move(pitch)),
      timing_(std::move(timing)),
      timbre_speaker_(np_idx & 0xff),
      pitch_speaker_((np_idx >> 8) & 0xff),
      expression_k_(param_exp - 3000),
      dynamics_offset_((param_dyn - 3000) / 100.f) {}

std::vector<V6Phoneme> V6Synthesizer::time(std::span<const V6Note> notes, int condition, double tick) const {
    return timing::edit(timing::run(*timing_, notes, condition), notes, tick);
}

std::vector<float> V6Synthesizer::frame_vectors(std::span<const V6Phoneme> timed, int& frames) const {
    auto phonemes = score_features::merge_dashes(timed);
    int width;
    auto rows = score_features::frame_rows(timbre_->root(), phonemes, frames, width);
    auto encoded = GatedStack(timbre_->root().get_property(keys::Encoder), width, 64, 40).run_aligned(rows, frames);
    auto records = score_features::frame_records(score_features::table(timbre_->root()), phonemes, false);
    std::vector<float> vec(static_cast<size_t>(frames) * 47);
    for (int f = 0; f < frames; f++) {
        std::memcpy(&vec[static_cast<size_t>(f) * 47], &records[static_cast<size_t>(f) * 3 * score_features::RecordWidth], 7 * sizeof(float));
        std::memcpy(&vec[static_cast<size_t>(f) * 47 + 7], &encoded[static_cast<size_t>(f) * 40], 40 * sizeof(float));
    }
    return vec;
}

PitchCurveResult V6Synthesizer::pitch_curve(std::span<const V6Phoneme> phonemes) const {
    int frames;
    auto rows = VpitFeatures(pitch_->root()).rows(phonemes, frames);
    PitchModel model(pitch_->root(), std::span<const float>(rows).first(VpitFeatures::Width), PitchNoise, pitch_speaker_);
    PitchCurveResult result{std::vector<float>(frames), std::vector<float>(frames)};
    for (int f = 0; f < frames; f++) {
        auto step = model.step(std::span<const float>(rows).subspan(static_cast<size_t>(f) * VpitFeatures::Width, VpitFeatures::Width));
        result.f0[f] = step.f0;
        result.residual[f] = step.residual;
    }
    return result;
}

std::vector<float> V6Synthesizer::f0(std::span<const V6Note> notes, std::span<const V6Phoneme> timed, const V6PitchEdit* pitch_edit) const {
    int part_frames = to_frame(notes.back().end);
    std::vector<float> pitch_part(part_frames, vsm_pitch::Gap), residual_part(part_frames, vsm_pitch::Gap);
    auto sections = timing::sections(timed);
    auto curves = parallel_map(sections.size(), [&](size_t i) { return pitch_curve(sections[i].phonemes); });
    for (size_t i = 0; i < sections.size(); i++) {
        int at = to_frame(sections[i].origin);
        int n = std::min(static_cast<int>(curves[i].f0.size()), part_frames - at);
        std::copy_n(curves[i].f0.begin(), n, pitch_part.begin() + at);
        std::copy_n(curves[i].residual.begin(), n, residual_part.begin() + at);
    }
    return vsm_pitch::f0_curve(pitch_part, residual_part, vsm_pitch::pitch_notes(notes, timed), part_frames, pitch_edit);
}

RenderResult V6Synthesizer::render(std::span<const V6Note> notes, std::span<const V6Phoneme> timed, const V6Controls& controls,
                                   const V6PitchEdit* pitch_edit, std::span<const TimeSpan> breaths, int exhalation) const {
    using frame_driver::Hop;
    double end = notes.back().end;
    int part_frames = to_frame(end);
    auto f0_task = std::async(std::launch::async, [&] { return f0(notes, timed, pitch_edit); });
    auto sections = timing::sections(timed);
    auto vectors = parallel_map(sections.size(), [&](size_t i) {
        SectionVectors s;
        s.vec = frame_vectors(sections[i].phonemes, s.frames);
        s.origin = to_frame(sections[i].origin);
        return s;
    });
    auto part_f0 = f0_task.get();
    bool with_breaths = !breaths.empty();
    auto gate = frame_driver::gate(timed, part_frames, false);
    auto ramps = frame_driver::expression_ramps(frame_driver::amplitude_spans(notes, timed), part_frames);
    std::vector<float> breath_gate, harmonics;
    if (with_breaths) {
        breath_gate = frame_driver::gate(timed, part_frames, true, exhalation);
        harmonics = frame_driver::breath_harmonics(timed, part_frames);
    }
    auto mixed = parallel_map(vectors.size(), [&](size_t i) {
        const auto& s = vectors[i];
        auto at = [&](int f) { return std::min(f + s.origin, part_frames - 1); };
        std::vector<V6Frame> inputs;
        inputs.reserve(s.frames + frame_driver::Latency);
        for (int c = 0; c < s.frames + frame_driver::Latency; c++) {
            int f = std::min(c, s.frames - 1);
            inputs.push_back({{s.vec.begin() + static_cast<size_t>(f) * 47, s.vec.begin() + static_cast<size_t>(f + 1) * 47},
                              std::clamp(part_f0[at(f)], -6900.f, 6200.f), frame_driver::control(controls.expression_at(at(f)), expression_k_, ramps[at(f)]),
                              0.3f * frame_driver::bipolar(controls.character_at(at(f)), -64, 63)});
        }
        auto channels = FrameSynth(timbre_->root(), timbre_speaker_).run(inputs);
        auto mix = [&](const std::vector<float>& g, const std::vector<float>* h) {
            return frame_driver::mix(
                channels.harmonic, channels.noise, s.frames, [&](int f) { return frame_driver::bipolar(controls.air_at(at(f)), -100, 100); },
                [&](int f) { return frame_driver::dynamics_gain(controls.dynamics_at(at(f)), dynamics_offset_); }, [&](int f) { return g[at(f)]; },
                [&](int f) { return h ? (*h)[at(f)] : 1.f; });
        };
        SectionMix result;
        result.dry = mix(gate, nullptr);
        if (with_breaths) result.wet = mix(breath_gate, &harmonics);
        return result;
    });

    struct Placed {
        int offset;
        std::vector<float> samples;
    };
    auto outputs = parallel_map(vectors.size(), [&](size_t i) {
        int start = vectors[i].origin * Hop;
        const auto& dry = mixed[i].dry;
        std::vector<float> section(i == vectors.size() - 1 ? static_cast<size_t>(part_frames) * Hop - start : dry.size());
        std::copy_n(dry.begin(), std::min(section.size(), dry.size()), section.begin());
        if (with_breaths) {
            const auto& wet = mixed[i].wet;
            for (const auto& span : breaths) {
                int b = static_cast<int>(round_even(span.start * 32000)), e = static_cast<int>(round_even(span.end * 32000));
                int ramp = std::min(320, (e - b) / 2);
                for (int j = std::max(b - start, 0); j < std::min(e - start, static_cast<int>(wet.size())); j++) {
                    int t = start + j;
                    float w = ramp == 0 ? 1.f : std::min(1.f, std::min(t - b, e - 1 - t) / static_cast<float>(ramp));
                    section[j] += w * (wet[j] - section[j]);
                }
            }
        }
        return Placed{resampled(static_cast<size_t>(start)),
                      speex_resampler::resample(section, 32000, V6Synthesizer::SampleRate, resampled(section.size()))};
    });
    size_t length = static_cast<size_t>(round_away(end * SampleRate));
    for (const auto& o : outputs) length = std::max(length, o.offset + o.samples.size());
    std::vector<float> output(length);
    for (const auto& o : outputs)
        for (size_t j = 0; j < o.samples.size(); j++) output[o.offset + j] += o.samples[j];
    return {std::move(output), std::move(part_f0)};
}

}
