#include "openv6/frame_driver.hpp"

#include <algorithm>
#include <cmath>

#include "openv6/util.hpp"

namespace openv6::frame_driver {

namespace {

constexpr float AirMax = 1.5873016f;

int frame_of(double seconds) { return static_cast<int>(round_away(seconds * 200)); }

}

float noise_gain(float v) {
    float db = v <= 1.f ? (v <= 0.f ? v * 24.f : v * 12.f) : 12.f - (v - 1.f) * 6.f / (AirMax - 1.f);
    return std::pow(10.f, db * 0.05f);
}

float harmonic_gain(float v) {
    float knee = AirMax * 0.95f;
    if (v < knee) return std::pow(10.f, std::min((v - 1.f) * -12.f / (knee - 1.f), 0.f) * 0.05f);
    return (AirMax - v) * std::pow(10.f, -0.6f) / (AirMax - knee);
}

float dynamics_gain(int dynamics, float offset_db) {
    float db = std::clamp(offset_db, -24.f, 24.f);
    if (dynamics < 64) db = (std::log10(static_cast<float>(dynamics) * 0.015625f) * 14.8f - 0.0778f + db) / 20.f;
    else if (dynamics == 64) db /= 20.f;
    else db = (static_cast<float>(dynamics - 64) / 63.f * 7.6542f + db) / 20.f;
    return std::pow(10.f, db);
}

float bipolar(int value, int min, int max) {
    int v = std::clamp(value, min, max);
    return v < 1 ? static_cast<float>(v) * 0.015625f : static_cast<float>(v) / 63.f;
}

std::vector<AmplitudeSpan> amplitude_spans(std::span<const V6Note> notes, std::span<const V6Phoneme> timed) {
    std::vector<AmplitudeSpan> spans;
    for (size_t i = 0; i < notes.size(); i++) {
        if (notes[i].rest()) continue;
        const V6Phoneme *first = nullptr, *last = nullptr;
        for (const auto& p : timed) {
            if (p.note != static_cast<int>(i)) continue;
            if (!first) first = &p;
            last = &p;
        }
        const auto& e = notes[i].expression;
        spans.push_back({frame_of(first->start), frame_of(last->end), e.amplitude_whole, e.amplitude_start, e.amplitude_end});
    }
    return spans;
}

std::vector<float> expression_ramps(std::span<const AmplitudeSpan> spans, int frames) {
    std::vector<float> ramps(frames, 0.f);
    for (const auto& span : spans) {
        int first = std::clamp(span.first, 0, frames - 1), end = std::clamp(span.last, 0, frames - 1);
        float whole = (span.whole - 0.5f) * 2.f, start = (span.start - 0.5f) * 2.f, finish = (span.end - 0.5f) * 2.f;
        int half = (end - first) / 2, mid = first + half;
        for (int f = first; f < end; f++) {
            float v = f < mid ? lerp(start, whole, static_cast<float>(f - mid + half) / static_cast<float>(half))
                              : lerp(whole, finish, static_cast<float>(f - mid) / static_cast<float>(end - mid));
            ramps[f] = std::clamp(v, -1.f, 1.f);
        }
    }
    return ramps;
}

float control(int expression, int k, float ramp) {
    float x = static_cast<float>(expression - 50) / 50.f;
    if (ramp != 0) x = std::clamp(ramp + x, -1.f, 1.f);
    float kk = static_cast<float>(k) / 3000.f;
    return std::clamp(x - kk * x * x + kk, -1.f, 1.f);
}

std::vector<float> gate(std::span<const V6Phoneme> timed, int frames, bool breaths, int exhalation) {
    auto silent = [&](const V6Phoneme& p) { return p.name == "sil" || (!breaths && p.name == "br"); };
    std::vector<float> gate(frames, 1.f);
    const int count = static_cast<int>(timed.size());
    for (int first = 0; first < count; first++) {
        if (!silent(timed[first])) continue;
        int last = first;
        while (last + 1 < count && silent(timed[last + 1])) last++;
        int s = frame_of(timed[first].start);
        int e = frame_of(timed[last].end);
        bool followed = last < count - 1 && !(breaths && timed[last + 1].name == "br");
        if (e - s > 13) {
            for (int f = std::max(s, 0); f < e && f < frames; f++) {
                if (f < s + 8) {
                    if (first == 0) gate[f] = 0;
                    else if (f > s + 2) gate[f] = 1.f - static_cast<float>(f - s - 2) / 6.f;
                } else if (!followed || f < e - 7) {
                    gate[f] = 0;
                } else if (f < e - 3) {
                    gate[f] = static_cast<float>(f - (e - 7)) * 0.25f;
                }
            }
        }
        first = last;
    }
    if (breaths) {
        float db = static_cast<float>(exhalation - 5) * 4.8f;
        for (const auto& p : timed) {
            if (p.name != "br") continue;
            int s = frame_of(p.start);
            int e = frame_of(p.end);
            for (int f = std::max(s, 0); f < e && f < frames; f++) {
                float k = static_cast<float>(f < s + 12 ? f - s : f < e - 13 ? 12 : e - f - 1);
                gate[f] = std::pow(10.f, k / 12.f * db / 20.f);
            }
        }
    }
    return gate;
}

std::vector<float> breath_harmonics(std::span<const V6Phoneme> timed, int frames) {
    std::vector<float> gain(frames, 1.f);
    for (const auto& p : timed) {
        if (p.name != "br") continue;
        int s = frame_of(p.start);
        int e = frame_of(p.end);
        int half = std::clamp(e - s, 2, 12) / 2;
        for (int f = std::max(s, 0); f < e && f < frames; f++) {
            int k = f < s + half ? f - s : f > e - half ? e - f : -1;
            gain[f] = k < 0 ? 0 : std::pow(10.f, static_cast<float>(k * -120.0 / half) * 0.05f);
        }
    }
    return gain;
}

std::vector<float> mix(std::span<const float> ch0, std::span<const float> ch1, int frames, const std::function<float(int)>& air,
                       const std::function<float(int)>& dynamics, const std::function<float(int)>& gate,
                       const std::function<float(int)>& harmonics) {
    std::vector<float> output(static_cast<size_t>(frames) * Hop);
    const int calls = static_cast<int>(ch0.size()) / Hop;
    float prev_h = 0, prev_n = 0, prev_d = 0, prev_g = 0;
    for (int c = 0; c < calls; c++) {
        int f = std::clamp(c - Latency, 0, frames - 1);
        float v = air(f);
        float h = harmonic_gain(v) * harmonics(f), n = noise_gain(v), d = dynamics(f), g = gate(f);
        for (int i = 0; i < Hop; i++) {
            int64_t idx = static_cast<int64_t>(Hop) * c - Latency * Hop + i;
            if (idx < 0 || idx >= static_cast<int64_t>(output.size())) continue;
            float t = static_cast<float>(i) / Hop;
            float hg = prev_h != h ? lerp(prev_h, h, t) : h;
            float ng = prev_n != n ? lerp(prev_n, n, t) : n;
            float y = hg * ch0[c * Hop + i] + ng * ch1[c * Hop + i];
            float dg = prev_d != d ? lerp(prev_d, d, t) : d;
            if (dg != 1.f) y *= dg;
            float gg = prev_g != g ? lerp(prev_g, g, t) : g;
            if (gg != 1.f) y *= gg;
            output[idx] = y;
        }
        prev_h = h;
        prev_n = n;
        prev_d = d;
        prev_g = g;
    }
    return output;
}

}
