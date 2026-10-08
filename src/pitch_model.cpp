#include "openv6/pitch_model.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

#include "openv6/networks.hpp"
#include "openv6/score_features.hpp"

namespace openv6 {

uint32_t Xorshift128::next() {
    uint32_t t = x_ ^ (x_ << 11);
    x_ = y_;
    y_ = z_;
    z_ = w_;
    w_ = w_ ^ (w_ >> 19) ^ t ^ (t >> 8);
    return w_;
}

double Xorshift128::gauss(double mean, double scale) {
    double s = 0;
    for (int i = 0; i < 12; i++) s += next();
    return (s - 25769803770.0) * scale / 4294967295.0 + mean;
}

namespace {

constexpr std::string_view NetKey = "20b8dac0362664a54f087204590292d1";
constexpr std::string_view Config = "a4ba655a061e6151f80574e6088492a0";
constexpr std::string_view Low = "205192c4ec6717232a8e5e2515841a42", High = "f98f2b4f4eab060dad241d3e4fb1c4dc";
constexpr std::string_view DerivativeWidth = "4736d5027136dcf78ab9c95afeb80c60";

float norm(float v, float lo, float hi) { return std::clamp((v - lo) * 2.0f / (hi - lo) - 1.0f, -1.0f, 1.0f); }

std::vector<float> derivative_taps(float width, float period) {
    int half = static_cast<int>(width / period) / 2, n = 2 * half;
    if (n + 1 < 2) return {1.0f};
    std::vector<float> t;
    for (int k = half; k >= -half; k--) t.push_back((std::cos(k * 6.2831855f / n) * 0.46f + 0.54f) * k);
    float sum = 0;
    for (float v : t) sum += std::fabs(v);
    for (float& v : t) v /= sum * 2.0f;
    return t;
}

struct Biquad {
    static inline const struct Coefficients {
        float b0, b1, a1, a2;
        Coefficients() {
            float k = std::tan(1.2566371f) * 2.0f, k2 = k * k, den = k * 4.0f + k2 + 4.0f;
            b0 = k2 / den;
            b1 = (k2 + k2) / den;
            a1 = -(k2 + k2 - 8.0f) / den;
            a2 = -(k2 - k * 4.0f + 4.0f) / den;
        }
    } c{};
    float x1 = 0, x2 = 0, y1 = 0, y2 = 0;

    float run(float x) {
        float y = x1 * c.b1 + x * c.b0 + c.b0 * x2 + y1 * c.a1 + c.a2 * y2;
        x2 = x1;
        x1 = x;
        y2 = y1;
        y1 = y;
        return y;
    }
};

}

struct PitchModel::Impl {
    CausalStack net;
    float lo, hi, noise_scale;
    int speaker;
    std::vector<float> taps, tap_history;
    Xorshift128 rng;
    std::deque<std::vector<float>> pitch_ring, residual_ring;
    Biquad smooth[2];
    float hp0, hp1, hp2;
    int steps = 0;
    float hp_x = 0, hp_y = 0, prev = 0;
    int hold = 0;

    Impl(json::Value model, std::span<const float> first_row, float noise, int speaker_)
        : net(model.get_property(NetKey)), noise_scale(noise), speaker(speaker_) {
        auto cfg = model.get_property(Config);
        lo = cfg.get_property(Low).get_single();
        hi = cfg.get_property(High).get_single();
        taps = derivative_taps(cfg.get_property(DerivativeWidth).get_single(), score_features::Period);
        tap_history.assign(taps.size(), 0.0f);
        float t = std::tan(0.25132743f);
        float g = 1.0f / (t + 1.0f);
        hp0 = g;
        hp1 = -g;
        hp2 = (1.0f - t) * g;
        float last = 0;
        for (int i = 0; i < net.receptive_field(); i++) last = network_step(first_row, first_row[0], 0).f0;
        prev = last;
    }

    float derivative(float x) {
        std::copy_backward(tap_history.begin(), tap_history.end() - 1, tap_history.end());
        tap_history[0] = x;
        double sum = 0;
        for (size_t i = 0; i < taps.size(); i++) sum += static_cast<double>(taps[i]) * tap_history[i];
        return static_cast<float>(sum);
    }

    PitchStep network_step(std::span<const float> row, float previous, float noise) {
        float ar[2] = {norm(previous, lo, hi), std::tanh(derivative(previous)) + noise};
        float cond[65] = {};
        std::copy_n(row.begin() + 29, 64, cond);
        auto y = net.step(ar, cond, speaker);
        size_t n = y.size() / 2;
        std::vector<float> pitch(n), residual(n);
        for (size_t k = 0; k < n; k++) {
            residual[k] = (y[2 * k + 1] + 1.0f) * 0.5f * 1200.0f - 600.0f;
            pitch[k] = (y[2 * k] + 1.0f) * 0.5f * (hi - lo) + lo + residual[k];
        }
        pitch_ring.push_back(std::move(pitch));
        residual_ring.push_back(std::move(residual));
        int u = steps & 15, s = static_cast<int>(pitch_ring.size()) - 1 - u;
        float p = pitch_ring[s][u], r = residual_ring[s][u];
        if (u < 8) {
            float w = u * 0.125f;
            float old_p = s >= 16 ? pitch_ring[s - 16][u + 16] : 0.0f;
            float old_r = s >= 16 ? residual_ring[s - 16][u + 16] : 0.0f;
            p = (1.0f - w) * old_p + w * p;
            r = (1.0f - w) * old_r + w * r;
        }
        steps++;
        if (pitch_ring.size() > 64) {
            pitch_ring.pop_front();
            residual_ring.pop_front();
        }
        return {p, r};
    }

    PitchStep step(std::span<const float> row) {
        float noise = static_cast<float>(rng.gauss() * (std::clamp(noise_scale, 0.0f, 1.0f) * 5.0f));
        auto [p, residual] = network_step(row, prev, noise);
        float note = row[0];
        float bas = p - residual;
        int cat = static_cast<int>(round_even(row[10]));
        float next_vowel = row[15], prev_vowel = row[23];
        float pos = std::fmax(row[11], 0.0f), dur = row[12];
        float attack = std::fmin(std::fmax((0.6f - std::fmax(0.2f - dur, 0.0f) * 0.2f / 0.2f) * dur, 0.1f), dur * 0.6f);
        float release = std::fmax(dur * 0.65f, std::fmin(dur - 0.1f, dur * 0.8f));
        if (!(cat == 1 || cat == 2) || pos < 0 || dur <= pos) {
            if (note < bas) bas -= (bas - note) * 0.7f;
        } else {
            if (pos < 0.005f) {
                hp_x = hp_y = bas;
                hold = 0;
            }
            float hp = hp_x * hp1 + hp0 * bas + hp_y * hp2;
            hp_x = bas;
            hp_y = hp;
            float w;
            if (pos < attack) {
                w = pos / attack * (pos / attack);
                if (note < bas && prev_vowel <= 0.5f) w = std::fmin(w + 0.6f, 1.0f);
            } else if (pos < release) {
                w = 1.0f;
            } else {
                w = std::sqrt(std::fmax(1.0f - (pos - release) / (dur - release), 0.0f));
            }
            if (dur < 0.25f && next_vowel > 0.5f && hold > 1) w *= static_cast<float>(std::pow(0.8999999761581421, hold));
            bas = bas - w * (bas - note) + hp * std::fmin(w, pos);
        }
        float smoothed = smooth[1].run(smooth[0].run(bas));
        float output = smoothed + residual;
        if (std::fabs(smoothed - note) < 10.0f) hold++;
        prev = output;
        return {output, residual};
    }
};

PitchModel::PitchModel(json::Value model, std::span<const float> first_row, float noise, int speaker)
    : impl_(std::make_unique<Impl>(model, first_row, noise, speaker)) {}

PitchModel::~PitchModel() = default;

PitchStep PitchModel::step(std::span<const float> row) { return impl_->step(row); }

}
