#include "openv6/frame_synth.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>

#include "openv6/fft.hpp"
#include "openv6/model_file.hpp"
#include "openv6/networks.hpp"
#include "openv6/pitch_model.hpp"

namespace openv6 {

namespace {

using Complex = std::complex<double>;

constexpr float Fs = 32000.f;
constexpr float Pi = 3.1415927f;
constexpr float TwoPi = 6.2831855f;
constexpr float FloatTiny = 1.1754944e-38f;
constexpr int Hop = 160, Warmup = 77, FeatureDelay = 2, DecodeDelay = 6, Mask = (1 << 20) - 1;

constexpr std::string_view FrontendKey = "a4ba655a061e6151f80574e6088492a0";
constexpr std::string_view AnalysisKey = "a6d8c9aaef6372d25b26a8d1924776aa";
constexpr std::string_view RangeKey = "7a90c360132d8ebf6f4d4a6f7a002ba6";
constexpr std::string_view Bands = "81580c0610f9cab4a99e1254028175e6", LowKey = "597abe5c91fad4df0ac43693d3c1dffe";
constexpr std::string_view HighKey = "11752ac1d5abcf1d58c585e0e02aee8c", WarpKey = "de62ed5a302c6d0248b600effc6919eb";
constexpr std::string_view WindowKey = "23d0c1bc1cc7a4017306e8ec01cc5286", HannWindow = "b8421b2df71c2003122543a863983ff6";

double sin_d(float x) { return std::sin(static_cast<double>(x)); }
double cos_d(float x) { return std::cos(static_cast<double>(x)); }

class Lanes32 {
public:
    Lanes32() {
        for (int i = 0; i < 32; i++) state_[i] = static_cast<uint32_t>(i + 0x92d68ca2L);
    }

    void draw(int n, std::vector<float>& result) {
        int blocks = (n + 31) / 32;
        result.resize(static_cast<size_t>(blocks) * 32);
        for (int b = 0; b < blocks * 32; b += 32) {
            std::array<float, 32> acc{};
            for (int r = 0; r < 12; r++) {
                for (int i = 0; i < 32; i++) {
                    uint32_t s = state_[i];
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    state_[i] = s;
                    acc[i] += static_cast<int32_t>(s);
                }
            }
            for (int i = 0; i < 32; i++) result[b + i] = acc[i] / 4294967296.f;
        }
        result.resize(n);
    }

private:
    std::array<uint32_t, 32> state_;
};

class ArInput {
public:
    ArInput(std::vector<std::vector<float>> maxima, std::vector<std::vector<float>> minima, float noise)
        : maxima_(std::move(maxima)), minima_(std::move(minima)), noisy_(noise - 0.59900004f > 0) {
        for (const auto& m : maxima_) total_ += static_cast<int>(m.size());
    }

    void run(std::span<const float> source, std::vector<float>& output) {
        output.resize(total_);
        int start = 0;
        for (size_t g = 0; g < maxima_.size(); g++) {
            int n = static_cast<int>(maxima_[g].size());
            double floor_db = noisy_ ? std::abs(rng_.gauss()) * 12.0 - 140.0 : -140.0;
            float floor = std::pow(10.f, static_cast<float>(floor_db) * 0.1f);
            for (int i = 0; i < n; i++) {
                float v = std::pow(10.f, source[start + i] * 0.1f) + floor;
                v = static_cast<float>(std::log10(static_cast<double>(std::max(v, FloatTiny)))) * 10.f - minima_[g][i];
                output[start + i] = std::clamp((v + v) / (maxima_[g][i] - minima_[g][i]) - 1.f, -1.f, 1.f);
            }
            if (countdown_ < 1) {
                for (int i = 0; i < 12; i++) rng_.next();
                countdown_ = static_cast<int>(rng_.next() % 30 + 10);
            } else {
                countdown_--;
            }
            double jitter = noisy_ ? 0.6000000238418579 : 0.10000000149011612;
            for (int i = 0; i < n; i++) output[start + i] = static_cast<float>(rng_.gauss() * jitter + output[start + i]);
            start += n;
        }
    }

private:
    std::vector<std::vector<float>> maxima_, minima_;
    bool noisy_;
    int total_ = 0;
    Xorshift128 rng_;
    int countdown_ = 0;
};

class HarmonicRing {
public:
    static constexpr int Size = 4096, Width = 192;

    HarmonicRing() : rows_(static_cast<size_t>(Size) * Width) {}

    void advance(int t, float f_hz) {
        static constexpr float Poly[4] = {0.004333095f, -0.07943434f, 0.64589286f, -1.570791f};
        int blocks = std::min(24, static_cast<int>(32000.f * 0.5f / f_hz * 0.125f + 1.f));
        int n = 8 * blocks;
        uint32_t q = static_cast<uint32_t>(static_cast<int>(round_even(static_cast<float>(phase_) * 16777216.f)));
        float* row = &rows_[static_cast<size_t>(t & (Size - 1)) * Width];
        for (int k = 0; k < n; k++) {
            uint32_t v = static_cast<uint32_t>(k + 1) * q;
            float x = static_cast<int>(v & 0x7fffff) * 2.3841858e-07f + -1.f;
            float x2 = x * x;
            float y = (((x2 * Poly[0] + Poly[1]) * x2 + Poly[2]) * x2 + Poly[3]) * x;
            row[k] = std::bit_cast<float>(std::bit_cast<uint32_t>(y) ^ ((v >> 23) << 31));
        }
        phase_ += f_hz / 32000.0;
        if (phase_ > 1.0) phase_ -= 1.0;
    }

    std::span<const float> row(int t) const { return {&rows_[static_cast<size_t>(t & (Size - 1)) * Width], Width}; }

private:
    std::vector<float> rows_;
    double phase_ = 0;
};

float catmull_rom(float t, float p0, float p1, float p2, float p3) {
    return ((((p3 - p0) * 0.5f + (p1 - p2) * 1.5f) * t + (p0 - p1 * 2.5f + p2 + p2 - p3 * 0.5f)) * t + (p2 - p0) * 0.5f) * t + p1;
}

float all_pass_hz(float f) {
    float w = f * TwoPi / Fs;
    float a = static_cast<float>(std::atan(static_cast<double>(static_cast<float>(sin_d(w)) * 0.f / (1.f - static_cast<float>(cos_d(w)) * 0.f))));
    return (a + a + w) * Fs / TwoPi;
}

double magnitude(Complex c) { return std::hypot(c.real(), c.imag()); }

void band_analysis(std::span<const float> spectrum, std::span<const float> edges, int fft, bool add_floor, std::span<float> output,
                   float floor = 1e-7f) {
    for (size_t b = 0; b < edges.size() - 2; b++) {
        float e0 = edges[b], e1 = edges[b + 1], e2 = edges[b + 2];
        float x0 = fft * e0 / Fs, x1 = fft * e1 / Fs, x2 = fft * e2 / Fs;
        float norm = 2.f / (e2 - e0), s = 0.f;
        for (int k = static_cast<int>(std::ceil(x0)); k <= static_cast<int>(x2); k++) {
            float w = std::min((k - x0) / (x1 - x0), 1.f - (k - x1) / (x2 - x1));
            s += w * norm * spectrum[k];
        }
        float v = add_floor ? floor + s : std::max(floor, s);
        output[b] = static_cast<float>(std::log10(static_cast<double>(v))) * 20.f;
    }
}

struct DecoderOutput {
    std::vector<float> shaped, noise, feedback;
};

class FrameDecoder {
public:
    static constexpr int Frame = 512, Bins = Frame / 2 + 1;

    FrameDecoder(int width_a, int width_b, int harmonics, float alpha, float low, float high, std::vector<float> edges16, bool hann,
                 bool scale_by_pitch)
        : scale_by_pitch_(scale_by_pitch), width_a_(width_a), width_b_(width_b), harmonics_(harmonics), alpha_(alpha),
          edges16_(std::move(edges16)), fft_(Frame), curve_a_(width_a + 3), curve_b_(width_b + 3), buffer_(Frame + 2) {
        warp_low_ = warp(low);
        warp_high_ = warp(high);
        float sum = 0;
        std::vector<float> h(320);
        for (int k = 0; k < 320; k++) {
            h[k] = hann ? static_cast<float>(0.5 - 0.5 * std::cos(2 * M_PI * k / 320)) : 1.f;
            sum += h[k];
        }
        for (int k = 0; k < 320; k++) window512_[96 + k] = h[k] * 2.f / sum;
        for (int j = 0; j < Bins; j++) bin_warp_[j] = warp(all_pass_hz(j * Fs / Frame));
        for (int u = 0; u < Frame; u++) frame_warp_[u] = warp(u * Fs / Frame);
        peak_low_ = warp(1500.f);
        peak_high_ = warp(4000.f);
    }

    float warp(float f) const {
        float w = f * TwoPi / Fs;
        float s = static_cast<float>(sin_d(w)), c = static_cast<float>(cos_d(w));
        float a = static_cast<float>(std::atan(static_cast<double>(s * alpha_ / (1.f - c * alpha_))));
        return (a + a + w) / Pi;
    }

    DecoderOutput run(std::span<const float> head, float cents, const HarmonicRing& ring, int start, Lanes32& noise_rng) {
        float f0 = 440.f * std::pow(2.f, cents / 1200.f);
        update_curves(head);
        harmonic_amplitudes(f0);
        for (int j = 0; j < Bins; j++) {
            float a = curve_value(curve_a_, bin_warp_[j]);
            harm_db_[j] = a;
            noise_db_[j] = a + std::min(curve_value(curve_b_, bin_warp_[j]), 140.f);
        }
        DecoderOutput out;
        noise_frame(noise_rng, out.noise);
        int top = std::min(harmonics_ - 1, static_cast<int>(Fs * 0.5f / f0));
        std::vector<float> harm(Frame);
        std::span<const float> amps(amps_.data(), top);
        for (int i = 0; i < Frame; i++) harm[i] = native_math::dot(ring.row(start - Frame / 2 + i).first(top), amps);
        auto& noise = out.noise;
        double power = 0;
        for (int i = 0; i < Frame; i++) power += static_cast<double>(harm[i] + noise[i]) * (harm[i] + noise[i]);
        double p = 10 * std::log10(std::max(power / Frame, static_cast<double>(FloatTiny)));
        if (p > -6.02) {
            std::fill(harm.begin(), harm.end(), 0.f);
            std::fill(noise.begin(), noise.end(), 0.f);
        } else if (p > -12.04) {
            float a = std::pow(10.f, static_cast<float>(-(p + 12.04) * 0.75 * 0.05));
            for (int i = 0; i < Frame; i++) {
                harm[i] *= a;
                noise[i] *= a;
            }
        }
        envelope(f0);
        phase_angles(f0);
        forward(harm, Frame / 2);
        for (int j = 0; j < Bins; j++) {
            double angle = theta_[j];
            Complex rotation(std::cos(angle), std::sin(angle));
            Complex v = spectrum_[j];
            spectrum_[j] = Complex(v.real() * rotation.real() - v.imag() * rotation.imag(),
                                   v.imag() * rotation.real() + v.real() * rotation.imag());
        }
        inverse(Frame / 2, out.shaped);
        noise_rng.draw(Frame, dither_);
        float level = first_ ? 1e-4f : 5e-5f;
        first_ = false;
        std::array<float, Frame> windowed;
        for (int i = 0; i < Frame; i++) windowed[i] = (noise[i] + out.shaped[i] + level * dither_[i]) * window512_[i];
        forward(windowed, 0);
        std::array<float, Bins> mag;
        for (int j = 0; j < Bins; j++) {
            double m = magnitude(spectrum_[j]);
            mag[j] = static_cast<float>(std::sqrt(m * m + 1e-14));
        }
        out.feedback.resize(edges16_.size() - 2);
        band_analysis(mag, edges16_, Frame, true, out.feedback);
        return out;
    }

private:
    int curve_index(float warped, int width, float& t) const {
        float x = (std::clamp(warped, warp_low_, warp_high_) - warp_low_) * (width - 1) / (warp_high_ - warp_low_);
        int i = static_cast<int>(x);
        t = x - i;
        return i;
    }

    float curve_value(const std::vector<float>& curve, float warped) const {
        int width = static_cast<int>(curve.size()) - 3;
        float t;
        int i = curve_index(warped, width, t);
        float v = catmull_rom(t, curve[i], curve[i + 1], curve[i + 2], curve[i + 3]);
        return std::min(v, std::max(curve[i + 1], curve[i + 2]) + 3.01f);
    }

    static void pad(std::vector<float>& curve) {
        int width = static_cast<int>(curve.size()) - 3;
        curve[0] = curve[1];
        curve[width + 1] = curve[width + 2] = curve[width];
    }

    void suppress_peaks(std::vector<float>& a) const {
        int width = static_cast<int>(a.size()) - 3, last = width - 1;
        float lo = peak_low_, hi = peak_high_, span = warp_high_ - warp_low_;
        for (float floor : {-60.f, -45.f}) {
            std::vector<float> src = a;
            std::vector<float> mask(a.size());
            for (int i = 1; i < last; i++) {
                float pos = i * span / last + warp_low_;
                float c = src[i + 1];
                float left = floor == -60.f ? src[i] : src[i - 1];
                float right = floor == -60.f ? src[i + 2] : src[i + 3];
                if (!(lo < pos && floor < c && left + 3.f < c && right + 3.f < c)) continue;
                float shoulder = std::max(left, right) + 3.f;
                float gain = std::min(1.f, (c - shoulder) / 3.f) * std::min(1.f, (c - floor) / 12.f);
                gain = std::min(1.f, (pos - lo) / (hi - lo) * 4.f * gain);
                if (floor == -60.f) {
                    a[i + 1] -= (1.f - mask[i]) * (c - shoulder) * gain;
                    mask[i + 1] = 1.f;
                } else {
                    for (int j = i; j < i + 3; j++) a[j] -= (1.f - mask[j]) * std::max(0.f, (src[j] - shoulder) * gain);
                    mask[i] = mask[i + 1] = mask[i + 2] = 1.f;
                }
            }
        }
    }

    void update_curves(std::span<const float> head) {
        for (int i = 0; i < width_a_; i++) curve_a_[i + 1] = 1.f / (native_math::exp(head[i] * -2.f) + 1.f) * 160.f - 140.f;
        suppress_peaks(curve_a_);
        pad(curve_a_);
        for (int i = 0; i < width_b_; i++) curve_b_[i + 1] = 1.f / (native_math::exp(head[width_a_ + i] * -2.f) + 1.f) * 240.f - 120.f;
        pad(curve_b_);
        int o = width_a_ + width_b_;
        float p[4], d[4];
        for (int i = 0; i < 4; i++) {
            p[i] = 1.f / (native_math::exp(-head[o + i]) + 1.f) * TwoPi;
            d[i] = Pi / std::max(std::exp(head[o + 4 + i]) * 400.f, 1e-6f);
        }
        constexpr int map[7] = {0, 0, 1, 2, 3, 3, 3};
        for (int i = 0; i < 7; i++) {
            phase_[i] = p[map[i]];
            decay_[i] = d[map[i]];
        }
    }

    void harmonic_amplitudes(float f0) {
        amps_.assign(harmonics_, 0.f);
        float scale = scale_by_pitch_ ? std::sqrt(f0 / Fs) : 1.f, peak = -300.f;
        for (int k = 1; k < harmonics_; k++) {
            float fk = all_pass_hz(k * f0);
            if (Fs * 0.5f < fk) break;
            float v = std::pow(10.f, curve_value(curve_a_, warp(fk)) * 0.05f) * scale;
            if (fk < 4000.f) peak = std::max(peak, v);
            amps_[k - 1] = fk <= 10000.f ? v : std::min(v, peak);
        }
    }

    void forward(std::span<const float> x, int roll) {
        for (int i = 0; i < Frame; i++) buffer_[i] = x[((i + roll) % Frame + Frame) % Frame];
        fft_.forward(std::span<const double>(buffer_.data(), Frame), spectrum_);
    }

    void inverse(int roll, std::vector<float>& result) {
        fft_.inverse(spectrum_, std::span<double>(buffer_.data(), Frame));
        result.resize(Frame);
        for (int i = 0; i < Frame; i++) result[((i + roll) % Frame + Frame) % Frame] = static_cast<float>(buffer_[i]);
    }

    void noise_frame(Lanes32& noise_rng, std::vector<float>& result) {
        float guard = std::numeric_limits<float>::lowest();
        for (int j = 0; j < static_cast<int>(7000.f / Fs * Frame); j++) guard = std::max(guard, std::max(harm_db_[j], noise_db_[j]));
        for (int j = static_cast<int>(15000.f / Fs * Frame); j < Bins; j++) {
            float v = noise_db_[j];
            if (v > -32.f || (v > -44.f && v > guard + 12.f)) {
                noise_db_.fill(-140.f);
                break;
            }
        }
        noise_rng.draw(Frame, white_);
        forward(white_, Frame / 2);
        double mean = 0;
        int count = 0;
        for (int j = 0; j < Bins; j++) {
            float f = j * Fs / Frame;
            if (f >= 4000.f && f < 10000.f) {
                mean += std::pow(10.f, noise_db_[j] * 0.1f);
                count++;
            }
        }
        float high = static_cast<float>(std::log10(static_cast<double>(std::max(static_cast<float>(mean / count), FloatTiny)))) * 10.f + 12.f;
        for (int j = 0; j < Bins; j++) {
            float cap = j * Fs / Frame <= 12000.f ? 0.f : high;
            spectrum_[j] *= static_cast<double>(std::pow(10.f, std::min(noise_db_[j], cap) * 0.05f));
        }
        spectrum_[0] = spectrum_[1] = 0;
        inverse(Frame / 2, result);
    }

    void envelope(float f0) {
        std::array<float, Frame> full;
        int width = static_cast<int>(curve_a_.size()) - 3;
        for (int u = 0; u < Frame; u++) {
            float t;
            int i = curve_index(frame_warp_[u], width, t);
            full[u] = std::min(curve_value(curve_a_, frame_warp_[u]), std::max(curve_a_[i + 1], curve_a_[i + 2]) + 3.f);
        }
        double pos = Frame * f0 / Fs;
        int lo = std::min(static_cast<int>(pos), Bins - 2);
        double at_f0 = pos >= Bins - 1 ? harm_db_[Bins - 1] : harm_db_[lo] + (harm_db_[lo + 1] - harm_db_[lo]) * (pos - lo);
        int half = static_cast<int>(round_even(256000.0 / Fs));
        std::vector<float> w(2 * half + 1);
        double wsum = 0;
        for (int t = -half; t <= half; t++) {
            w[t + half] = static_cast<float>(0.54 + 0.46 * std::cos(2 * M_PI * t / (2 * half)));
            wsum += w[t + half];
        }
        for (int u = 0; u < Bins; u++) {
            double sum = 0;
            for (int t = -half; t <= half; t++) {
                int idx = std::abs(u + t);
                double v = idx * static_cast<double>(Fs) / Frame < f0 ? at_f0 : full[std::min(idx, Frame - 1)];
                sum += v * w[t + half];
            }
            env_[u] = static_cast<float>(sum / wsum);
        }
    }

    void phase_angles(float f0) {
        int lim = std::clamp(static_cast<int>(f0 * 512.f / Fs), 1, 256);
        for (int u = 0; u < Bins; u++) {
            float x = (std::clamp(frame_warp_[u], warp_low_, warp_high_) - warp_low_) * 3.f / (warp_high_ - warp_low_);
            int i = static_cast<int>(x);
            float t = x - i;
            float ph = (phase_[i + 2] - phase_[i + 1]) * t + phase_[i + 1];
            float dc = (decay_[i + 2] - decay_[i + 1]) * t + decay_[i + 1];
            float slope = u < lim ? env_[lim + 1] - env_[lim - 1]
                        : u < 0xf6 ? env_[std::min(u + 1, 256)] - env_[std::max(u - 1, 0)]
                                   : env_[0xf6] - env_[0xf4];
            theta_[u] = slope * dc * Fs / 512.f + ph;
        }
    }

    bool scale_by_pitch_;
    int width_a_, width_b_, harmonics_;
    float alpha_, warp_low_ = 0, warp_high_ = 0;
    std::vector<float> edges16_;
    RealFourierTransform fft_;
    std::vector<float> curve_a_, curve_b_;
    std::array<float, 7> phase_{}, decay_{};
    std::array<float, Frame> window512_{};
    std::vector<double> buffer_;
    std::array<Complex, Bins> spectrum_{};
    std::array<float, Bins> harm_db_{}, noise_db_{}, bin_warp_{}, env_{}, theta_{};
    std::array<float, Frame> frame_warp_{};
    std::vector<float> amps_, white_, dither_;
    float peak_low_ = 0, peak_high_ = 0;
    bool first_ = true;
};

float pitch_norm(float cents) {
    double p = std::clamp(static_cast<double>(cents), -4800.0, 3600.0);
    return static_cast<float>((p - -4800.0) * 2.0 / (3000.0 - -4800.0) - 1.0);
}

float interpolate(const V6Frame& prev, const V6Frame& cur, int s) { return (cur.f0 - prev.f0) * (static_cast<float>(s) / Hop) + prev.f0; }

}

struct FrameSynth::Impl {
    struct Segment {
        bool present = false;
        int start = 0;
        std::vector<float> samples;
    };

    GatedStack controls;
    CausalStack bf8;
    ArInput ar;
    FrameDecoder decoder;
    std::vector<float> edges80;
    int speaker;
    Xorshift128 jitter;
    Lanes32 noise_rng;
    HarmonicRing ring;
    std::array<float, 2 * Hop> window;
    std::array<float, 1024> analysis_window;
    std::vector<float> harm_out = std::vector<float>(1 << 20), noise_out = std::vector<float>(1 << 20);
    Segment ola[3][2];
    RealFourierTransform analysis_fft{1024};
    std::vector<double> analysis_buffer = std::vector<double>(1026);
    std::array<Complex, 513> spectrum;
    std::array<double, 513> warped_bins;
    float warped_alpha = std::numeric_limits<float>::quiet_NaN();

    static bool hann(json::Value config) { return config.get_property(WindowKey).get_string() == HannWindow; }

    static std::vector<std::vector<float>> tensors(json::Value list) {
        std::vector<std::vector<float>> result;
        for (auto v : list.enumerate_array()) result.push_back(ModelFile::floats(v));
        return result;
    }

    Impl(json::Value model, int speaker_)
        : controls(model.get_property(keys::Controls), 36, 128, 128),
          bf8(model.get_property(keys::Synthesis)),
          ar(tensors(model.get_property(RangeKey)[1]), tensors(model.get_property(RangeKey)[0]),
             model.get_property("2e24c75e83acd7307571ebfcf965cb26").get_property("75ffe1f947489bf0c52a4fb6a0c27ee7").get_single()),
          decoder(make_decoder(model)),
          speaker(speaker_) {
        auto analyses = model.get_property(AnalysisKey);
        edges80 = FrameSynth::edges(analyses[1]);
        for (int k = 0; k < 2 * Hop; k++) window[k] = static_cast<float>(0.5 - 0.5 * std::cos(2 * M_PI * k / (2 * Hop)));
        float sum = 0;
        for (int k = 0; k < 1024; k++) {
            analysis_window[k] = static_cast<float>(0.5 - 0.5 * std::cos(2 * M_PI * k / 1024));
            sum += analysis_window[k];
        }
        for (int k = 0; k < 1024; k++) analysis_window[k] /= sum * 0.5f;
    }

    static FrameDecoder make_decoder(json::Value model) {
        auto cfg = model.get_property(FrontendKey);
        auto analysis = model.get_property(AnalysisKey)[0];
        return FrameDecoder(cfg.get_property("cc02b901802a2dab4ad6a87db7381441").get_int32(),
                            cfg.get_property("cfbf47bb45b089531ff1876a3672d224").get_int32(),
                            cfg.get_property("ccd641e3a663fb48b75f0a3b54bd3235").get_int32(),
                            cfg.get_property("b378321d3911b8fa49ea0e7fb52b043c").get_single(),
                            cfg.get_property("01e3713136ef99dd5282f837d41d0d27").get_single(),
                            cfg.get_property("b76820dc488f421fd0cf6ae6a280ecc1").get_single(), FrameSynth::edges(analysis),
                            hann(analysis), cfg.get_property("08626dfefe919f108fb4ba6002db1b23").get_boolean());
    }

    static float pitch_ring_value(const V6Frame& initial, std::span<const V6Frame> frames, int k) {
        int t = k * Hop - 2 * Hop;
        if (t < 0) return initial.f0;
        int j = t / Hop, s = t % Hop;
        const V6Frame& prev = j == 0 ? initial : frames[j - 1];
        return (frames[j].f0 - prev.f0) * (static_cast<float>(s) / Hop) + prev.f0;
    }

    void push(int which, int start, std::vector<float> samples) { ola[which][(start / Hop) & 1] = {true, start, std::move(samples)}; }

    float pop(int which, int t) const {
        float s = 0.f;
        for (const auto& slot : ola[which]) {
            if (slot.present && t - slot.start >= 0 && t - slot.start < static_cast<int>(slot.samples.size())) s += slot.samples[t - slot.start];
        }
        return s;
    }

    void frame(int k, int t, float pitch, std::span<const float> cond, std::vector<float>& source, std::vector<float>& pitches,
               std::vector<float>& ar_input) {
        ar.run(source, ar_input);
        auto head = bf8.step(ar_input, cond, speaker);
        pitches.push_back(pitch);
        float cents = k >= DecodeDelay ? pitches[k - DecodeDelay] : 0.f;
        auto out = decoder.run(head, cents, ring, t - (FeatureDelay + DecodeDelay) * Hop, noise_rng);
        std::copy(out.feedback.begin(), out.feedback.begin() + 16, source.begin());
        std::fill(source.begin() + 16, source.begin() + 96, -140.f);
        std::vector<float> seg_h(2 * Hop), seg_n(2 * Hop), seg_w(2 * Hop);
        for (int i = 0; i < 2 * Hop; i++) {
            seg_h[i] = out.shaped[256 - Hop + i] * window[i];
            seg_n[i] = out.noise[256 - Hop + i] * window[i];
            seg_w[i] = window[i] * window[i];
        }
        push(0, t, std::move(seg_h));
        push(1, t, std::move(seg_n));
        push(2, t, std::move(seg_w));
        for (int i = 0; i < Hop; i++) {
            int u = t + i;
            harm_out[(u - Hop) & Mask] = pop(0, u);
            noise_out[(u - Hop) & Mask] = pop(1, u) / std::max(std::sqrt(pop(2, u)), 0.01f);
        }
    }

    void analyse(int t, std::vector<float>& source, float alpha) {
        for (int i = 0; i < 1024; i++) {
            int idx = (i + t - 512 - 4 * Hop) & Mask;
            analysis_buffer[i] = (harm_out[idx] + noise_out[idx]) * analysis_window[i];
        }
        analysis_fft.forward(std::span<const double>(analysis_buffer.data(), 1024), spectrum);
        std::array<double, 513> db;
        for (int u = 0; u < 513; u++) db[u] = 20 * std::log10(std::max(magnitude(spectrum[u]), static_cast<double>(FloatTiny)));
        if (alpha != warped_alpha) {
            for (int u = 0; u < 513; u++) {
                double w = u * 2 * M_PI / 1024;
                warped_bins[u] = (2 * std::atan(std::sin(w) * alpha / (1 - std::cos(w) * alpha)) + w) / (2 * M_PI) * 1024;
            }
            warped_alpha = alpha;
        }
        std::array<float, 513> lin;
        for (int u = 0; u < 513; u++) {
            double pos = warped_bins[u];
            int i = std::min(static_cast<int>(std::floor(pos)), 511);
            lin[u] = static_cast<float>(std::pow(10.0, ((db[i + 1] - db[i]) * (pos - i) + db[i]) * 0.05));
        }
        band_analysis(lin, edges80, 1024, false, std::span<float>(source.data() + 16, 80));
    }

    static void fade(float* harm, float* noise, const std::vector<float>& vec) {
        if (round_even(vec[0]) != 0 || !(vec[5] > 0.2f)) return;
        float end = vec[5] - 0.1f, h6 = vec[6];
        float (*gain)(int);
        if (h6 >= 0.09f && h6 < 0.095f) gain = [](int s) { return 1.f - s * 0.5f / Hop; };
        else if (h6 >= 0.095f && h6 < 0.1f) gain = [](int s) { return 0.5f - s * 0.5f / Hop; };
        else if (h6 >= 0.1f && h6 < end) gain = [](int) { return 0.f; };
        else if (h6 >= end && h6 < 0.11f && !(h6 < 0.09f)) gain = [](int s) { return s * 0.5f / Hop + 0.5f; };
        else return;
        for (int s = 0; s < Hop; s++) {
            harm[s] *= gain(s);
            noise[s] *= gain(s);
        }
    }

    FrameSynthOutput run(std::span<const V6Frame> frames) {
        int n = static_cast<int>(frames.size());
        V6Frame initial = frames[0];
        initial.control = 0;
        std::vector<float> x9(static_cast<size_t>(Warmup + n) * 36);
        for (int w = 0; w < Warmup; w++) std::copy_n(initial.vec.begin() + 7, 32, x9.begin() + w * 36);
        std::vector<float> features(static_cast<size_t>(n + 2) * 40);
        std::copy(initial.vec.begin() + 7, initial.vec.begin() + 47, features.begin() + 2 * 40);
        for (int j = 1; j < n; j++) std::copy(frames[j - 1].vec.begin() + 7, frames[j - 1].vec.begin() + 47, features.begin() + (j + 2) * 40);
        std::vector<float> plan(n);
        const V6Frame* prev = &initial;
        for (int k = 0; k < n; k++) {
            float p = pitch_ring_value(initial, frames, k);
            size_t o = static_cast<size_t>(Warmup + k) * 36;
            std::copy_n(features.begin() + k * 40, 32, x9.begin() + o);
            x9[o + 32] = pitch_norm(p);
            x9[o + 33] = 1.f;
            x9[o + 35] = std::clamp(prev->control, -1.5f, 1.5f);
            plan[k] = p;
            prev = &frames[k];
        }
        auto y9 = controls.run(x9, Warmup + n);
        std::vector<float> source(96, -140.f), ar_input;
        for (int w = 0; w < Warmup; w++) {
            ar.run(source, ar_input);
            bf8.step(ar_input, std::span<const float>(y9.data() + static_cast<size_t>(w) * 128, 128), speaker);
        }
        FrameSynthOutput output;
        output.harmonic.resize(static_cast<size_t>(n) * Hop);
        output.noise.resize(static_cast<size_t>(n) * Hop);
        std::vector<float> pitches;
        prev = &initial;
        for (int k = 0; k < n; k++) {
            const V6Frame& cur = frames[k];
            int t0 = k * Hop;
            for (int s = 0; s < Hop; s++) {
                int t = t0 + s;
                float cents = interpolate(*prev, cur, s);
                float g = std::clamp(static_cast<float>(jitter.gauss()), -2.f, 2.f);
                float f_hz = std::pow(2.f, (g + cents) / 1200.f) * 440.f;
                ring.advance(t, f_hz);
                if (s == 0) {
                    frame(k, t, plan[k], std::span<const float>(y9.data() + static_cast<size_t>(Warmup + k) * 128, 128), source, pitches, ar_input);
                    analyse(t, source, cur.alpha);
                }
                output.harmonic[t] = harm_out[(t - Hop) & Mask];
                output.noise[t] = noise_out[(t - Hop) & Mask];
            }
            fade(&output.harmonic[t0], &output.noise[t0], prev->vec);
            prev = &cur;
        }
        return output;
    }
};

FrameSynth::FrameSynth(json::Value model, int speaker) : impl_(std::make_unique<Impl>(model, speaker)) {}

FrameSynth::~FrameSynth() = default;

FrameSynthOutput FrameSynth::run(std::span<const V6Frame> frames) { return impl_->run(frames); }

std::vector<float> FrameSynth::edges(json::Value config) {
    int bands = config.get_property(Bands).get_int32();
    float low = config.get_property(LowKey).get_single(), high = config.get_property(HighKey).get_single();
    auto warp = config.get_property(WarpKey);
    bool allpass = warp.kind() == json::Kind::Number;
    float alpha = allpass ? warp.get_single() : 0.f;
    auto forward = [&](float f) {
        if (allpass) {
            float w = f * TwoPi / Fs;
            float a = std::atan(std::sin(w) * alpha / (1.f - std::cos(w) * alpha));
            return (a + a + w) / Pi;
        }
        return f < 1000.f ? (f - 0.f) / 66.666664f : std::log(f / 1000.f) / (std::log(6.4f) / 27.f) + 15.000001f;
    };
    auto inverse = [&](float m) {
        if (allpass) {
            float w = m * Pi;
            float a = std::atan(std::sin(w) * -alpha / (1.f - std::cos(w) * -alpha));
            return (a + a + w) * Fs * 0.5f / Pi;
        }
        return m < 15.000001f ? m * 66.666664f + 0.f : std::exp(std::log(6.4f) / 27.f * (m - 15.000001f)) * 1000.f;
    };
    float lo = forward(low), hi = forward(high);
    std::vector<float> edges(bands + 2);
    for (int i = 0; i < bands + 2; i++) edges[i] = inverse(i * (hi - lo) / (bands + 1) + lo);
    return edges;
}

}
