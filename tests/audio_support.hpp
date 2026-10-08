#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <numbers>
#include <span>
#include <vector>

#include "openv6/fft.hpp"
#include "timing_support.hpp"

namespace audio_support {

using namespace openv6;

inline std::vector<float> read_pcm16(const std::filesystem::path& path) {
    std::string wav = timing_support::read_text(path);
    std::vector<float> pcm((wav.size() - 44) / 2);
    for (size_t i = 0; i < pcm.size(); i++) {
        int16_t v;
        std::memcpy(&v, wav.data() + 44 + i * 2, 2);
        pcm[i] = v / 32768.f;
    }
    return pcm;
}

inline
double band_distance(std::span<const float> a, std::span<const float> b) {
    RealFourierTransform fft(1024);
    std::vector<double> window(1024);
    for (int i = 0; i < 1024; i++) window[i] = 0.5 - 0.5 * std::cos(2 * std::numbers::pi * i / 1024);
    std::vector<int> edges;
    for (int i = 0; i < 21; i++) {
        int edge = static_cast<int>(2 * std::pow(465 / 2.0, i / 20.0));
        if (edges.empty() || edges.back() != edge) edges.push_back(edge);
    }
    std::vector<double> buffer_a(1024), buffer_b(1024);
    std::vector<std::complex<double>> sa(513), sb(513);
    std::vector<double> diffs;
    for (size_t start = 0; start + 1024 <= std::min(a.size(), b.size()); start += 256) {
        double energy = 0;
        for (int i = 0; i < 1024; i++) {
            buffer_a[i] = a[start + i] * window[i];
            buffer_b[i] = b[start + i] * window[i];
            energy += buffer_b[i] * buffer_b[i];
        }
        if (10 * std::log10(energy / 1024 + 1e-30) < -60) continue;
        fft.forward(buffer_a, sa);
        fft.forward(buffer_b, sb);
        for (size_t band = 0; band + 1 < edges.size(); band++) {
            double pa = 0, pb = 0;
            for (int k = edges[band]; k < edges[band + 1]; k++) {
                pa += std::norm(sa[k]);
                pb += std::norm(sb[k]);
            }
            diffs.push_back(std::fabs(10 * std::log10(pa + 1e-12) - 10 * std::log10(pb + 1e-12)));
        }
    }
    std::sort(diffs.begin(), diffs.end());
    return diffs[diffs.size() / 2];
}

}
