
#include "openv6/speex_resampler.hpp"

#include <cmath>
#include <cstdint>
#include <numeric>

namespace openv6::speex_resampler {

namespace {
constexpr double Kaiser8[] = {
    0.99635258, 1.00000000, 0.99635258, 0.98548012, 0.96759014, 0.94302200, 0.91223751, 0.87580811, 0.83439927,
    0.78875245, 0.73966538, 0.68797126, 0.63451750, 0.58014482, 0.52566725, 0.47185369, 0.41941150, 0.36897272,
    0.32108304, 0.27619388, 0.23465776, 0.19672670, 0.16255380, 0.13219758, 0.10562887, 0.08273982, 0.06335451,
    0.04724088, 0.03412321, 0.02369490, 0.01563093, 0.00959968, 0.00527363, 0.00233883, 0.00050000, 0.00000000,
};
constexpr int KaiserOversample = 32, FilterLength = 48, Oversample = 8;
constexpr float Cutoff = 0.917f;

double compute_func(float x) {
    float y = x * KaiserOversample;
    int ind = static_cast<int>(std::floor(y));
    float frac = y - ind;
    double i3 = -0.1666666667 * frac + 0.1666666667 * (frac * frac * frac);
    double i2 = frac + 0.5 * (frac * frac) - 0.5 * (frac * frac * frac);
    double i0 = -0.3333333333 * frac + 0.5 * (frac * frac) - 0.1666666667 * (frac * frac * frac);
    double i1 = 1.0f - i3 - i2 - i0;
    return i0 * Kaiser8[ind] + i1 * Kaiser8[ind + 1] + i2 * Kaiser8[ind + 2] + i3 * Kaiser8[ind + 3];
}

float sinc(float cutoff, float x, int n) {
    float xx = x * cutoff;
    if (std::fabs(x) < 1e-6) return cutoff;
    if (std::fabs(x) > .5 * n) return 0;
    return static_cast<float>(cutoff * std::sin(M_PI * xx) / (M_PI * xx) * compute_func(static_cast<float>(std::fabs(2.0 * x / n))));
}

std::vector<float> build_table() {
    std::vector<float> table(Oversample * FilterLength + 8);
    for (int i = -4; i < Oversample * FilterLength + 4; i++) {
        table[i + 4] = sinc(Cutoff, i / static_cast<float>(Oversample) - FilterLength / 2, FilterLength);
    }
    return table;
}
}

std::vector<float> resample(std::span<const float> input, int in_rate, int out_rate, int output_length) {
    static const std::vector<float> table = build_table();
    int g = std::gcd(in_rate, out_rate);
    uint32_t num = static_cast<uint32_t>(in_rate / g), den = static_cast<uint32_t>(out_rate / g);
    int int_advance = static_cast<int>(num / den), frac_advance = static_cast<int>(num % den);
    std::vector<float> padded(FilterLength - 1 + input.size() + FilterLength * 2, 0.f);
    std::copy(input.begin(), input.end(), padded.begin() + (FilterLength - 1));
    std::vector<float> output(output_length, 0.f);
    int last = FilterLength / 2;
    uint32_t frac = 0;
    const float* tbl = table.data();
    for (int m = 0; m < output_length && static_cast<size_t>(last + FilterLength) <= padded.size(); m++) {
        int offset = static_cast<int>(frac * Oversample / den);
        float f = static_cast<float>(frac * Oversample % den) / den;
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int j = 0; j < FilterLength; j++) {
            float x = padded[last + j];
            int t = 4 + (j + 1) * Oversample - offset;
            a0 += x * tbl[t - 2];
            a1 += x * tbl[t - 1];
            a2 += x * tbl[t];
            a3 += x * tbl[t + 1];
        }
        float c0 = -0.16667f * f + 0.16667f * f * f * f;
        float c1 = f + 0.5f * f * f - 0.5f * f * f * f;
        float c3 = -0.33333f * f + 0.5f * f * f - 0.16667f * f * f * f;
        float c2 = static_cast<float>(1.0 - c0 - c1 - c3);
        output[m] = c0 * a0 + c1 * a1 + c2 * a2 + c3 * a3;
        last += int_advance;
        frac += static_cast<uint32_t>(frac_advance);
        if (frac >= den) {
            frac -= den;
            last++;
        }
    }
    return output;
}

}
