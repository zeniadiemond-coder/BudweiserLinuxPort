#pragma once

#include <complex>
#include <span>
#include <vector>

namespace openv6 {

class RealFourierTransform {
public:
    explicit RealFourierTransform(int length);

    int length() const { return length_; }

    void forward(std::span<const double> samples, std::span<std::complex<double>> spectrum);
    void inverse(std::span<const std::complex<double>> spectrum, std::span<double> samples);

private:
    void transform(bool inverse);

    int length_;
    int half_;
    std::vector<std::complex<double>> twiddles_;
    std::vector<std::complex<double>> split_twiddles_;
    std::vector<int> bit_reverse_;
    std::vector<std::complex<double>> work_;
};

}
