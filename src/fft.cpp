#include "openv6/fft.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace openv6 {

RealFourierTransform::RealFourierTransform(int length) : length_(length), half_(length / 2) {
    if (length < 4 || (length & (length - 1)) != 0) throw std::invalid_argument("FFT length must be a power of two >= 4");
    twiddles_.resize(half_ / 2 > 0 ? half_ / 2 : 1);
    for (int k = 0; k < half_ / 2; k++) twiddles_[k] = std::polar(1.0, -2.0 * M_PI * k / half_);
    split_twiddles_.resize(half_ + 1);
    for (int k = 0; k <= half_; k++) split_twiddles_[k] = std::polar(1.0, -2.0 * M_PI * k / length_);
    bit_reverse_.resize(half_);
    int bits = 0;
    while ((1 << bits) < half_) bits++;
    for (int i = 0; i < half_; i++) {
        int r = 0;
        for (int b = 0; b < bits; b++)
            if (i & (1 << b)) r |= 1 << (bits - 1 - b);
        bit_reverse_[i] = r;
    }
    work_.resize(half_);
}

void RealFourierTransform::transform(bool inverse) {
    for (int i = 0; i < half_; i++) {
        int j = bit_reverse_[i];
        if (i < j) std::swap(work_[i], work_[j]);
    }
    for (int size = 2; size <= half_; size <<= 1) {
        int step = half_ / size;
        for (int start = 0; start < half_; start += size) {
            for (int k = 0; k < size / 2; k++) {
                std::complex<double> w = twiddles_[k * step];
                if (inverse) w = std::conj(w);
                std::complex<double> a = work_[start + k], b = work_[start + k + size / 2] * w;
                work_[start + k] = a + b;
                work_[start + k + size / 2] = a - b;
            }
        }
    }
}

void RealFourierTransform::forward(std::span<const double> samples, std::span<std::complex<double>> spectrum) {
    for (int i = 0; i < half_; i++) work_[i] = {samples[2 * i], samples[2 * i + 1]};
    transform(false);
    for (int k = 0; k <= half_; k++) {
        std::complex<double> a = work_[k % half_], b = std::conj(work_[(half_ - k) % half_]);
        std::complex<double> even = (a + b) * 0.5, odd = (a - b) * std::complex<double>(0, -0.5);
        spectrum[k] = even + split_twiddles_[k] * odd;
    }
    spectrum[0] = {spectrum[0].real(), 0};
    spectrum[half_] = {spectrum[half_].real(), 0};
}

void RealFourierTransform::inverse(std::span<const std::complex<double>> spectrum, std::span<double> samples) {
    for (int k = 0; k < half_; k++) {
        std::complex<double> xk = spectrum[k], xm = std::conj(spectrum[half_ - k]);
        if (k == 0) xk = xk.real();
        if (k == 0) xm = std::conj(std::complex<double>(spectrum[half_].real()));
        std::complex<double> even = (xk + xm) * 0.5;
        std::complex<double> odd = (xk - xm) * 0.5 * std::conj(split_twiddles_[k]);
        work_[k] = even + std::complex<double>(0, 1) * odd;
    }
    transform(true);
    double scale = 1.0 / half_;
    for (int i = 0; i < half_; i++) {
        samples[2 * i] = work_[i].real() * scale;
        samples[2 * i + 1] = work_[i].imag() * scale;
    }
}

}
