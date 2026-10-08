#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "check.hpp"
#include "openv6/networks.hpp"

using namespace openv6;

namespace {

std::vector<float> random_floats(std::mt19937& rng, size_t n) {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (float& f : v) f = d(rng);
    return v;
}

void test_packed_matrix() {
    std::mt19937 rng(1);
    for (auto [rows, cols] : {std::pair{1, 1}, {7, 5}, {32, 64}, {33, 17}, {96, 200}, {130, 3}}) {
        auto w = random_floats(rng, static_cast<size_t>(rows) * cols);
        auto x = random_floats(rng, cols);
        auto y = random_floats(rng, rows);
        auto expected = y;
        for (int r = 0; r < rows; r++) {
            float acc = 0;
            for (int c = 0; c < cols; c++) acc = std::fmaf(w[r * cols + c], x[c], acc);
            expected[r] += acc;
        }
        PackedMatrix(w, rows, cols).mul_add(x, y);
        for (int r = 0; r < rows; r++) CHECK_MSG(y[r] == expected[r], std::to_string(rows) + "x" + std::to_string(cols));
    }
}

void test_native_math() {
    CHECK(std::fabs(native_math::exp(0.0f) - 1.0f) < 1e-3f);
    for (float x = -10; x <= 10; x += 0.37f) {
        CHECK_MSG(std::fabs(native_math::exp(x) / std::exp(x) - 1.0f) < 1e-3f, std::to_string(x));
        CHECK(std::fabs(native_math::tanh(x) - std::tanh(x)) < 2e-3f);
        CHECK(std::fabs(native_math::sigmoid(x) - 1.0f / (1.0f + std::exp(-x))) < 1e-3f);
    }
    CHECK(native_math::leaky_clamp(-2.0f) == -0.02f);
    CHECK(native_math::leaky_clamp(9.0f) == 4.0f);
    CHECK(native_math::leaky_clamp(1.5f) == 1.5f);
}

void test_dot() {
    std::mt19937 rng(2);
    for (int n : {0, 1, 7, 8, 9, 64, 131}) {
        auto a = random_floats(rng, n), b = random_floats(rng, n);
        double exact = 0;
        for (int i = 0; i < n; i++) exact += double(a[i]) * b[i];
        CHECK(std::fabs(native_math::dot(a, b) - exact) < 1e-4);
    }
    float a[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9}, ones[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    CHECK(native_math::dot(a, ones) == 45.0f);
    std::vector<float> w = {1, 2, 3, 4, 5, 6}, x = {1, 1, 1}, y = {10, 20};
    native_math::mat_vec(w, 2, 3, x, y);
    CHECK(y[0] == 16.0f && y[1] == 35.0f);
}

void test_kernel_major() {
    std::vector<float> w(2 * 2 * 3);
    for (size_t i = 0; i < w.size(); i++) w[i] = static_cast<float>(i);
    auto r = GatedStack::kernel_major(w, 2);
    CHECK(r[0] == w[0] && r[1] == w[3] && r[2] == w[1] && r[3] == w[4] && r[4] == w[2] && r[5] == w[5]);
}

}

int main() {
    test_packed_matrix();
    test_native_math();
    test_dot();
    test_kernel_major();
    if (g_failures) return 1;
    std::puts("test_networks ok");
    return 0;
}
