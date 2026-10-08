#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "openv6/json.hpp"

namespace openv6 {

namespace keys {
inline constexpr std::string_view Params = "3c77d931295c40d4d75d971398fee8a1";
inline constexpr std::string_view Dilations = "b028f94998d330bc643746f7a453feed";
inline constexpr std::string_view Pre = "932ef94a406f2ac2793fd98a2fe2e008";
inline constexpr std::string_view Layers = "60934ffa2a466aa4dea617defa78f36b";
inline constexpr std::string_view Conv = "3d8a9ac90f7ac150415ce60a7b3aea38";
inline constexpr std::string_view Skip = "88f599b6848cb38e980ea3467fbec4b6";
inline constexpr std::string_view Post = "ed68c4c9013230e898fe12529dc1a63e";
inline constexpr std::string_view PostWeight = "96e855e8076802b01546388bf18cd3c8";
inline constexpr std::string_view Final = "a3e0d41f3cb9977373f3e0fd64b59617";
inline constexpr std::string_view Embedding = "084246b66df966440092e15db1a64d20";
inline constexpr std::string_view Condition = "72e89ef9aa4547f206cecfa2269de01b";
inline constexpr std::string_view LayerCondition = "477aa7b96a1e0944307793fdd4f4a96b";
inline constexpr std::string_view Weight = "30c16498ff30815c43d58f610a91f938";
inline constexpr std::string_view Bias = "c169d8ef47cc23f097ff312d08a6f829";
inline constexpr std::string_view Inputs = "3fecf2d78547692b4157889e04a498f0";
inline constexpr std::string_view Hidden = "247e226db757c8012220030394d22b2e";
inline constexpr std::string_view Skips = "731aacbf65e8845bb02f662f0b402053";
inline constexpr std::string_view Outputs = "2aff5930df33d03ad1f33c3091886f13";
inline constexpr std::string_view PostActivation = "69ebce9ff768f54cd4d95647e4d878cf";
inline constexpr std::string_view TanhActivation = "8a8d5b352266bf85b49ce7ad682a1149";
inline constexpr std::string_view Encoder = "c688ac0ef614fc2004874245946c9580";
inline constexpr std::string_view Controls = "ab464905e3eb532d4ba956c7464f5773";
inline constexpr std::string_view Synthesis = "6b46ed7b862f0e89a13164f3f8658380";
}

namespace native_math {
inline float exp(float x) {
    float p = x * 1.442695f;
    float offset = p < 0 ? 1.0f : 0.0f;
    float q = std::fmax(p, -126.0f);
    float z = q - std::truncf(q) + offset;
    float v = (27.728024f / (4.8425255f - z) + q + 121.274055f - z * 1.4901291f) * 8388608.0f;
    return std::bit_cast<float>(static_cast<uint32_t>(static_cast<int64_t>(v)));
}

inline float tanh(float x) { return 2.0f / (exp(x * -2.0f) + 1.0f) - 1.0f; }
inline float sigmoid(float x) { return 1.0f / (exp(-x) + 1.0f); }
inline float leaky_clamp(float x) { return std::fmin(x < 0 ? x * 0.01f : x, 4.0f); }

inline float dot(std::span<const float> a, std::span<const float> b) {
    int n = static_cast<int>(a.size()), i = 0;
    float acc[8] = {};
    for (; i <= n - 8; i += 8)
        for (int j = 0; j < 8; j++) acc[j] += a[i + j] * b[i + j];
    float low = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    float high = (acc[4] + acc[5]) + (acc[6] + acc[7]);
    float sum = low + high;
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

inline void mat_vec(std::span<const float> w, int rows, int cols, std::span<const float> x, std::span<float> y) {
    for (int r = 0; r < rows; r++) y[r] += dot(w.subspan(static_cast<size_t>(r) * cols, cols), x.first(cols));
}
}

class PackedMatrix {
public:
    PackedMatrix(std::span<const float> w, int rows, int cols);
    void mul_add(std::span<const float> x, std::span<float> y) const;

private:
    int rows_ = 0, cols_ = 0;
    std::vector<float> packed_;
};

class GatedStack {
public:
    GatedStack(json::Value section, int inputs, int hidden, int outputs);

    static std::vector<float> kernel_major(std::span<const float> w, int hidden);

    int latency() const;

    std::vector<float> run(std::span<const float> x, int frames) const;

    std::vector<float> run_aligned(std::span<const float> x, int frames) const;

private:
    int inputs_, hidden_, skips_ = 0, outputs_;
    std::vector<int> dilations_;
    std::vector<float> pre_b_, post_b_, final_b_;
    std::vector<std::vector<float>> conv_b_;
    std::unique_ptr<PackedMatrix> pre_, post_, final_;
    std::vector<PackedMatrix> conv_, skip_;
};

class CausalStack {
public:
    explicit CausalStack(json::Value section);

    int inputs, hidden, skips, outputs, condition_width;

    int receptive_field() const;

    std::span<const float> step(std::span<const float> feedback, std::span<const float> conditioning, int speaker);

private:
    float activate(float v) const;

    std::vector<int> dilations_;
    bool tanh_post_ = false;
    std::vector<float> embedding_;
    std::unique_ptr<PackedMatrix> pre_, condition_, final_;
    std::vector<PackedMatrix> layers_, skips_, posts_;
    std::vector<std::vector<float>> history_;
    int span_ = 0;
    std::vector<float> input_, x_, a_, skip_, y_, z_, result_;
    int t_ = 0;
};

}
