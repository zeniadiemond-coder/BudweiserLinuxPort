#include "openv6/networks.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>

#include "openv6/model_file.hpp"
#include "openv6/parallel.hpp"

namespace openv6 {

namespace {

constexpr int Block = 32;

template <bool Fma>
[[gnu::always_inline]] inline void mul_add_kernel(const float* packed, int rows, int cols, const float* x, float* y) {
    for (int start = 0; start < rows; start += Block) {
        float acc[Block] = {};
        const float* p = packed + static_cast<size_t>(start / Block) * cols * Block;
        for (int c = 0; c < cols; c++, p += Block) {
            float xc = x[c];
            for (int j = 0; j < Block; j++) {
                if constexpr (Fma) acc[j] = __builtin_fmaf(p[j], xc, acc[j]);
                else acc[j] += p[j] * xc;
            }
        }
        int count = std::min(Block, rows - start);
        for (int j = 0; j < count; j++) y[start + j] += acc[j];
    }
}

void mul_add_generic(const float* packed, int rows, int cols, const float* x, float* y) {
    mul_add_kernel<false>(packed, rows, cols, x, y);
}

#if defined(__x86_64__) || defined(__i386__)
[[gnu::target("avx2,fma")]] void mul_add_fma(const float* packed, int rows, int cols, const float* x, float* y) {
    mul_add_kernel<true>(packed, rows, cols, x, y);
}

bool cpu_has_fma() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

const bool has_fma = cpu_has_fma();
#endif

std::vector<int> int_values(json::Value value) {
    auto v = ModelFile::ints(value);
    return {v.begin(), v.end()};
}

}

PackedMatrix::PackedMatrix(std::span<const float> w, int rows, int cols)
    : rows_(rows), cols_(cols), packed_(static_cast<size_t>((rows + Block - 1) / Block) * cols * Block) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) packed_[(static_cast<size_t>(r / Block) * cols + c) * Block + r % Block] = w[static_cast<size_t>(r) * cols + c];
}

void PackedMatrix::mul_add(std::span<const float> x, std::span<float> y) const {
#if defined(__x86_64__) || defined(__i386__)
    if (has_fma) return mul_add_fma(packed_.data(), rows_, cols_, x.data(), y.data());
#endif
    mul_add_generic(packed_.data(), rows_, cols_, x.data(), y.data());
}

GatedStack::GatedStack(json::Value section, int inputs, int hidden, int outputs)
    : inputs_(inputs), hidden_(hidden), outputs_(outputs) {
    dilations_ = int_values(section.get_property(keys::Params).get_property(keys::Dilations));
    auto pre = section.get_property(keys::Pre);
    pre_ = std::make_unique<PackedMatrix>(ModelFile::floats(pre.get_property(keys::Weight)), hidden, inputs);
    pre_b_ = ModelFile::floats(pre.get_property(keys::Bias));
    std::vector<std::vector<float>> skip_w;
    for (auto layer : section.get_property(keys::Layers).enumerate_array()) {
        auto conv = layer.get_property(keys::Conv);
        conv_.emplace_back(kernel_major(ModelFile::floats(conv.get_property(keys::Weight)), hidden), 2 * hidden, 3 * hidden);
        conv_b_.push_back(ModelFile::floats(conv.get_property(keys::Bias)));
        skip_w.push_back(ModelFile::floats(layer.get_property(keys::Skip).get_property(keys::Weight)));
    }
    skips_ = static_cast<int>(skip_w[0].size()) / hidden;
    for (auto& w : skip_w) skip_.emplace_back(w, skips_, hidden);
    auto post = section.get_property(keys::Post)[0][0].get_property(keys::PostWeight);
    post_ = std::make_unique<PackedMatrix>(ModelFile::floats(post.get_property(keys::Weight)), skips_, skips_);
    post_b_ = ModelFile::floats(post.get_property(keys::Bias));
    auto final = section.get_property(keys::Final)[0];
    final_ = std::make_unique<PackedMatrix>(ModelFile::floats(final.get_property(keys::Weight)), outputs, skips_);
    final_b_ = ModelFile::floats(final.get_property(keys::Bias));
}

std::vector<float> GatedStack::kernel_major(std::span<const float> w, int hidden) {
    std::vector<float> result(w.size());
    int outs = static_cast<int>(w.size()) / (hidden * 3);
    for (int o = 0; o < outs; o++)
        for (int k = 0; k < 3; k++)
            for (int c = 0; c < hidden; c++) result[(o * 3 + k) * hidden + c] = w[(o * hidden + c) * 3 + k];
    return result;
}

int GatedStack::latency() const { return std::accumulate(dilations_.begin(), dilations_.end(), 0); }

std::vector<float> GatedStack::run(std::span<const float> x, int frames) const {
    const int hidden = hidden_, skips = skips_;
    std::vector<float> h(static_cast<size_t>(frames) * hidden);
    parallel_for(frames, [&](int begin, int end) {
        for (int t = begin; t < end; t++) {
            std::span<float> row(h.data() + static_cast<size_t>(t) * hidden, hidden);
            std::copy(pre_b_.begin(), pre_b_.end(), row.begin());
            pre_->mul_add(x.subspan(static_cast<size_t>(t) * inputs_, inputs_), row);
        }
    });
    std::vector<float> skip_sum(static_cast<size_t>(frames) * skips);
    std::vector<float> next(static_cast<size_t>(frames) * hidden);
    for (size_t i = 0; i < dilations_.size(); i++) {
        int d = dilations_[i];
        int delay = std::accumulate(dilations_.begin() + i + 1, dilations_.end(), 0);
        const auto& layer = conv_[i];
        const auto& b = conv_b_[i];
        parallel_for(frames, [&](int begin, int end) {
            std::vector<float> taps(3 * hidden), a(2 * hidden);
            for (int t = begin; t < end; t++) {
                for (int k = 0; k < 3; k++) {
                    int src = t - (2 - k) * d;
                    float* tap = taps.data() + k * hidden;
                    if (src >= 0) std::memcpy(tap, h.data() + static_cast<size_t>(src) * hidden, hidden * sizeof(float));
                    else std::fill(tap, tap + hidden, 0.0f);
                }
                std::copy(b.begin(), b.end(), a.begin());
                layer.mul_add(taps, a);
                float* out = next.data() + static_cast<size_t>(t) * hidden;
                for (int c = 0; c < hidden; c++) out[c] = native_math::tanh(a[c]) * native_math::sigmoid(a[hidden + c]);
            }
        });
        std::swap(h, next);
        const auto& layer_skip = skip_[i];
        parallel_for(std::max(frames - delay, 0), [&](int begin, int end) {
            for (int t = begin; t < end; t++)
                layer_skip.mul_add(std::span<const float>(h.data() + static_cast<size_t>(t) * hidden, hidden),
                                   std::span<float>(skip_sum.data() + static_cast<size_t>(t + delay) * skips, skips));
        });
    }
    std::vector<float> result(static_cast<size_t>(frames) * outputs_);
    parallel_for(frames, [&](int begin, int end) {
        std::vector<float> y(skips), z(skips);
        for (int t = begin; t < end; t++) {
            for (int j = 0; j < skips; j++) y[j] = native_math::tanh(skip_sum[static_cast<size_t>(t) * skips + j]);
            std::copy(post_b_.begin(), post_b_.end(), z.begin());
            post_->mul_add(y, z);
            for (int j = 0; j < skips; j++) z[j] = native_math::tanh(z[j]);
            std::span<float> r(result.data() + static_cast<size_t>(t) * outputs_, outputs_);
            std::copy(final_b_.begin(), final_b_.end(), r.begin());
            final_->mul_add(z, r);
            for (float& v : r) v = native_math::tanh(v);
        }
    });
    return result;
}

std::vector<float> GatedStack::run_aligned(std::span<const float> x, int frames) const {
    int lat = latency(), warm = 2 * lat + 1, total = warm + frames + lat;
    std::vector<float> padded(static_cast<size_t>(total) * inputs_);
    for (int i = 0; i < total; i++) {
        int src = i < warm ? 0 : std::min(i - warm, frames - 1);
        std::memcpy(padded.data() + static_cast<size_t>(i) * inputs_, x.data() + static_cast<size_t>(src) * inputs_, inputs_ * sizeof(float));
    }
    auto y = run(padded, total);
    return {y.begin() + static_cast<size_t>(warm + lat) * outputs_, y.begin() + static_cast<size_t>(warm + lat + frames) * outputs_};
}

CausalStack::CausalStack(json::Value section) {
    auto p = section.get_property(keys::Params);
    inputs = p.get_property(keys::Inputs).get_int32();
    hidden = p.get_property(keys::Hidden).get_int32();
    skips = p.get_property(keys::Skips).get_int32();
    outputs = p.get_property(keys::Outputs).get_int32();
    tanh_post_ = p.get_property(keys::PostActivation).get_string() == keys::TanhActivation;
    dilations_ = int_values(p.get_property(keys::Dilations));
    pre_ = std::make_unique<PackedMatrix>(ModelFile::floats(section.get_property(keys::Pre).get_property(keys::Weight)), hidden, inputs);
    embedding_ = ModelFile::floats(section.get_property(keys::Embedding));
    auto cond_w = ModelFile::floats(section.get_property(keys::Condition).get_property(keys::Weight));
    condition_width = static_cast<int>(cond_w.size()) / skips;
    condition_ = std::make_unique<PackedMatrix>(cond_w, skips, condition_width);
    int width = 3 * hidden + condition_width;
    for (auto layer : section.get_property(keys::Layers).enumerate_array()) {
        auto conv = GatedStack::kernel_major(ModelFile::floats(layer.get_property(keys::Conv).get_property(keys::Weight)), hidden);
        auto cond = ModelFile::floats(layer.get_property(keys::LayerCondition).get_property(keys::Weight));
        std::vector<float> w(static_cast<size_t>(2 * hidden) * width);
        for (int o = 0; o < 2 * hidden; o++) {
            std::copy_n(conv.begin() + static_cast<size_t>(o) * 3 * hidden, 3 * hidden, w.begin() + static_cast<size_t>(o) * width);
            std::copy_n(cond.begin() + static_cast<size_t>(o) * condition_width, condition_width,
                        w.begin() + static_cast<size_t>(o) * width + 3 * hidden);
        }
        layers_.emplace_back(w, 2 * hidden, width);
        skips_.emplace_back(ModelFile::floats(layer.get_property(keys::Skip).get_property(keys::Weight)), skips, hidden);
    }
    for (auto layer : section.get_property(keys::Post)[0].enumerate_array())
        posts_.emplace_back(ModelFile::floats(layer.get_property(keys::PostWeight).get_property(keys::Weight)), skips, skips);
    final_ = std::make_unique<PackedMatrix>(ModelFile::floats(section.get_property(keys::Final)[0].get_property(keys::Weight)), outputs, skips);
    span_ = 2 * *std::max_element(dilations_.begin(), dilations_.end()) + 1;
    history_.assign(dilations_.size(), std::vector<float>(static_cast<size_t>(span_) * hidden));
    input_.resize(width);
    x_.resize(hidden);
    a_.resize(2 * hidden);
    skip_.resize(skips);
    y_.resize(skips);
    z_.resize(skips);
    result_.resize(outputs);
}

int CausalStack::receptive_field() const { return 2 * std::accumulate(dilations_.begin(), dilations_.end(), 0) + 1; }

float CausalStack::activate(float v) const { return tanh_post_ ? native_math::tanh(v) : native_math::leaky_clamp(v); }

std::span<const float> CausalStack::step(std::span<const float> feedback, std::span<const float> conditioning, int speaker) {
    std::span<float> c(input_.data() + 3 * hidden, condition_width);
    int embed = condition_width - static_cast<int>(conditioning.size());
    size_t rows = embedding_.size() / embed;
    if (speaker < 0 || static_cast<size_t>(speaker) >= rows)
        throw InvalidDataError("Speaker row " + std::to_string(speaker) + " is outside the model's " + std::to_string(rows) + " embedding rows.");
    std::copy(conditioning.begin(), conditioning.end(), c.begin());
    std::copy_n(embedding_.begin() + static_cast<size_t>(speaker) * embed, embed, c.begin() + conditioning.size());
    std::fill(x_.begin(), x_.end(), 0.0f);
    pre_->mul_add(feedback, x_);
    std::fill(skip_.begin(), skip_.end(), 0.0f);
    condition_->mul_add(c, skip_);
    for (size_t i = 0; i < dilations_.size(); i++) {
        int d = dilations_[i];
        auto& hist = history_[i];
        std::copy(x_.begin(), x_.end(), hist.begin() + static_cast<size_t>(t_ % span_) * hidden);
        size_t o1 = static_cast<size_t>((((t_ - d) % span_) + span_) % span_) * hidden;
        size_t o2 = static_cast<size_t>((((t_ - 2 * d) % span_) + span_) % span_) * hidden;
        std::copy_n(hist.begin() + o2, hidden, input_.begin());
        std::copy_n(hist.begin() + o1, hidden, input_.begin() + hidden);
        std::copy(x_.begin(), x_.end(), input_.begin() + 2 * hidden);
        std::fill(a_.begin(), a_.end(), 0.0f);
        layers_[i].mul_add(input_, a_);
        for (int ch = 0; ch < hidden; ch++) x_[ch] = native_math::tanh(a_[ch]) * native_math::sigmoid(a_[hidden + ch]);
        skips_[i].mul_add(x_, skip_);
    }
    t_++;
    for (int j = 0; j < skips; j++) y_[j] = activate(skip_[j]);
    for (const auto& post : posts_) {
        std::fill(z_.begin(), z_.end(), 0.0f);
        post.mul_add(y_, z_);
        for (int j = 0; j < skips; j++) y_[j] = activate(z_[j]);
    }
    std::fill(result_.begin(), result_.end(), 0.0f);
    final_->mul_add(y_, result_);
    return result_;
}

}
