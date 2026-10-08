#include "openv6/vtmg_model.hpp"

#include <cmath>
#include <cstring>
#include <vector>

#include "openv6/base64.hpp"
#include "openv6/json.hpp"
#include "openv6/md5.hpp"
#include "openv6/model_file.hpp"

namespace openv6 {

namespace {
constexpr std::string_view WeightSection = "6957915c8dfb4d554a8df92ee40e523d";
constexpr std::string_view ConsonantSection = "16560fbd272d1f5059d620dc7e15a5aa";
constexpr uint8_t NamingSuffix[] = {0x69, 0x23, 0x33, 0x65, 0x2e, 0x63, 0x5a, 0x54, 0x25, 0x35, 0x6b, 0x39,
                                    0x6a, 0x32, 0x64, 0x2c, 0x32, 0x5a, 0x72, 0x53, 0x2f, 0x2b, 0x2f, 0x32};

struct Parameter {
    std::string name;
    std::vector<int> shape;
};

std::vector<Parameter> build_parameters() {
    std::vector<Parameter> p = {
        {"pre1.weight", {32, 82}}, {"pre1.bias", {32}}, {"pre2.weight", {32, 32}}, {"pre2.bias", {32}},
    };
    for (int stage = 1; stage <= 3; stage++) {
        std::string s = std::to_string(stage);
        p.push_back({"conv" + s + ".weight", {64, 32, 3}});
        p.push_back({"conv" + s + ".bias", {64}});
        p.push_back({"cond" + s + ".weight", {64, 0}});
        p.push_back({"cond" + s + ".bias", {64}});
    }
    p.push_back({"linear.weight", {3, 32}});
    p.push_back({"linear.bias", {3}});
    return p;
}

std::string name_key(const std::string& name) {
    std::vector<uint8_t> bytes(name.begin(), name.end());
    bytes.insert(bytes.end(), std::begin(NamingSuffix), std::end(NamingSuffix));
    return md5_hex(bytes);
}

float sigmoid(float v) { return 1.f / (1.f + std::exp(-v)); }
}

VtmgModel::VtmgModel(std::span<const uint8_t> container) {
    auto span = container.size() >= 128 ? container.first(container.size() - 128) : container;
    auto document = json::Document::parse(ModelFile::decode_span(span));
    auto root = document->root();
    auto section = root.get_property(WeightSection);
    for (auto m : root.get_property(ConsonantSection).enumerate_object()) consonant_weights_[std::string(m.name)] = m.value.get_single();
    for (const auto& [name, shape] : build_parameters()) {
        json::Value value;
        if (!section.try_get_property(name_key(name), value) || value.kind() != json::Kind::String ||
            !value.get_string().starts_with("*F32")) {
            throw InvalidDataError("VTMG parameter " + name + " is missing.");
        }
        std::string_view body = value.get_string().substr(4);
        size_t bytes = base64_decoded_size(body);
        size_t count = 1;
        for (int d : shape) count *= d;
        if (name == "cond1.weight") conditions_ = static_cast<int>(bytes / 4 / 64);
        if (name.starts_with("cond") && name.ends_with(".weight")) count = 64 * conditions_;
        if (bytes != count * 4) throw InvalidDataError("VTMG parameter " + name + " has an unexpected size.");
        std::vector<float> data(count);
        base64_decode_into(body, data.data());
        weights_[name] = std::move(data);
    }
}

VtmgModel VtmgModel::load(const std::string& path) { return VtmgModel(read_file(path)); }

std::vector<float> VtmgModel::forward(std::span<const float> features, int frames, int condition) const {
    if (features.size() != static_cast<size_t>(frames) * 82) throw InvalidDataError("VTMG features must be 82 wide per frame.");
    if (frames < 7) throw InvalidDataError("VTMG requires at least seven frames.");
    if (condition < 0 || condition >= conditions_) throw InvalidDataError("VTMG condition is out of range.");
    const auto& pre1W = weights_.at("pre1.weight");
    const auto& pre1B = weights_.at("pre1.bias");
    const auto& pre2W = weights_.at("pre2.weight");
    const auto& pre2B = weights_.at("pre2.bias");
    std::vector<float> x(static_cast<size_t>(frames) * 32);
    for (int f = 0; f < frames; f++) {
        for (int j = 0; j < 32; j++) {
            float sum = pre1B[j];
            for (int i = 0; i < 82; i++) sum += features[f * 82 + i] * pre1W[j * 82 + i];
            x[f * 32 + j] = std::tanh(sum);
        }
    }
    std::vector<float> projected(static_cast<size_t>(frames) * 32);
    for (int f = 0; f < frames; f++) {
        for (int j = 0; j < 32; j++) {
            float sum = pre2B[j];
            for (int i = 0; i < 32; i++) sum += x[f * 32 + i] * pre2W[j * 32 + i];
            projected[f * 32 + j] = std::tanh(sum);
        }
    }
    x = std::move(projected);
    int length = frames;
    for (int stage = 1; stage <= 3; stage++) {
        std::string s = std::to_string(stage);
        const auto& convW = weights_.at("conv" + s + ".weight");
        const auto& convB = weights_.at("conv" + s + ".bias");
        const auto& condW = weights_.at("cond" + s + ".weight");
        const auto& condB = weights_.at("cond" + s + ".bias");
        int output_length = length - 2;
        std::vector<float> gated(static_cast<size_t>(output_length) * 64);
        for (int f = 0; f < output_length; f++) {
            for (int channel = 0; channel < 64; channel++) {
                float sum = convB[channel] + condW[channel * conditions_ + condition] + condB[channel];
                for (int c = 0; c < 32; c++)
                    for (int k = 0; k < 3; k++) sum += x[(f + k) * 32 + c] * convW[(channel * 32 + c) * 3 + k];
                gated[f * 64 + channel] = sum;
            }
        }
        std::vector<float> next(static_cast<size_t>(output_length) * 32);
        for (int f = 0; f < output_length; f++)
            for (int j = 0; j < 32; j++) next[f * 32 + j] = sigmoid(gated[f * 64 + j]) * std::tanh(gated[f * 64 + 32 + j]);
        x = std::move(next);
        length = output_length;
    }
    const auto& linearW = weights_.at("linear.weight");
    const auto& linearB = weights_.at("linear.bias");
    std::vector<float> result(static_cast<size_t>(length) * 3);
    for (int f = 0; f < length; f++) {
        for (int channel = 0; channel < 3; channel++) {
            float sum = linearB[channel];
            for (int j = 0; j < 32; j++) sum += x[f * 32 + j] * linearW[channel * 32 + j];
            result[f * 3 + channel] = sum;
        }
    }
    return result;
}

}
