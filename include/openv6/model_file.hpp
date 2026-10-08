#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "openv6/json.hpp"
#include "openv6/util.hpp"

namespace openv6 {

using PhonemeTable = StringMap<int>;

float half_to_float(uint16_t half);

class ModelFile {
public:
    explicit ModelFile(std::span<const uint8_t> container);
    static ModelFile load(const std::string& path);

    json::Value root() const { return document_->root(); }

    static std::string decode_span(std::span<const uint8_t> data);

    static std::vector<float> floats(json::Value value);
    static std::vector<int32_t> ints(json::Value value);
    static PhonemeTable phoneme_table(json::Value value);

private:
    std::unique_ptr<json::Document> document_;
};

std::vector<uint8_t> read_file(const std::string& path);

}
