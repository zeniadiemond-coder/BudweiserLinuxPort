#include "openv6/model_file.hpp"

#include <bit>
#include <cstring>
#include <fstream>

#include "openv6/base64.hpp"

namespace openv6 {

static_assert(std::endian::native == std::endian::little);

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw InvalidDataError("cannot open " + path);
    std::vector<uint8_t> data(static_cast<size_t>(in.tellg()));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return data;
}

float half_to_float(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f, mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int shift = std::countl_zero(mant) - 21;
            mant = (mant << shift) & 0x3ff;
            bits = sign | static_cast<uint32_t>(127 - 15 + 1 - shift) << 23 | mant << 13;
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | mant << 13;
    } else {
        bits = sign | (exp + 127 - 15) << 23 | mant << 13;
    }
    return std::bit_cast<float>(bits);
}

std::string ModelFile::decode_span(std::span<const uint8_t> data) {
    if (data.size() < 16) throw InvalidDataError("V6 model span is shorter than the 16-byte header.");
    uint32_t seed;
    std::memcpy(&seed, data.data() + 12, 4);
    uint32_t x = 0x05491333, y = 0x159A55E5, z = 0x1F123BB5;
    const uint8_t* source = data.data() + 16;
    size_t length = data.size() - 16;
    std::string output(length, '\0');
    size_t words = length / 4;
    for (size_t i = 0; i < words; i++) {
        uint32_t old_x = x, old_y = y, old_z = z;
        uint32_t mixed = (seed << 11) ^ seed;
        x = old_x ^ mixed ^ (((old_x >> 11) ^ mixed) >> 8);
        uint32_t word;
        std::memcpy(&word, source + i * 4, 4);
        word ^= x;
        std::memcpy(output.data() + i * 4, &word, 4);
        seed = old_y;
        y = old_z;
        z = old_x;
    }
    size_t tail = length - words * 4;
    if (tail > 0) {
        uint32_t tail_value = 0;
        for (size_t i = 0; i < tail; i++) tail_value |= static_cast<uint32_t>(source[words * 4 + i]) << (8 * i);
        uint32_t mixed = (seed << 11) ^ seed;
        uint32_t key = (((x >> 11) ^ mixed) >> 8) ^ mixed ^ x ^ tail_value;
        for (size_t i = 0; i < tail; i++) output[words * 4 + i] = static_cast<char>(key >> (8 * i));
    }
    return output;
}

ModelFile::ModelFile(std::span<const uint8_t> container) {
    auto span = container.size() >= 128 ? container.first(container.size() - 128) : container;
    document_ = json::Document::parse(decode_span(span));
}

ModelFile ModelFile::load(const std::string& path) { return ModelFile(read_file(path)); }

std::vector<float> ModelFile::floats(json::Value value) {
    std::string_view text = value.get_string();
    if (text.size() < 4) throw InvalidDataError("Expected a tensor string.");
    std::string_view type = text.substr(0, 4), body = text.substr(4);
    size_t bytes = base64_decoded_size(body);
    if (type == "*F32") {
        std::vector<float> result(bytes / 4);
        std::vector<uint8_t> tail;
        if (bytes % 4 == 0) {
            base64_decode_into(body, result.data());
        } else {
            tail.resize(bytes);
            base64_decode_into(body, tail.data());
            std::memcpy(result.data(), tail.data(), result.size() * 4);
        }
        return result;
    }
    if (type == "*F16") {
        auto raw = base64_decode(body);
        std::vector<float> result(raw.size() / 2);
        for (size_t i = 0; i < result.size(); i++) {
            uint16_t h;
            std::memcpy(&h, raw.data() + i * 2, 2);
            result[i] = half_to_float(h);
        }
        return result;
    }
    throw InvalidDataError("Unsupported tensor type " + std::string(type) + ".");
}

std::vector<int32_t> ModelFile::ints(json::Value value) {
    std::string_view text = value.get_string();
    if (!text.starts_with("*S32")) throw InvalidDataError("Expected an S32 tensor.");
    auto raw = base64_decode(text.substr(4));
    std::vector<int32_t> result(raw.size() / 4);
    std::memcpy(result.data(), raw.data(), result.size() * 4);
    return result;
}

PhonemeTable ModelFile::phoneme_table(json::Value value) {
    PhonemeTable table;
    for (auto m : value.enumerate_object()) table[std::string(m.name)] = m.value.get_int32();
    return table;
}

}
