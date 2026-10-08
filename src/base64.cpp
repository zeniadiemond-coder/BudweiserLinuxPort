#include "openv6/base64.hpp"

#include <array>

#include "openv6/util.hpp"

namespace openv6 {

namespace {
constexpr std::array<int8_t, 256> make_table() {
    std::array<int8_t, 256> t{};
    for (auto& v : t) v = -1;
    for (int i = 0; i < 26; i++) {
        t['A' + i] = static_cast<int8_t>(i);
        t['a' + i] = static_cast<int8_t>(26 + i);
    }
    for (int i = 0; i < 10; i++) t['0' + i] = static_cast<int8_t>(52 + i);
    t['+'] = 62;
    t['/'] = 63;
    return t;
}
constexpr auto table = make_table();

size_t padding(std::string_view text) {
    if (text.size() % 4 != 0) throw InvalidDataError("base64 length is not a multiple of 4");
    size_t pad = 0;
    if (!text.empty() && text.back() == '=') pad++;
    if (text.size() > 1 && text[text.size() - 2] == '=') pad++;
    return pad;
}
}

size_t base64_decoded_size(std::string_view text) { return text.size() / 4 * 3 - padding(text); }

void base64_decode_into(std::string_view text, void* output) {
    size_t pad = padding(text);
    auto* out = static_cast<uint8_t*>(output);
    size_t groups = text.size() / 4;
    const auto* in = reinterpret_cast<const uint8_t*>(text.data());
    for (size_t g = 0; g < groups; g++, in += 4) {
        bool last = g + 1 == groups;
        int a = table[in[0]], b = table[in[1]];
        int c = (last && pad == 2) ? 0 : table[in[2]];
        int d = (last && pad >= 1) ? 0 : table[in[3]];
        if ((a | b | c | d) < 0) throw InvalidDataError("invalid base64 character");
        uint32_t v = (a << 18) | (b << 12) | (c << 6) | d;
        if (!last || pad == 0) {
            out[0] = v >> 16;
            out[1] = v >> 8;
            out[2] = v;
            out += 3;
        } else {
            *out++ = v >> 16;
            if (pad == 1) *out++ = v >> 8;
        }
    }
}

std::vector<uint8_t> base64_decode(std::string_view text) {
    std::vector<uint8_t> out(base64_decoded_size(text));
    base64_decode_into(text, out.data());
    return out;
}

}
