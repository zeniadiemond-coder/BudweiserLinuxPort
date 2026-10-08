#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace openv6 {

std::vector<uint8_t> base64_decode(std::string_view text);

size_t base64_decoded_size(std::string_view text);
void base64_decode_into(std::string_view text, void* out);

}
