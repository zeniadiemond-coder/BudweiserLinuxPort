#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>

namespace openv6 {

std::array<uint8_t, 16> md5(std::span<const uint8_t> data);
std::string md5_hex(std::span<const uint8_t> data);

}
