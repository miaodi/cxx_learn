#pragma once
#include <cstdint>
namespace MCRand {
std::uint64_t twiddle_origin(std::uint64_t u, std::uint64_t v);

std::uint64_t twiddle_new(std::uint64_t u, std::uint64_t v);
} // namespace MCRand