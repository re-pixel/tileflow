/**
 * @file constants.hpp
 * @brief Global constants for the mini-runtime.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace mini_runtime {

/// Tile dimension (fixed 32x32 tiles)
constexpr uint32_t TILE_DIM = 32;

/// Tile size in bytes (32 * 32 * 4 bytes per float)
constexpr size_t TILE_BYTES = TILE_DIM * TILE_DIM * sizeof(float);

/// Memory alignment for SIMD operations (cache line)
constexpr size_t ALIGNMENT = 64;

}  // namespace mini_runtime
