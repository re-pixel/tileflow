/**
 * @file kernels.hpp
 * @brief Tile-level compute kernels.
 *
 * All kernels operate on fixed 32x32 tiles.
 */

#pragma once

#include <cstdint>

#include "mini_runtime/constants.hpp"

namespace mini_runtime {

/**
 * @brief Reference implementation of 32x32 tile matrix multiply-accumulate.
 *
 * Computes: C[32x32] += A[32x32] @ B[32x32]
 *
 * @param C Accumulator tile (read-modify-write). Must be ALIGNMENT-aligned.
 * @param A Left operand tile (read-only). Must be ALIGNMENT-aligned.
 * @param B Right operand tile (read-only). Must be ALIGNMENT-aligned.
 *
 * @note This is a correctness-first implementation. The AVX2/AVX-512
 *       optimized versions are implemented in Week 5.
 */
void matmul_tile_ref(float* C, const float* A, const float* B);

/**
 * @brief Apply ReLU activation in-place to a 32x32 tile.
 *
 * Computes: C[i] = max(0, C[i]) for all elements.
 *
 * @param C Tile to modify in-place.
 */
void relu_tile_inplace(float* C);

/**
 * @brief Fused matmul + ReLU (for future optimization).
 *
 * Computes: C += A @ B, then C = max(0, C)
 *
 * @note Currently calls matmul_tile_ref + relu_tile_inplace.
 *       Can be fused for better cache utilization.
 */
void matmul_tile_relu_ref(float* C, const float* A, const float* B);

}  // namespace mini_runtime
