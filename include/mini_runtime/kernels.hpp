/**
 * @file kernels.hpp
 * @brief Tile-level compute kernels.
 *
 * All kernels operate on fixed 32x32 tiles.
 *
 * This header declares:
 * - Reference implementations (matmul_tile_ref, relu_tile_inplace)
 * - AVX2-optimized implementations (matmul_tile_avx2, relu_tile_avx2)
 * - Dispatched entry points (matmul_tile, relu_tile)
 * - Explicit implementation selection for benchmarking
 */

#pragma once

#include <cstdint>

#include "mini_runtime/constants.hpp"

namespace mini_runtime {

// ============================================================================
// Kernel implementation selector (for benchmarking)
// ============================================================================

/**
 * @brief Enumeration of available kernel implementations.
 *
 * Used with explicit dispatch functions to select specific implementations
 * for benchmarking and comparison purposes.
 */
enum class KernelImpl {
    Reference,  ///< Naive triple-nested loop (correctness-first)
    AVX2        ///< AVX2+FMA optimized with 6x16 micro-kernel
};

// ============================================================================
// Reference implementations (always available)
// ============================================================================

/**
 * @brief Reference implementation of 32x32 tile matrix multiply-accumulate.
 *
 * Computes: C[32x32] += A[32x32] @ B[32x32]
 *
 * @param C Accumulator tile (read-modify-write). Must be ALIGNMENT-aligned.
 * @param A Left operand tile (read-only). Must be ALIGNMENT-aligned.
 * @param B Right operand tile (read-only). Must be ALIGNMENT-aligned.
 *
 * @note This is a correctness-first implementation using naive triple-nested
 *       loops. Use matmul_tile() for the auto-dispatched optimized version.
 */
void matmul_tile_ref(float* C, const float* A, const float* B);

/**
 * @brief Apply ReLU activation in-place to a 32x32 tile (reference).
 *
 * Computes: C[i] = max(0, C[i]) for all elements.
 *
 * @param C Tile to modify in-place.
 */
void relu_tile_inplace(float* C);

/**
 * @brief Fused matmul + ReLU reference implementation.
 *
 * Computes: C += A @ B, then C = max(0, C)
 *
 * @note Calls matmul_tile_ref + relu_tile_inplace sequentially.
 */
void matmul_tile_relu_ref(float* C, const float* A, const float* B);

// ============================================================================
// AVX2-optimized implementations (available when compiled with -mavx2 -mfma)
// ============================================================================

#if defined(__AVX2__) && defined(__FMA__)

/**
 * @brief AVX2-optimized 32x32 tile matrix multiply-accumulate.
 *
 * Uses a 6x16 micro-kernel design that maximizes FMA throughput:
 * - 12 YMM accumulators (6 rows × 2 vectors)
 * - Broadcast-based A loading
 * - K-loop unrolling by 4
 *
 * @param C Accumulator tile (read-modify-write).
 * @param A Left operand tile (read-only).
 * @param B Right operand tile (read-only).
 *
 * @see docs/roofline.md for performance analysis.
 */
void matmul_tile_avx2(float* C, const float* A, const float* B);

/**
 * @brief AVX2-optimized ReLU activation.
 *
 * Processes 8 floats at a time using _mm256_max_ps.
 *
 * @param C Tile to modify in-place.
 */
void relu_tile_avx2(float* C);

/**
 * @brief AVX2-optimized fused matmul + ReLU.
 *
 * @param C Accumulator tile (read-modify-write).
 * @param A Left operand tile (read-only).
 * @param B Right operand tile (read-only).
 */
void matmul_tile_relu_avx2(float* C, const float* A, const float* B);

#endif  // defined(__AVX2__) && defined(__FMA__)

// ============================================================================
// Dispatched entry points (auto-select best implementation)
// ============================================================================

/**
 * @brief Dispatch to best available matmul implementation.
 *
 * Selects AVX2 if compiled with -mavx2 -mfma, otherwise uses reference.
 */
void matmul_tile(float* C, const float* A, const float* B);

/**
 * @brief Dispatch to best available ReLU implementation.
 */
void relu_tile(float* C);

/**
 * @brief Dispatch to best available fused matmul+ReLU implementation.
 */
void matmul_tile_relu(float* C, const float* A, const float* B);

// ============================================================================
// Explicit implementation selection (for benchmarking)
// ============================================================================

/**
 * @brief Execute matmul with explicitly selected implementation.
 *
 * @param impl Implementation to use. Falls back to Reference if AVX2
 *             is requested but not available.
 */
void matmul_tile(float* C, const float* A, const float* B, KernelImpl impl);

/**
 * @brief Execute ReLU with explicitly selected implementation.
 */
void relu_tile(float* C, KernelImpl impl);

/**
 * @brief Execute fused matmul+ReLU with explicitly selected implementation.
 */
void matmul_tile_relu(float* C, const float* A, const float* B, KernelImpl impl);

// ============================================================================
// Utility functions
// ============================================================================

/**
 * @brief Check if AVX2 kernels are available at compile time.
 *
 * @return true if compiled with AVX2+FMA support.
 */
bool is_avx2_available();

/**
 * @brief Get the name of the currently active kernel implementation.
 *
 * @return "AVX2+FMA" or "Reference" depending on compile flags.
 */
const char* get_active_kernel_name();

}  // namespace mini_runtime
