/**
 * @file matmul_avx2.cpp
 * @brief AVX2-optimized 32x32 tile matmul implementation.
 *
 * This implementation uses AVX2 and FMA intrinsics to achieve high throughput.
 * The kernel is designed around a 6x16 micro-kernel that maximizes register
 * utilization while avoiding register spilling.
 *
 * Performance characteristics:
 * - 256-bit YMM registers (8 floats per vector)
 * - FMA3 instructions for fused multiply-add
 * - Broadcast-based A operand loading
 * - K-loop unrolling for instruction-level parallelism
 *
 * @see docs/roofline.md for performance analysis
 */

#include "mini_runtime/kernels.hpp"

#if defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>

namespace mini_runtime {

namespace {

/**
 * @brief 6x16 micro-kernel for AVX2 matmul.
 *
 * Processes 6 rows of C and 16 columns using:
 * - 12 YMM accumulators (6 rows × 2 vectors of 8 floats)
 * - 1 YMM for A broadcast
 * - 2 YMM for B loads
 *
 * @param C Output tile pointer (row i, column j_start)
 * @param A Input tile A pointer (row i_start)
 * @param B Input tile B pointer
 * @param i_start Starting row in A/C
 * @param j_start Starting column in B/C
 * @param num_rows Number of rows to process (6 for main loop, 2 for epilogue)
 */
template <int NumRows>
inline void microkernel_avx2(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int i_start,
    int j_start
) {
    static_assert(NumRows == 6 || NumRows == 2, "NumRows must be 6 or 2");

    // Accumulator registers for C[i_start:i_start+NumRows, j_start:j_start+16]
    // Each row needs 2 YMM registers (16 floats = 2×8)
    __m256 c00, c01;  // Row 0: columns [j_start:j_start+8], [j_start+8:j_start+16]
    __m256 c10, c11;  // Row 1
    __m256 c20, c21;  // Row 2
    __m256 c30, c31;  // Row 3
    __m256 c40, c41;  // Row 4
    __m256 c50, c51;  // Row 5

    // Load existing accumulator values from C
    c00 = _mm256_loadu_ps(&C[(i_start + 0) * TILE_DIM + j_start]);
    c01 = _mm256_loadu_ps(&C[(i_start + 0) * TILE_DIM + j_start + 8]);
    c10 = _mm256_loadu_ps(&C[(i_start + 1) * TILE_DIM + j_start]);
    c11 = _mm256_loadu_ps(&C[(i_start + 1) * TILE_DIM + j_start + 8]);

    if constexpr (NumRows >= 6) {
        c20 = _mm256_loadu_ps(&C[(i_start + 2) * TILE_DIM + j_start]);
        c21 = _mm256_loadu_ps(&C[(i_start + 2) * TILE_DIM + j_start + 8]);
        c30 = _mm256_loadu_ps(&C[(i_start + 3) * TILE_DIM + j_start]);
        c31 = _mm256_loadu_ps(&C[(i_start + 3) * TILE_DIM + j_start + 8]);
        c40 = _mm256_loadu_ps(&C[(i_start + 4) * TILE_DIM + j_start]);
        c41 = _mm256_loadu_ps(&C[(i_start + 4) * TILE_DIM + j_start + 8]);
        c50 = _mm256_loadu_ps(&C[(i_start + 5) * TILE_DIM + j_start]);
        c51 = _mm256_loadu_ps(&C[(i_start + 5) * TILE_DIM + j_start + 8]);
    }

    // K-loop: iterate over reduction dimension
    // Unroll by 4 for better instruction-level parallelism
    for (uint32_t k = 0; k < TILE_DIM; k += 4) {
        // Process k, k+1, k+2, k+3 in sequence
        for (int kk = 0; kk < 4; ++kk) {
            uint32_t k_idx = k + kk;

            // Load B[k_idx, j_start:j_start+16] (2 vectors of 8 floats)
            __m256 b0 = _mm256_loadu_ps(&B[k_idx * TILE_DIM + j_start]);
            __m256 b1 = _mm256_loadu_ps(&B[k_idx * TILE_DIM + j_start + 8]);

            // Broadcast A[i, k_idx] and FMA for each row
            __m256 a0 = _mm256_broadcast_ss(&A[(i_start + 0) * TILE_DIM + k_idx]);
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a0, b1, c01);

            __m256 a1 = _mm256_broadcast_ss(&A[(i_start + 1) * TILE_DIM + k_idx]);
            c10 = _mm256_fmadd_ps(a1, b0, c10);
            c11 = _mm256_fmadd_ps(a1, b1, c11);

            if constexpr (NumRows >= 6) {
                __m256 a2 = _mm256_broadcast_ss(&A[(i_start + 2) * TILE_DIM + k_idx]);
                c20 = _mm256_fmadd_ps(a2, b0, c20);
                c21 = _mm256_fmadd_ps(a2, b1, c21);

                __m256 a3 = _mm256_broadcast_ss(&A[(i_start + 3) * TILE_DIM + k_idx]);
                c30 = _mm256_fmadd_ps(a3, b0, c30);
                c31 = _mm256_fmadd_ps(a3, b1, c31);

                __m256 a4 = _mm256_broadcast_ss(&A[(i_start + 4) * TILE_DIM + k_idx]);
                c40 = _mm256_fmadd_ps(a4, b0, c40);
                c41 = _mm256_fmadd_ps(a4, b1, c41);

                __m256 a5 = _mm256_broadcast_ss(&A[(i_start + 5) * TILE_DIM + k_idx]);
                c50 = _mm256_fmadd_ps(a5, b0, c50);
                c51 = _mm256_fmadd_ps(a5, b1, c51);
            }
        }
    }

    // Store accumulators back to C
    _mm256_storeu_ps(&C[(i_start + 0) * TILE_DIM + j_start], c00);
    _mm256_storeu_ps(&C[(i_start + 0) * TILE_DIM + j_start + 8], c01);
    _mm256_storeu_ps(&C[(i_start + 1) * TILE_DIM + j_start], c10);
    _mm256_storeu_ps(&C[(i_start + 1) * TILE_DIM + j_start + 8], c11);

    if constexpr (NumRows >= 6) {
        _mm256_storeu_ps(&C[(i_start + 2) * TILE_DIM + j_start], c20);
        _mm256_storeu_ps(&C[(i_start + 2) * TILE_DIM + j_start + 8], c21);
        _mm256_storeu_ps(&C[(i_start + 3) * TILE_DIM + j_start], c30);
        _mm256_storeu_ps(&C[(i_start + 3) * TILE_DIM + j_start + 8], c31);
        _mm256_storeu_ps(&C[(i_start + 4) * TILE_DIM + j_start], c40);
        _mm256_storeu_ps(&C[(i_start + 4) * TILE_DIM + j_start + 8], c41);
        _mm256_storeu_ps(&C[(i_start + 5) * TILE_DIM + j_start], c50);
        _mm256_storeu_ps(&C[(i_start + 5) * TILE_DIM + j_start + 8], c51);
    }
}

}  // namespace

void matmul_tile_avx2(float* C, const float* A, const float* B) {
    // Process 32x32 tile using 6x16 micro-kernels
    //
    // Layout:
    //   - 32 rows = 5 × 6 + 2 (main iterations + remainder)
    //   - 32 cols = 2 × 16 (two micro-kernel widths)
    //
    // For each column block (0-15, 16-31):
    //   - Process rows 0-5, 6-11, 12-17, 18-23, 24-29 (5 iterations of 6 rows)
    //   - Process rows 30-31 (2-row epilogue)

    // Process column block [0:16]
    for (int i = 0; i < 30; i += 6) {
        microkernel_avx2<6>(C, A, B, i, 0);
    }
    // Handle remaining 2 rows for column block [0:16]
    microkernel_avx2<2>(C, A, B, 30, 0);

    // Process column block [16:32]
    for (int i = 0; i < 30; i += 6) {
        microkernel_avx2<6>(C, A, B, i, 16);
    }
    // Handle remaining 2 rows for column block [16:32]
    microkernel_avx2<2>(C, A, B, 30, 16);
}

void relu_tile_avx2(float* C) {
    // SIMD-vectorized ReLU: C[i] = max(0, C[i])
    //
    // Process 8 floats at a time using AVX2.
    // Total elements: 32 × 32 = 1024 = 128 × 8
    
    __m256 zero = _mm256_setzero_ps();
    
    constexpr uint32_t TILE_ELEMS = TILE_DIM * TILE_DIM;
    for (uint32_t i = 0; i < TILE_ELEMS; i += 8) {
        __m256 val = _mm256_loadu_ps(&C[i]);
        __m256 result = _mm256_max_ps(val, zero);
        _mm256_storeu_ps(&C[i], result);
    }
}

void matmul_tile_relu_avx2(float* C, const float* A, const float* B) {
    // Fused matmul + ReLU using AVX2
    //
    // Note: True fusion would apply ReLU during the store phase of the
    // micro-kernel. For simplicity, we call the two operations sequentially
    // here, which is still efficient since C is in L1 cache after matmul.
    
    matmul_tile_avx2(C, A, B);
    relu_tile_avx2(C);
}

}  // namespace mini_runtime

#endif  // defined(__AVX2__) && defined(__FMA__)
