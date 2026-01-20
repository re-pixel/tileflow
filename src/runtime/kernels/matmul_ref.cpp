/**
 * @file matmul_ref.cpp
 * @brief Reference implementation of tile-level matmul and activations.
 *
 * This is a correctness-first implementation using naive triple-nested loops.
 * Performance optimizations (AVX2/AVX-512) are implemented in Week 5.
 */

#include "mini_runtime/kernels.hpp"

namespace mini_runtime {

void matmul_tile_ref(float* C, const float* A, const float* B) {
    // C[32x32] += A[32x32] @ B[32x32]
    // 
    // A is [TILE_DIM x TILE_DIM], row-major: A[i,k] = A[i * TILE_DIM + k]
    // B is [TILE_DIM x TILE_DIM], row-major: B[k,j] = B[k * TILE_DIM + j]
    // C is [TILE_DIM x TILE_DIM], row-major: C[i,j] = C[i * TILE_DIM + j]
    //
    // This implementation prioritizes clarity over performance.
    // Loop order: i (row), j (col), k (reduction) — standard row-major order.

    for (uint32_t i = 0; i < TILE_DIM; ++i) {
        for (uint32_t j = 0; j < TILE_DIM; ++j) {
            float sum = C[i * TILE_DIM + j];  // Start with existing accumulator value
            
            for (uint32_t k = 0; k < TILE_DIM; ++k) {
                sum += A[i * TILE_DIM + k] * B[k * TILE_DIM + j];
            }
            
            C[i * TILE_DIM + j] = sum;
        }
    }
}

void relu_tile_inplace(float* C) {
    // ReLU: C[i] = max(0, C[i])
    constexpr uint32_t TILE_ELEMS = TILE_DIM * TILE_DIM;
    
    for (uint32_t i = 0; i < TILE_ELEMS; ++i) {
        if (C[i] < 0.0f) {
            C[i] = 0.0f;
        }
    }
}

void matmul_tile_relu_ref(float* C, const float* A, const float* B) {
    // Fused version: C += A @ B, then ReLU(C)
    // Currently implemented as two separate calls.
    // Can be fused for better cache behavior in future optimization.
    matmul_tile_ref(C, A, B);
    relu_tile_inplace(C);
}

}  // namespace mini_runtime
