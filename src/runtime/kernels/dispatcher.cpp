/**
 * @file dispatcher.cpp
 * @brief Kernel dispatcher for runtime selection between implementations.
 *
 * This module provides unified entry points that automatically dispatch to the
 * best available kernel implementation based on compile-time feature detection.
 *
 * Dispatch strategy:
 * - If compiled with AVX2+FMA support, use AVX2 kernels
 * - Otherwise, fall back to reference implementations
 *
 * The explicit `KernelImpl` parameter allows benchmarks to compare implementations
 * on the same machine.
 */

#include "mini_runtime/kernels.hpp"

namespace mini_runtime {

// ============================================================================
// Auto-dispatched entry points (select best available implementation)
// ============================================================================

void matmul_tile(float* C, const float* A, const float* B) {
#if defined(__AVX2__) && defined(__FMA__)
    matmul_tile_avx2(C, A, B);
#else
    matmul_tile_ref(C, A, B);
#endif
}

void relu_tile(float* C) {
#if defined(__AVX2__) && defined(__FMA__)
    relu_tile_avx2(C);
#else
    relu_tile_inplace(C);
#endif
}

void matmul_tile_relu(float* C, const float* A, const float* B) {
#if defined(__AVX2__) && defined(__FMA__)
    matmul_tile_relu_avx2(C, A, B);
#else
    matmul_tile_relu_ref(C, A, B);
#endif
}

// ============================================================================
// Explicit implementation selection (for benchmarking)
// ============================================================================

void matmul_tile(float* C, const float* A, const float* B, KernelImpl impl) {
    switch (impl) {
        case KernelImpl::Reference:
            matmul_tile_ref(C, A, B);
            break;
        case KernelImpl::AVX2:
#if defined(__AVX2__) && defined(__FMA__)
            matmul_tile_avx2(C, A, B);
#else
            // Fall back to reference if AVX2 not available
            matmul_tile_ref(C, A, B);
#endif
            break;
    }
}

void relu_tile(float* C, KernelImpl impl) {
    switch (impl) {
        case KernelImpl::Reference:
            relu_tile_inplace(C);
            break;
        case KernelImpl::AVX2:
#if defined(__AVX2__) && defined(__FMA__)
            relu_tile_avx2(C);
#else
            relu_tile_inplace(C);
#endif
            break;
    }
}

void matmul_tile_relu(float* C, const float* A, const float* B, KernelImpl impl) {
    switch (impl) {
        case KernelImpl::Reference:
            matmul_tile_relu_ref(C, A, B);
            break;
        case KernelImpl::AVX2:
#if defined(__AVX2__) && defined(__FMA__)
            matmul_tile_relu_avx2(C, A, B);
#else
            matmul_tile_relu_ref(C, A, B);
#endif
            break;
    }
}

// ============================================================================
// Utility functions
// ============================================================================

bool is_avx2_available() {
#if defined(__AVX2__) && defined(__FMA__)
    return true;
#else
    return false;
#endif
}

const char* get_active_kernel_name() {
#if defined(__AVX2__) && defined(__FMA__)
    return "AVX2+FMA";
#else
    return "Reference";
#endif
}

}  // namespace mini_runtime
