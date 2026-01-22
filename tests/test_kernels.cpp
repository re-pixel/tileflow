/**
 * @file test_kernels.cpp
 * @brief Unit tests for tile-level compute kernels.
 *
 * Tests cover:
 * - Reference kernel correctness
 * - AVX2 kernel correctness (when available)
 * - Numerical accuracy between implementations
 * - Edge cases (identity, accumulation, large values)
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <memory>
#include <random>

#include "mini_runtime/kernels.hpp"
#include "mini_runtime/constants.hpp"

using namespace mini_runtime;

namespace {

// Aligned memory allocation helper
template <typename T>
T* aligned_alloc_array(size_t count) {
    void* ptr = std::aligned_alloc(ALIGNMENT, count * sizeof(T));
    return static_cast<T*>(ptr);
}

// RAII wrapper for aligned memory
class AlignedTile {
public:
    AlignedTile() : data_(aligned_alloc_array<float>(TILE_DIM * TILE_DIM)) {
        clear();
    }
    
    ~AlignedTile() {
        std::free(data_);
    }
    
    // Non-copyable
    AlignedTile(const AlignedTile&) = delete;
    AlignedTile& operator=(const AlignedTile&) = delete;
    
    float* data() { return data_; }
    const float* data() const { return data_; }
    
    float& at(int i, int j) { return data_[i * TILE_DIM + j]; }
    const float& at(int i, int j) const { return data_[i * TILE_DIM + j]; }
    
    void clear() {
        for (size_t i = 0; i < TILE_DIM * TILE_DIM; ++i) {
            data_[i] = 0.0f;
        }
    }
    
    void fill(float value) {
        for (size_t i = 0; i < TILE_DIM * TILE_DIM; ++i) {
            data_[i] = value;
        }
    }
    
    void set_identity() {
        clear();
        for (uint32_t i = 0; i < TILE_DIM; ++i) {
            at(i, i) = 1.0f;
        }
    }
    
    void randomize(std::mt19937& rng, float scale = 1.0f) {
        std::normal_distribution<float> dist(0.0f, scale);
        for (size_t i = 0; i < TILE_DIM * TILE_DIM; ++i) {
            data_[i] = dist(rng);
        }
    }
    
    void copy_from(const AlignedTile& other) {
        for (size_t i = 0; i < TILE_DIM * TILE_DIM; ++i) {
            data_[i] = other.data_[i];
        }
    }
    
private:
    float* data_;
};

// Compare two tiles with tolerance
bool tiles_equal(const AlignedTile& a, const AlignedTile& b, float rtol = 1e-5f, float atol = 1e-6f) {
    for (size_t i = 0; i < TILE_DIM * TILE_DIM; ++i) {
        float diff = std::abs(a.data()[i] - b.data()[i]);
        float tol = atol + rtol * std::abs(b.data()[i]);
        if (diff > tol) {
            return false;
        }
    }
    return true;
}

// Compute max absolute difference between tiles
float max_diff(const AlignedTile& a, const AlignedTile& b) {
    float max_d = 0.0f;
    for (size_t i = 0; i < TILE_DIM * TILE_DIM; ++i) {
        float diff = std::abs(a.data()[i] - b.data()[i]);
        max_d = std::max(max_d, diff);
    }
    return max_d;
}

}  // namespace

// =============================================================================
// Reference Kernel Tests
// =============================================================================

class ReferenceKernelTest : public ::testing::Test {
protected:
    AlignedTile A, B, C;
    std::mt19937 rng{42};
};

TEST_F(ReferenceKernelTest, IdentityMatrixLeft) {
    // C = 0; A = I; B = random => C = B
    A.set_identity();
    B.randomize(rng);
    C.clear();
    
    matmul_tile_ref(C.data(), A.data(), B.data());
    
    EXPECT_TRUE(tiles_equal(C, B));
}

TEST_F(ReferenceKernelTest, IdentityMatrixRight) {
    // C = 0; A = random; B = I => C = A
    A.randomize(rng);
    B.set_identity();
    C.clear();
    
    matmul_tile_ref(C.data(), A.data(), B.data());
    
    EXPECT_TRUE(tiles_equal(C, A));
}

TEST_F(ReferenceKernelTest, Accumulation) {
    // C = 1; A = I; B = I => C = 1 + I
    C.fill(1.0f);
    A.set_identity();
    B.set_identity();
    
    matmul_tile_ref(C.data(), A.data(), B.data());
    
    // Check diagonal elements are 2.0, off-diagonal are 1.0
    for (uint32_t i = 0; i < TILE_DIM; ++i) {
        for (uint32_t j = 0; j < TILE_DIM; ++j) {
            float expected = (i == j) ? 2.0f : 1.0f;
            EXPECT_FLOAT_EQ(C.at(i, j), expected) << "at (" << i << ", " << j << ")";
        }
    }
}

TEST_F(ReferenceKernelTest, ZeroMatrix) {
    // C = random; A = 0; B = random => C = C_initial
    AlignedTile C_initial;
    C.randomize(rng);
    C_initial.copy_from(C);
    A.clear();
    B.randomize(rng);
    
    matmul_tile_ref(C.data(), A.data(), B.data());
    
    EXPECT_TRUE(tiles_equal(C, C_initial));
}

TEST_F(ReferenceKernelTest, ReLUPositive) {
    // All positive values should remain unchanged
    for (size_t i = 0; i < TILE_DIM * TILE_DIM; ++i) {
        C.data()[i] = static_cast<float>(i + 1);
    }
    
    AlignedTile expected;
    expected.copy_from(C);
    
    relu_tile_inplace(C.data());
    
    EXPECT_TRUE(tiles_equal(C, expected));
}

TEST_F(ReferenceKernelTest, ReLUNegative) {
    // All negative values should become zero
    for (size_t i = 0; i < TILE_DIM * TILE_DIM; ++i) {
        C.data()[i] = -static_cast<float>(i + 1);
    }
    
    relu_tile_inplace(C.data());
    
    for (size_t i = 0; i < TILE_DIM * TILE_DIM; ++i) {
        EXPECT_FLOAT_EQ(C.data()[i], 0.0f);
    }
}

TEST_F(ReferenceKernelTest, ReLUMixed) {
    // Mixed positive/negative values
    C.randomize(rng);
    
    AlignedTile expected;
    for (size_t i = 0; i < TILE_DIM * TILE_DIM; ++i) {
        expected.data()[i] = std::max(0.0f, C.data()[i]);
    }
    
    relu_tile_inplace(C.data());
    
    EXPECT_TRUE(tiles_equal(C, expected));
}

// =============================================================================
// AVX2 Kernel Tests (conditional compilation)
// =============================================================================

#if defined(__AVX2__) && defined(__FMA__)

class AVX2KernelTest : public ::testing::Test {
protected:
    AlignedTile A, B, C_ref, C_avx2;
    std::mt19937 rng{42};
};

TEST_F(AVX2KernelTest, MatchesReference) {
    // Random matrices should produce identical results
    A.randomize(rng, 0.1f);
    B.randomize(rng, 0.1f);
    C_ref.clear();
    C_avx2.clear();
    
    matmul_tile_ref(C_ref.data(), A.data(), B.data());
    matmul_tile_avx2(C_avx2.data(), A.data(), B.data());
    
    float diff = max_diff(C_ref, C_avx2);
    EXPECT_LT(diff, 1e-5f) << "Max diff: " << diff;
    EXPECT_TRUE(tiles_equal(C_ref, C_avx2));
}

TEST_F(AVX2KernelTest, MatchesReferenceWithAccumulation) {
    // Test with non-zero initial C
    A.randomize(rng, 0.1f);
    B.randomize(rng, 0.1f);
    C_ref.randomize(rng, 0.01f);
    C_avx2.copy_from(C_ref);
    
    matmul_tile_ref(C_ref.data(), A.data(), B.data());
    matmul_tile_avx2(C_avx2.data(), A.data(), B.data());
    
    float diff = max_diff(C_ref, C_avx2);
    EXPECT_LT(diff, 1e-4f) << "Max diff: " << diff;
    EXPECT_TRUE(tiles_equal(C_ref, C_avx2, 1e-4f, 1e-5f));
}

TEST_F(AVX2KernelTest, IdentityMatrix) {
    // A = I => C = B
    A.set_identity();
    B.randomize(rng);
    C_ref.clear();
    C_avx2.clear();
    
    matmul_tile_avx2(C_avx2.data(), A.data(), B.data());
    
    EXPECT_TRUE(tiles_equal(C_avx2, B));
}

TEST_F(AVX2KernelTest, LargeValues) {
    // Test numerical stability with larger values
    A.randomize(rng, 100.0f);
    B.randomize(rng, 100.0f);
    C_ref.clear();
    C_avx2.clear();
    
    matmul_tile_ref(C_ref.data(), A.data(), B.data());
    matmul_tile_avx2(C_avx2.data(), A.data(), B.data());
    
    // Use relative tolerance for large values
    EXPECT_TRUE(tiles_equal(C_ref, C_avx2, 1e-4f, 1e-2f));
}

TEST_F(AVX2KernelTest, SmallValues) {
    // Test with very small values
    A.randomize(rng, 1e-4f);
    B.randomize(rng, 1e-4f);
    C_ref.clear();
    C_avx2.clear();
    
    matmul_tile_ref(C_ref.data(), A.data(), B.data());
    matmul_tile_avx2(C_avx2.data(), A.data(), B.data());
    
    EXPECT_TRUE(tiles_equal(C_ref, C_avx2, 1e-4f, 1e-10f));
}

TEST_F(AVX2KernelTest, ReLUMatchesReference) {
    // ReLU should produce identical results
    AlignedTile C_relu_ref, C_relu_avx2;
    C_relu_ref.randomize(rng);
    C_relu_avx2.copy_from(C_relu_ref);
    
    relu_tile_inplace(C_relu_ref.data());
    relu_tile_avx2(C_relu_avx2.data());
    
    EXPECT_TRUE(tiles_equal(C_relu_ref, C_relu_avx2, 0.0f, 0.0f));
}

TEST_F(AVX2KernelTest, FusedMatmulReLU) {
    // Test fused operation
    A.randomize(rng, 0.1f);
    B.randomize(rng, 0.1f);
    C_ref.clear();
    C_avx2.clear();
    
    matmul_tile_relu_ref(C_ref.data(), A.data(), B.data());
    matmul_tile_relu_avx2(C_avx2.data(), A.data(), B.data());
    
    EXPECT_TRUE(tiles_equal(C_ref, C_avx2, 1e-4f, 1e-5f));
}

TEST_F(AVX2KernelTest, MultipleIterations) {
    // Test multiple matmuls accumulating into C
    A.randomize(rng, 0.05f);
    B.randomize(rng, 0.05f);
    C_ref.clear();
    C_avx2.clear();
    
    for (int iter = 0; iter < 10; ++iter) {
        matmul_tile_ref(C_ref.data(), A.data(), B.data());
        matmul_tile_avx2(C_avx2.data(), A.data(), B.data());
    }
    
    EXPECT_TRUE(tiles_equal(C_ref, C_avx2, 1e-3f, 1e-4f));
}

#endif  // defined(__AVX2__) && defined(__FMA__)

// =============================================================================
// Dispatcher Tests
// =============================================================================

class DispatcherTest : public ::testing::Test {
protected:
    AlignedTile A, B, C;
    std::mt19937 rng{42};
};

TEST_F(DispatcherTest, AutoDispatch) {
    // Auto-dispatched kernel should produce correct results
    A.randomize(rng, 0.1f);
    B.randomize(rng, 0.1f);
    
    AlignedTile C_expected;
    C_expected.clear();
    C.clear();
    
    // Reference for comparison
    matmul_tile_ref(C_expected.data(), A.data(), B.data());
    
    // Auto-dispatched
    matmul_tile(C.data(), A.data(), B.data());
    
    EXPECT_TRUE(tiles_equal(C, C_expected, 1e-4f, 1e-5f));
}

TEST_F(DispatcherTest, ExplicitReference) {
    A.randomize(rng, 0.1f);
    B.randomize(rng, 0.1f);
    
    AlignedTile C_expected;
    C_expected.clear();
    C.clear();
    
    matmul_tile_ref(C_expected.data(), A.data(), B.data());
    matmul_tile(C.data(), A.data(), B.data(), KernelImpl::Reference);
    
    EXPECT_TRUE(tiles_equal(C, C_expected, 0.0f, 0.0f));
}

TEST_F(DispatcherTest, ExplicitAVX2) {
    A.randomize(rng, 0.1f);
    B.randomize(rng, 0.1f);
    
    AlignedTile C_expected;
    C_expected.clear();
    C.clear();
    
    matmul_tile_ref(C_expected.data(), A.data(), B.data());
    matmul_tile(C.data(), A.data(), B.data(), KernelImpl::AVX2);
    
    // AVX2 falls back to reference if not available
    EXPECT_TRUE(tiles_equal(C, C_expected, 1e-4f, 1e-5f));
}

TEST_F(DispatcherTest, AVX2Available) {
    // This test just checks the API works
    bool avx2 = is_avx2_available();
    const char* name = get_active_kernel_name();
    
    if (avx2) {
        EXPECT_STREQ(name, "AVX2+FMA");
    } else {
        EXPECT_STREQ(name, "Reference");
    }
}

// =============================================================================
// Performance Sanity Tests
// =============================================================================

#if defined(__AVX2__) && defined(__FMA__)

TEST(PerformanceTest, AVX2FasterThanReference) {
    // This is a sanity check that AVX2 is actually faster
    // Not a strict benchmark, but catches obvious regressions
    
    constexpr int WARMUP = 10;
    constexpr int ITERS = 100;
    
    AlignedTile A, B, C;
    std::mt19937 rng{123};
    A.randomize(rng);
    B.randomize(rng);
    
    // Warmup reference
    for (int i = 0; i < WARMUP; ++i) {
        C.clear();
        matmul_tile_ref(C.data(), A.data(), B.data());
    }
    
    // Time reference
    auto ref_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        matmul_tile_ref(C.data(), A.data(), B.data());
    }
    auto ref_end = std::chrono::high_resolution_clock::now();
    auto ref_us = std::chrono::duration_cast<std::chrono::microseconds>(ref_end - ref_start).count();
    
    // Warmup AVX2
    for (int i = 0; i < WARMUP; ++i) {
        C.clear();
        matmul_tile_avx2(C.data(), A.data(), B.data());
    }
    
    // Time AVX2
    auto avx2_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        matmul_tile_avx2(C.data(), A.data(), B.data());
    }
    auto avx2_end = std::chrono::high_resolution_clock::now();
    auto avx2_us = std::chrono::duration_cast<std::chrono::microseconds>(avx2_end - avx2_start).count();
    
    // AVX2 should be at least 2x faster (conservative threshold for CI)
    double speedup = static_cast<double>(ref_us) / static_cast<double>(avx2_us);
    EXPECT_GT(speedup, 2.0) << "AVX2 speedup: " << speedup << "x (ref: " << ref_us << "us, avx2: " << avx2_us << "us)";
}

#endif  // defined(__AVX2__) && defined(__FMA__)

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
