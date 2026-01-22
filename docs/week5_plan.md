# Week 5 Plan — SIMD Tiled Kernels & Performance Analysis

**Date:** 2026-01-22  
**Phase:** Phase 2 (Backend Optimization, C++)  
**Week 5 Goal:** Implement AVX2-optimized $32\times 32$ matmul kernels and establish a performance analysis framework using roofline modeling.

---

## 1) Executive Summary

Week 5 transitions from "correctness" to "performance." The reference kernel from Week 4 is intentionally naive (triple-nested loops). This week, we implement SIMD-optimized kernels and measure their performance against theoretical hardware limits.

| Component | Description |
|-----------|-------------|
| **AVX2 Kernel** | SIMD-vectorized $32\times 32$ matmul using 256-bit registers |
| **Micro-kernel Design** | Register-blocked inner loop for maximum FMA throughput |
| **Roofline Model** | Framework to understand compute vs. memory boundedness |
| **Benchmarking Suite** | Isolated kernel benchmarks + end-to-end MLP timing |
| **Kernel Dispatcher** | Runtime selection between reference and AVX2 implementations |

**Key Insight:** A $32\times 32$ tile is $32 \times 32 \times 4 = 4096$ bytes (4 KiB). This fits comfortably in L1 cache (typically 32-48 KiB). The kernel is therefore **compute-bound**, making SIMD optimization highly effective.

---

## 2) Background: Why SIMD Matters

### 2.1 The Reference Kernel Problem

The Week 4 reference kernel executes:
- $32 \times 32 \times 32 = 32{,}768$ multiply-accumulate operations per tile
- Each operation is 2 FLOPs (multiply + add) → **65,536 FLOPs per tile**

With scalar code, each FLOP requires one CPU cycle. A 4.5 GHz CPU achieves ~4.5 GFLOP/s scalar.

### 2.2 AVX2 Potential

AVX2 provides:
- **256-bit registers** (YMM0–YMM15) holding 8 floats each
- **FMA instructions** (`vfmadd231ps`) that compute $a \times b + c$ in one instruction
- **Two FMA units** on modern CPUs (Haswell+) for 2×16 = 32 FLOPs/cycle

Theoretical peak on a 4.5 GHz CPU: $4.5 \times 10^9 \times 32 = 144$ GFLOP/s.

**Realistic target:** 4–8× speedup over reference (limited by register pressure, memory latency, and instruction overhead).

---

## 3) Design Decisions & Rationale

### 3.1 Micro-kernel Strategy

**Decision:** Implement a $6 \times 16$ micro-kernel as the inner loop.

**Rationale:**
- AVX2 has 16 YMM registers. We need registers for:
  - **Accumulators:** The "hot" values we're computing
  - **A operands:** Broadcast from matrix A
  - **B operands:** Loaded from matrix B
- A $6 \times 16$ micro-kernel uses:
  - 12 registers for accumulators (6 rows × 2 YMM vectors per row)
  - 1 register for A broadcast
  - 2 registers for B loads
  - 1 spare for address computation
- This maximizes register utilization without spilling

**Alternative considered:** $8 \times 8$ micro-kernel (simpler but lower throughput due to more broadcast overhead).

### 3.2 Memory Layout

**Decision:** Keep row-major layout. Do not transpose B.

**Rationale:**
- Transposing B would require a pre-pass (extra memory traffic)
- For $32\times 32$ tiles that fit in L1, the penalty of strided access to B is minimal
- Keeping the layout consistent with the reference kernel simplifies validation

**Future optimization (not Week 5):** Pre-pack B tiles in a "panel" layout for better vectorization.

### 3.3 Kernel Dispatch

**Decision:** Runtime dispatch based on compile-time feature detection.

**Implementation:**
```cpp
// In kernels.hpp
void matmul_tile(float* C, const float* A, const float* B);

// In dispatcher.cpp
void matmul_tile(float* C, const float* A, const float* B) {
#if defined(__AVX2__) && defined(__FMA__)
    matmul_tile_avx2(C, A, B);
#else
    matmul_tile_ref(C, A, B);
#endif
}
```

**Rationale:**
- Simple and predictable
- Avoids CPUID overhead at runtime
- User compiles with `-mavx2 -mfma` if their CPU supports it

---

## 4) Implementation Steps

### 4.1 AVX2 Kernel Implementation (`src/runtime/kernels/matmul_avx2.cpp`)

**Task:** Implement the optimized $32\times 32$ matmul.

**Pseudo-code (high-level structure):**

```cpp
void matmul_tile_avx2(float* C, const float* A, const float* B) {
    // Process 6 rows of A at a time, 16 columns of B at a time
    for (int i = 0; i < 32; i += 6) {
        // Load 6×16 accumulator block from C into registers
        __m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
        // ... load from C[i:i+6, 0:16] and C[i:i+6, 16:32] ...

        for (int k = 0; k < 32; ++k) {
            // Broadcast A[i+0, k], A[i+1, k], ..., A[i+5, k]
            __m256 a0 = _mm256_broadcast_ss(&A[(i+0)*32 + k]);
            // ... repeat for a1–a5 ...

            // Load B[k, 0:8] and B[k, 8:16]
            __m256 b0 = _mm256_loadu_ps(&B[k*32 + 0]);
            __m256 b1 = _mm256_loadu_ps(&B[k*32 + 8]);

            // FMA: c_ij += a_i * b_j
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a0, b1, c01);
            // ... repeat for all 12 accumulators ...
        }

        // Store accumulators back to C
        _mm256_storeu_ps(&C[(i+0)*32 + 0], c00);
        // ... repeat for all stores ...
    }
    
    // Handle remaining 2 rows (32 = 6*5 + 2)
    // Use a smaller micro-kernel or scalar fallback
}
```

**Implementation notes:**
- Use `_mm256_fmadd_ps` (FMA3) for fused multiply-add
- Use `_mm256_broadcast_ss` to replicate A element across 8 lanes
- Unroll the k-loop by 4 for instruction-level parallelism
- Handle the "remainder" rows (32 mod 6 = 2) with a 2×16 epilogue

**Deliverable:** `matmul_avx2.cpp` with a working, tested AVX2 kernel.

---

### 4.2 AVX2 ReLU Implementation

**Task:** SIMD-vectorized ReLU for post-matmul activation.

```cpp
void relu_tile_avx2(float* C) {
    __m256 zero = _mm256_setzero_ps();
    for (int i = 0; i < 32*32; i += 8) {
        __m256 val = _mm256_loadu_ps(&C[i]);
        __m256 result = _mm256_max_ps(val, zero);
        _mm256_storeu_ps(&C[i], result);
    }
}
```

**Note:** ReLU is memory-bound (trivial compute). The speedup will be modest, but it's good practice.

---

### 4.3 Kernel Dispatcher (`src/runtime/kernels/dispatcher.cpp`)

**Task:** Create unified entry points that select the best implementation.

**API:**
```cpp
namespace mini_runtime {

// Unified dispatch (auto-selects best kernel)
void matmul_tile(float* C, const float* A, const float* B);
void relu_tile(float* C);
void matmul_tile_relu(float* C, const float* A, const float* B);

// Explicit selection (for benchmarking)
enum class KernelImpl { Reference, AVX2 };
void matmul_tile(float* C, const float* A, const float* B, KernelImpl impl);

}  // namespace mini_runtime
```

**Deliverable:** `dispatcher.cpp` that routes to ref or AVX2 based on compile flags.

---

### 4.4 Update Engine to Use Dispatched Kernels

**Task:** Modify `engine.cpp` to call `matmul_tile()` instead of `matmul_tile_ref()`.

**Change:**
```cpp
// Before (Week 4):
matmul_tile_ref(c_ptr, a_ptr, b_ptr);

// After (Week 5):
matmul_tile(c_ptr, a_ptr, b_ptr);
```

**Deliverable:** Engine uses the optimized kernel when available.

---

### 4.5 Build System Updates (`CMakeLists.txt`)

**Task:** Add AVX2 compilation flags and new source files.

**Changes:**
```cmake
# Add AVX2/FMA flags for kernel files
set(KERNEL_SOURCES
    src/runtime/kernels/matmul_ref.cpp
    src/runtime/kernels/matmul_avx2.cpp
    src/runtime/kernels/dispatcher.cpp
)

# Compile kernels with AVX2 if available
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

if(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
    set_source_files_properties(
        src/runtime/kernels/matmul_avx2.cpp
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
    )
    add_definitions(-DHAS_AVX2)
endif()
```

**Deliverable:** Build system that enables AVX2 on supported platforms.

---

## 5) Roofline Model & Performance Analysis

### 5.1 What is the Roofline Model?

The roofline model plots achievable performance (GFLOP/s) against **arithmetic intensity** (FLOPs/byte).

```
Performance (GFLOP/s)
    ^
    |          ______________________ Peak Compute (56 GFLOP/s)
    |         /
    |        /  Memory-Bound Region
    |       /
    |      /   Compute-Bound Region
    |     /
    |    /
    +---/---------------------------------> Arithmetic Intensity (FLOP/byte)
       Ridge Point
```

- **Memory-bound:** Performance limited by memory bandwidth
- **Compute-bound:** Performance limited by CPU's FMA throughput
- **Ridge point:** The arithmetic intensity where compute and memory constraints meet

### 5.2 Our Kernel's Arithmetic Intensity

For a $32\times 32$ tile matmul:
- **FLOPs:** $2 \times 32^3 = 65{,}536$
- **Bytes loaded:** $3 \times 32 \times 32 \times 4 = 12{,}288$ (A + B + C)
- **Bytes stored:** $32 \times 32 \times 4 = 4{,}096$ (C)
- **Total memory traffic:** $12{,}288 + 4{,}096 = 16{,}384$ bytes

**Arithmetic Intensity:** $65{,}536 / 16{,}384 = 4.0$ FLOP/byte

For a typical CPU with:
- Peak compute: 56 GFLOP/s (AVX2 FMA)
- Memory bandwidth: 40 GB/s (DDR4-2400)
- Ridge point: $56 / 40 = 1.4$ FLOP/byte

**Our kernel (4.0 FLOP/byte) is compute-bound.** This is excellent—AVX2 optimization will directly translate to speedup.

### 5.3 Roofline Documentation

**Task:** Create `docs/roofline.md` explaining:
1. The roofline model concept
2. How to measure peak compute and memory bandwidth
3. Our kernel's arithmetic intensity calculation
4. Expected vs. achieved performance

**Deliverable:** `docs/roofline.md` with diagrams and analysis.

---

## 6) Benchmarking Suite

### 6.1 Isolated Kernel Benchmark (`benchmarks/matmul_32x32.py`)

**Task:** Micro-benchmark that measures kernel throughput.

**Methodology:**
1. Allocate aligned A, B, C tiles
2. Warm up (10 iterations)
3. Time 1000 iterations
4. Report: average time, throughput (GFLOP/s), speedup vs reference

**Output format:**
```
32x32 Matmul Kernel Benchmark
=============================
Reference kernel:  0.22 ms/1000 iters →   4.5 GFLOP/s
AVX2 kernel:       0.02 ms/1000 iters →  45.0 GFLOP/s
Speedup: 10.0x

Theoretical peak: 144 GFLOP/s (assuming 4.5 GHz, 2 FMA units)
Efficiency: 31.2%
```

### 6.2 End-to-End MLP Benchmark (`benchmarks/mlp_e2e.py`)

**Task:** Measure full pipeline performance.

**Methodology:**
1. Build MLP graph, compile schedule (once)
2. Execute schedule 100 times with different random inputs
3. Report: total time, breakdown (Python overhead, C++ execution)

**Deliverable:** Two benchmark scripts in `benchmarks/`.

---

## 7) Testing Strategy

### 7.1 Correctness Tests

**Test cases:**
1. **Identity test:** $C = 0$, $A = I$, $B = X$ → $C = X$
2. **Accumulation test:** $C = 1$, $A = I$, $B = I$ → $C = 1 + I$
3. **Random test:** Compare AVX2 output to reference output (within $\epsilon = 10^{-5}$)
4. **Numerical edge cases:** Large values, small values, negative values

**Location:** `tests/test_kernels.cpp` (C++) and `tests/test_kernels.py` (Python via pybind11)

### 7.2 Performance Regression Tests

**Task:** Add a CI check that AVX2 kernel is at least 2× faster than reference.

**Rationale:** Catches accidental regressions (e.g., vectorization disabled by a compiler flag change).

---

## 8) Deliverables Checklist

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | `src/runtime/kernels/matmul_avx2.cpp` — AVX2 kernel | ☐ |
| 2 | `src/runtime/kernels/dispatcher.cpp` — Kernel selection | ☐ |
| 3 | Update `include/mini_runtime/kernels.hpp` with new declarations | ☐ |
| 4 | Update `CMakeLists.txt` for AVX2 compilation | ☐ |
| 5 | Update `setup.py` for AVX2 compilation | ☐ |
| 6 | Update `engine.cpp` to use dispatched kernels | ☐ |
| 7 | `benchmarks/matmul_32x32.py` — Isolated kernel benchmark | ☐ |
| 8 | `benchmarks/mlp_e2e.py` — End-to-end benchmark | ☐ |
| 9 | `docs/roofline.md` — Performance analysis documentation | ☐ |
| 10 | `tests/test_kernels_avx2.cpp` — Correctness tests | ☐ |
| 11 | Update `agent.md` with Week 5 completion status | ☐ |

---

## 9) Timeline (Suggested)

| Day | Task |
|-----|------|
| Day 1 | Implement AVX2 kernel (basic structure, no unrolling) |
| Day 2 | Add k-loop unrolling, handle remainder rows |
| Day 3 | Implement dispatcher, update engine |
| Day 4 | Build system updates, verify compilation |
| Day 5 | Write benchmarks, collect initial numbers |
| Day 6 | Roofline documentation, performance analysis |
| Day 7 | Testing, polish, Week 5 report |

---

## 10) Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| AVX2 not available on dev machine | Cannot test optimized kernel | Use Docker with `-march=haswell` or cloud VM |
| Compiler auto-vectorizes reference kernel | Reduces apparent speedup | Use `-fno-tree-vectorize` for reference |
| Numerical differences between ref and AVX2 | False test failures | Use relative tolerance ($10^{-5}$), check for NaN/Inf |
| Register spilling in micro-kernel | Poor performance | Profile with `perf`, adjust micro-kernel size |

---

## 11) References

- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- Agner Fog's Optimization Manuals: https://www.agner.org/optimize/
- BLIS micro-kernel design: https://github.com/flame/blis
- Roofline Model paper: Williams et al., "Roofline: An Insightful Visual Performance Model"

---

## 12) Success Criteria

Week 5 is complete when:

1. ✅ AVX2 kernel produces identical results to reference (within tolerance)
2. ✅ AVX2 kernel is at least **4× faster** than reference in isolated benchmark
3. ✅ End-to-end MLP benchmark shows measurable speedup
4. ✅ Roofline analysis documents theoretical vs. achieved performance
5. ✅ All existing tests pass with the new kernel dispatcher
