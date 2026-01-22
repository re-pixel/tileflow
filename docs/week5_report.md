# Week 5 Report — SIMD Tiled Kernels & Performance Analysis (mini-compiler)

**Date:** 2026-01-22  
**Phase:** Phase 2 (Backend Optimization, C++)  
**Week 5 Goal:** Implement AVX2-optimized 32×32 matmul kernels, establish a roofline-based performance analysis framework, and create comprehensive benchmarking tools.

---

## 1) Executive Summary

Week 5 successfully transitioned the project from **correctness** to **performance**:

- **AVX2 Kernel:** Implemented a 6×16 micro-kernel design using 256-bit YMM registers, achieving **20× speedup** over the reference implementation.
- **Kernel Dispatcher:** Compile-time feature detection (`__AVX2__` + `__FMA__`) with auto-dispatch and explicit implementation selection for benchmarking.
- **Build System Updates:** Both CMake and setup.py now detect and enable AVX2+FMA optimizations automatically.
- **Benchmarking Suite:** Isolated kernel benchmark (`matmul_32x32.py`) and end-to-end pipeline benchmark (`mlp_e2e.py`).
- **Roofline Documentation:** Comprehensive analysis showing our 32×32 tiles are compute-bound (AI = 4.0 FLOP/byte >> ridge point).
- **Test Coverage:** C++ GTest suite and Python pytest suite covering correctness, numerical stability, and performance regression.

The AVX2 kernel achieves **33.6 GFLOP/s** (30% of theoretical peak) on an AMD Ryzen 7 7730U.

---

## 2) Key Implementation Details

### 2.1 AVX2 Kernel Design (`src/runtime/kernels/matmul_avx2.cpp`)

The optimized kernel uses a **6×16 micro-kernel** strategy designed around AVX2's 16 YMM registers:

| Register Purpose | Count | Description |
|------------------|-------|-------------|
| Accumulators | 12 | 6 rows × 2 vectors (8 floats each) |
| A broadcast | 1 | Reused for each row's `_mm256_broadcast_ss` |
| B loads | 2 | Two 8-float vectors per K iteration |
| Spare | 1 | Address computation |

**Key optimizations:**
- `_mm256_broadcast_ss` replicates A[i,k] across 8 lanes
- `_mm256_fmadd_ps` for fused multiply-add (2 FLOPs per element)
- K-loop unrolling by 4 for instruction-level parallelism
- 2-row epilogue handles 32 = 5×6 + 2 remainder

```cpp
// Inner loop structure (simplified)
for (uint32_t k = 0; k < TILE_DIM; k += 4) {
    for (int kk = 0; kk < 4; ++kk) {
        __m256 b0 = _mm256_loadu_ps(&B[k_idx * TILE_DIM + j_start]);
        __m256 b1 = _mm256_loadu_ps(&B[k_idx * TILE_DIM + j_start + 8]);
        
        __m256 a0 = _mm256_broadcast_ss(&A[(i_start + 0) * TILE_DIM + k_idx]);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        // ... repeat for all 6 rows
    }
}
```

### 2.2 Kernel Dispatcher (`src/runtime/kernels/dispatcher.cpp`)

Runtime dispatch based on compile-time feature detection:

```cpp
void matmul_tile(float* C, const float* A, const float* B) {
#if defined(__AVX2__) && defined(__FMA__)
    matmul_tile_avx2(C, A, B);
#else
    matmul_tile_ref(C, A, B);
#endif
}
```

**Explicit selection for benchmarking:**
```cpp
enum class KernelImpl { Reference, AVX2 };
void matmul_tile(float* C, const float* A, const float* B, KernelImpl impl);
```

**Utility functions:**
- `is_avx2_available()` — Returns true if compiled with AVX2+FMA
- `get_active_kernel_name()` — Returns "AVX2+FMA" or "Reference"

### 2.3 Build System Updates

**CMakeLists.txt:**
```cmake
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

if(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
    set_source_files_properties(
        src/runtime/kernels/matmul_avx2.cpp
        src/runtime/kernels/dispatcher.cpp
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
    )
endif()
```

**setup.py:**
- Runtime AVX2 support detection via test compilation
- Conditional source file inclusion
- Per-file compiler flags

### 2.4 pybind11 Extensions (`src/bindings/bindings.cpp`)

New bindings for benchmarking:

| Binding | Description |
|---------|-------------|
| `KernelImpl` | Enum with `Reference` and `AVX2` values |
| `is_avx2_available()` | Check compile-time AVX2 support |
| `get_active_kernel_name()` | Get active kernel name string |
| `matmul_tile_bench(C, A, B, impl?)` | Direct kernel call with optional impl selection |
| `relu_tile_bench(C, impl)` | Direct ReLU call with impl selection |

### 2.5 Roofline Analysis (`docs/roofline.md`)

Key findings for 32×32 tile matmul:

| Metric | Value |
|--------|-------|
| FLOPs per tile | 65,536 (2 × 32³) |
| Memory traffic | 16,384 bytes (A + B + C load + C store) |
| Arithmetic intensity | **4.0 FLOP/byte** |
| L1 ridge point | ~0.16 FLOP/byte |
| Classification | **Compute-bound** |

The high arithmetic intensity (4.0 >> 0.16) confirms SIMD optimization is highly effective for our tile size.

---

## 3) Deliverables Check

| Deliverable | Status | Location |
|-------------|--------|----------|
| **AVX2 Kernel** | ✅ Complete | `src/runtime/kernels/matmul_avx2.cpp` |
| **Kernel Dispatcher** | ✅ Complete | `src/runtime/kernels/dispatcher.cpp` |
| **Updated kernels.hpp** | ✅ Complete | `include/mini_runtime/kernels.hpp` |
| **CMakeLists.txt AVX2** | ✅ Complete | `CMakeLists.txt` |
| **setup.py AVX2** | ✅ Complete | `setup.py` |
| **Engine dispatcher use** | ✅ Complete | `src/runtime/engine/engine.cpp` |
| **Kernel benchmark** | ✅ Complete | `benchmarks/matmul_32x32.py` |
| **E2E benchmark** | ✅ Complete | `benchmarks/mlp_e2e.py` |
| **Roofline docs** | ✅ Complete | `docs/roofline.md` |
| **C++ tests** | ✅ Complete | `tests/test_kernels.cpp` |
| **Python tests** | ✅ Complete | `tests/test_kernels_avx2.py` |

---

## 4) Example Output

### 4.1 Kernel Benchmark (`python3 benchmarks/matmul_32x32.py`)

```text
============================================================
32x32 Tile Matmul Kernel Benchmark
============================================================

CPU: AMD Ryzen 7 7730U with Radeon Graphics
Current frequency: 1996 MHz
Theoretical peak: 112.0 GFLOP/s (assuming 3.5 GHz, 2 FMA units)

Active kernel: AVX2+FMA
AVX2 available: True

Verifying AVX2 kernel correctness...
  ✓ Results match (max relative diff: 0.00e+00)

Benchmarking Reference kernel (1000 iterations)...
  Time: 39.11 ms total, 39.11 µs/iter
  Throughput: 1.68 GFLOP/s
  Efficiency: 1.5% of theoretical peak

Benchmarking AVX2 kernel (1000 iterations)...
  Time: 1.95 ms total, 1.95 µs/iter
  Throughput: 33.62 GFLOP/s
  Efficiency: 30.0% of theoretical peak

============================================================
Summary
============================================================
Reference:    1.68 GFLOP/s
AVX2:        33.62 GFLOP/s
Speedup:    20.06x

Theoretical peak: 112.0 GFLOP/s
AVX2 efficiency:  30.0%

------------------------------------------------------------
Performance Analysis
------------------------------------------------------------
FLOPs per tile:        65,536
Memory traffic:        16,384 bytes
Arithmetic intensity:  4.00 FLOP/byte
Tile fits in L1:       Yes (4 KiB per tile, 12 KiB total)
Bound:                 Compute-bound (AI > 1.4 ridge point)
```

### 4.2 End-to-End Benchmark (`python3 benchmarks/mlp_e2e.py`)

```text
============================================================
End-to-End MLP Benchmark
============================================================

Active kernel: AVX2+FMA
AVX2 available: True
SRAM size: 256 KiB

------------------------------------------------------------
Compilation Phase
------------------------------------------------------------

Building graph...
  Original ops: 5

Running compiler passes...

  Compilation time breakdown:
    Fusion:      0.019 ms
    Tiling:      0.026 ms
    Lowering:    0.194 ms
    Scheduling:  1.111 ms
    ─────────────────────────
    Total:       1.351 ms

  Schedule statistics:
    Scheduled ops:    99
    Loads emitted:    39
    Loads eliminated: 49
    Peak SRAM:        52.0 KiB

------------------------------------------------------------
Execution Benchmark
------------------------------------------------------------

  Warmup iterations: 10
  Benchmark iterations: 100

  Execution time per iteration:
    Mean:    0.148 ms
    Std:     0.030 ms
    Min:     0.128 ms
    Max:     0.374 ms
    p50:     0.138 ms
    p99:     0.221 ms

------------------------------------------------------------
Performance Analysis
------------------------------------------------------------

  Total FLOPs per forward pass: 2,883,584
  Throughput: 19.50 GFLOP/s

============================================================
Summary
============================================================

  Pipeline:
    Compilation time:       1.35 ms (one-time)
    Execution time (avg):   0.15 ms/iter
    Effective throughput: 19.50 GFLOP/s

  Schedule efficiency:
    Load reuse: 49/88 (55.7%)
    Peak SRAM utilization: 20.3%
```

### 4.3 MLP Runtime Demo (with AVX2)

```text
======================================================================
Week 4 Demo — MLP Execution with C++ Runtime
======================================================================

...

======================================================================
Correctness Verification
======================================================================

  Output shape: (128, 32)
  Max absolute difference: 3.73e-08
  Mean absolute difference: 3.57e-09

  Tolerance check (rtol=0.0001, atol=1e-05):
    ✓ PASS

======================================================================
Summary
======================================================================

  Pipeline stages:
    1. Graph ops:      3
    2. Lowered uOps:   148
    3. Scheduled ops:  99
    4. C++ execution:  99 ops

  Memory efficiency:
    Load reuse:        49/88 (55.7%)
    Peak SRAM:         52.0 KiB / 256 KiB

  Correctness: ✓ VERIFIED
```

---

## 5) Test Coverage

### 5.1 Python Tests (`tests/test_kernels_avx2.py`)

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestKernelAvailability` | 3 | API availability checks |
| `TestReferenceKernel` | 4 | Identity, accumulation, numpy comparison |
| `TestAVX2Kernel` | 6 | Correctness vs reference, numerical stability |
| `TestAutoDispatch` | 1 | Auto-dispatched matches numpy |
| `TestReLU` | 4 | ReLU correctness for both implementations |
| `TestInputValidation` | 2 | Shape and dimension checks |
| `TestPerformance` | 1 | AVX2 > 2× faster than reference |

### 5.2 C++ Tests (`tests/test_kernels.cpp`)

| Test Fixture | Tests | Description |
|--------------|-------|-------------|
| `ReferenceKernelTest` | 6 | Identity, accumulation, zero matrix, ReLU |
| `AVX2KernelTest` | 8 | Matches reference, large/small values, fused ops |
| `DispatcherTest` | 4 | Auto and explicit dispatch, utility functions |
| `PerformanceTest` | 1 | AVX2 speedup > 2× |

All **62 project tests pass** (21 new kernel tests + 41 existing).

---

## 6) Architecture Diagram

```
                     Kernel Selection Flow
─────────────────────────────────────────────────────────────────────────

  Engine::dispatch(SchedExecute)
           │
           ▼
  ┌────────────────────────────────────────────────────────────────────┐
  │                     matmul_tile(C, A, B)                           │
  │                      (dispatcher.cpp)                              │
  │                                                                    │
  │   #if defined(__AVX2__) && defined(__FMA__)                       │
  │   ┌─────────────────────────────────────────────────────────────┐ │
  │   │                  matmul_tile_avx2()                          │ │
  │   │                                                              │ │
  │   │  ┌────────────────────────────────────────────────────────┐ │ │
  │   │  │              6×16 Micro-kernel                         │ │ │
  │   │  │                                                        │ │ │
  │   │  │  YMM0-11:  Accumulators (6 rows × 2 vectors)          │ │ │
  │   │  │  YMM12:    A broadcast                                 │ │ │
  │   │  │  YMM13-14: B loads                                     │ │ │
  │   │  │                                                        │ │ │
  │   │  │  for k in 0..32 step 4:                               │ │ │
  │   │  │    vbroadcastss + vfmadd231ps × 12                    │ │ │
  │   │  └────────────────────────────────────────────────────────┘ │ │
  │   └─────────────────────────────────────────────────────────────┘ │
  │   #else                                                           │
  │   ┌─────────────────────────────────────────────────────────────┐ │
  │   │                  matmul_tile_ref()                           │ │
  │   │                                                              │ │
  │   │  for i in 0..32:                                            │ │
  │   │    for j in 0..32:                                          │ │
  │   │      for k in 0..32:                                        │ │
  │   │        C[i,j] += A[i,k] * B[k,j]                            │ │
  │   └─────────────────────────────────────────────────────────────┘ │
  │   #endif                                                          │
  └────────────────────────────────────────────────────────────────────┘
```

---

## 7) Performance Analysis

### 7.1 Speedup Breakdown

| Metric | Reference | AVX2 | Improvement |
|--------|-----------|------|-------------|
| Time per tile | 39.1 µs | 1.95 µs | **20× faster** |
| Throughput | 1.68 GFLOP/s | 33.6 GFLOP/s | **20× higher** |
| Efficiency | 1.5% | 30.0% | **20× better** |

### 7.2 Why 30% Efficiency?

The 30% of theoretical peak is reasonable given:

1. **Instruction overhead:** Broadcast, load, and address computation
2. **K-loop structure:** Not fully pipelined across iterations
3. **Memory subsystem:** Even L1 access has non-zero latency
4. **Register pressure:** 16 YMM registers limits blocking options

Further optimization (AVX-512, panel packing) could improve this.

### 7.3 End-to-End Impact

| Metric | Value |
|--------|-------|
| MLP execution time | 0.15 ms/iteration |
| Effective throughput | 19.5 GFLOP/s |
| Compilation overhead | 1.35 ms (one-time) |

The lower E2E throughput (19.5 vs 33.6 GFLOP/s) is due to:
- Python↔C++ call overhead
- Schedule dispatch overhead
- Memory copy operations (LOAD/STORE)

---

## 8) File Summary

### New Files Created

```
src/runtime/kernels/
├── matmul_avx2.cpp      # AVX2 optimized kernel
└── dispatcher.cpp       # Kernel dispatch logic

benchmarks/
├── matmul_32x32.py      # Isolated kernel benchmark
└── mlp_e2e.py           # End-to-end benchmark

docs/
├── roofline.md          # Performance analysis documentation
└── week5_report.md      # This report

tests/
├── test_kernels.cpp     # C++ GTest suite
└── test_kernels_avx2.py # Python pytest suite
```

### Modified Files

| File | Changes |
|------|---------|
| `include/mini_runtime/kernels.hpp` | Added AVX2 declarations, dispatcher API, `KernelImpl` enum |
| `src/runtime/engine/engine.cpp` | Use dispatched `matmul_tile()` |
| `src/bindings/bindings.cpp` | Added benchmark functions, `KernelImpl` enum |
| `CMakeLists.txt` | AVX2 detection, per-file compile flags |
| `setup.py` | Runtime AVX2 detection, conditional sources |
| `agent.md` | Updated with Week 5 completion |

---

## 9) Look Ahead (Week 6)

With optimized kernels in place, the next phase is **Concurrency**:

- **Dual-Thread Runtime:** Separate data movement and compute threads
- **Circular Buffer:** Thread-safe tile pointer passing between threads
- **Producer-Consumer Model:** Overlap LOAD/STORE with COMPUTE
- **Synchronization:** Minimize lock contention for sustained throughput
