# Week 4 Report — C++ Runtime & Python Bridge (mini-compiler)

**Date:** 2026-01-20  
**Phase:** Phase 2 (Runtime Backend, C++)  
**Week 4 Goal:** Implement a C++ execution runtime with pybind11 bindings that consumes Week 3's scheduled operations and executes them on simulated hardware.

---

## 1) Executive Summary

Week 4 successfully implemented the **C++ Runtime Backend**:

- **pybind11 Bindings:** Complete Python↔C++ bridge exposing `Engine`, `SchedLoad`, `SchedExecute`, `SchedStore`, and configuration/stats classes.
- **SRAM Arena:** `std::aligned_alloc`-based memory arena simulating on-chip SRAM with 64-byte alignment and bounds validation.
- **Tensor Storage:** DRAM simulation with tile-aligned padding for efficient strided access to arbitrary-sized tensors.
- **Reference Kernel:** Naive triple-loop 32×32 matmul-accumulate kernel for correctness verification.
- **Schedule Executor:** C++ engine that dispatches schedule operations, handling tile loads/stores with proper striding and accumulator clearing.
- **Python Wrapper:** High-level `Runtime` class bridging the Python scheduler output to C++ execution.

The runtime correctly executes end-to-end MLP workloads with **max error < 4e-8** compared to NumPy reference.

---

## 2) Key Implementation Details

### 2.1 Build System

Extended the project with C++ compilation support:

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Standalone CMake build (optional) |
| `setup.py` | pybind11 extension build via `pip install -e .` |

Build requirements: C++17, pybind11 ≥2.11, numpy ≥1.24.

### 2.2 C++ Headers (`include/mini_runtime/`)

| Header | Contents |
|--------|----------|
| `constants.hpp` | `TILE_DIM=32`, `TILE_BYTES=4096`, `ALIGNMENT=64` |
| `arena.hpp` | `SRAMArena` class declaration |
| `tensors.hpp` | `TensorStorage` class for DRAM with padding |
| `schedule.hpp` | `SchedLoad`, `SchedExecute`, `SchedStore` structs |
| `kernels.hpp` | `matmul_tile_ref()`, `relu_tile_inplace()` declarations |
| `engine.hpp` | `Engine` class with `Config`, `Stats`, `execute()` |

### 2.3 SRAM Arena (`src/runtime/memory/arena.cpp`)

A simple aligned memory buffer simulating on-chip SRAM:

```cpp
class SRAMArena {
    std::unique_ptr<float[], Deleter> data_;  // 64-byte aligned
    size_t size_bytes_;
public:
    float* ptr(uint32_t offset);              // Get pointer at offset
    void validate_range(uint32_t offset, uint32_t bytes);  // Bounds check
    void clear();                             // Zero entire arena
};
```

### 2.4 Tensor Storage (`src/runtime/memory/tensors.cpp`)

DRAM simulation with tile-boundary padding:

| Feature | Description |
|---------|-------------|
| **Tile-aligned dimensions** | Rows/cols padded to multiples of 32 |
| **Row-major storage** | Standard C layout with padding columns |
| **`tile_ptr(id, row, col)`** | Direct pointer to tile start |
| **`get_stride(id)`** | Returns padded row width for strided copy |

This allows efficient tile extraction even when tensor dimensions aren't tile-aligned.

### 2.5 Reference Kernel (`src/runtime/kernels/matmul_ref.cpp`)

Correctness-first implementation:

```cpp
void matmul_tile_ref(float* C, const float* A, const float* B) {
    // C[32x32] += A[32x32] @ B[32x32]
    for (uint32_t i = 0; i < TILE_DIM; ++i) {
        for (uint32_t j = 0; j < TILE_DIM; ++j) {
            float sum = C[i * TILE_DIM + j];
            for (uint32_t k = 0; k < TILE_DIM; ++k) {
                sum += A[i * TILE_DIM + k] * B[k * TILE_DIM + j];
            }
            C[i * TILE_DIM + j] = sum;
        }
    }
}
```

ReLU is applied in-place during store operations when requested.

### 2.6 Schedule Executor (`src/runtime/engine/engine.cpp`)

The core execution engine:

| Method | Operation |
|--------|-----------|
| `dispatch(SchedLoad)` | Copy tile from TensorStorage to SRAM (row-by-row for striding) |
| `dispatch(SchedExecute)` | Clear accumulator if k=0, then call `matmul_tile_ref()` |
| `dispatch(SchedStore)` | Apply optional ReLU, copy tile from SRAM to TensorStorage |

**Critical fix:** The accumulator must be cleared when `k == 0` (first partial product for an output tile). Without this, reused accumulator slots contain stale data from previous tiles.

```cpp
void Engine::dispatch(const SchedExecute& op) {
    // ...
    if (op.k == 0) {
        std::memset(C, 0, TILE_BYTES);  // Clear stale accumulator
    }
    matmul_tile_ref(C, A, B);
    // ...
}
```

### 2.7 pybind11 Bindings (`src/bindings/bindings.cpp`)

Complete Python interface:

| Binding | Python Type |
|---------|-------------|
| `mini_runtime.Engine` | Main execution engine |
| `mini_runtime.EngineConfig` | SRAM size configuration |
| `mini_runtime.EngineStats` | Execution statistics |
| `mini_runtime.SchedLoad` | Load operation |
| `mini_runtime.SchedExecute` | Execute operation |
| `mini_runtime.SchedStore` | Store operation |
| `mini_runtime.TILE_DIM` | Constant (32) |
| `mini_runtime.TILE_BYTES` | Constant (4096) |

NumPy arrays are passed directly via pybind11's buffer protocol.

### 2.8 Python Wrapper (`src/compiler/runtime.py`)

High-level interface for end-to-end execution:

```python
class Runtime:
    def register_graph_tensors(self, graph: Graph) -> None
    def set_tensor(self, name: str, data: np.ndarray) -> None
    def execute(self, schedule: list[SchedOp]) -> None
    def get_tensor(self, name: str) -> np.ndarray
    
    @property
    def stats(self) -> EngineStats
```

Handles schedule conversion from Python `SchedOp` to C++ format automatically.

---

## 3) Deliverables Check

| Deliverable | Status | Location |
|-------------|--------|----------|
| **CMake/setup.py build** | ✅ Complete | `CMakeLists.txt`, `setup.py` |
| **C++ headers** | ✅ Complete | `include/mini_runtime/*.hpp` |
| **SRAM Arena** | ✅ Complete | `src/runtime/memory/arena.cpp` |
| **Tensor Storage** | ✅ Complete | `src/runtime/memory/tensors.cpp` |
| **Reference Kernel** | ✅ Complete | `src/runtime/kernels/matmul_ref.cpp` |
| **Schedule Executor** | ✅ Complete | `src/runtime/engine/engine.cpp` |
| **pybind11 Bindings** | ✅ Complete | `src/bindings/bindings.cpp` |
| **Python Wrapper** | ✅ Complete | `src/compiler/runtime.py` |
| **Tests** | ✅ Complete (13 tests) | `tests/test_runtime.py` |
| **Demo Script** | ✅ Complete | `examples/mlp_runtime.py` |

---

## 4) Example Output

Running `python examples/mlp_runtime.py`:

```text
======================================================================
Week 4 Demo — MLP Execution with C++ Runtime
======================================================================

[1] Building MLP graph...
    Original graph: 5 ops
    Tensors: 9

[2] Running Fusion Pass...
    After fusion: 3 ops
      - fused_matmul1_relu1 (FusedMatMulReLU)
      - fused_matmul2_relu2 (FusedMatMulReLU)
      - matmul3 (MatMul)

[3] Running Tiling Pass...
    Tiling validated: True

[4] Running Lowering Pass...
    Generated 148 micro-ops (uOps)

[5] Running Scheduling Pass...
    SRAM size: 256 KiB

======================================================================
Schedule Statistics
======================================================================
  Total scheduled ops: 99
  Loads emitted:       39
  Loads eliminated:    49 (55.7% reuse)
  Peak SRAM usage:     53,248 bytes (52.0 KiB)
  Final live bytes:    0

======================================================================
C++ Runtime Execution
======================================================================

  Created runtime with 256 KiB SRAM

  Registering tensors...
    - x: (128, 128)
    - w1: (128, 64)
    - w2: (64, 32)
    - w3: (32, 32)

  Setting input data...

  Executing schedule (99 ops)...

  Execution complete!
    LOADs executed:  39
    EXECs executed:  44
    STOREs executed: 16

======================================================================
Correctness Verification
======================================================================

  Output shape: (128, 32)
  Max absolute difference: 3.73e-08
  Mean absolute difference: 5.06e-09

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

The `tests/test_runtime.py` suite validates:

| Test Category | Tests |
|---------------|-------|
| **Engine Basics** | Tensor registration, roundtrip, padded tensor handling |
| **Single Tile MatMul** | Identity matrix, random values, accumulation semantics |
| **ReLU** | Zeros negatives, preserves positives |
| **Multi-Tile** | 64×64 matmul (2×2 tiles) |
| **End-to-End MLP** | Single layer, 3-layer MLP with fusion |
| **Statistics** | Op counting, stats reset |

All **41 project tests pass** (13 runtime + 28 existing).

---

## 6) Architecture Diagram

```
Week 3 Output                      Week 4 Runtime
─────────────────────────────────────────────────────────────────────────

  ┌──────────────────┐              ┌─────────────────────────────────┐
  │   SchedOps       │              │         Python Wrapper          │
  │   (physical)     │              │       (runtime.py)              │
  │                  │              │                                 │
  │ SchedLoad        │ ────────────▶│  register_graph_tensors()       │
  │ SchedExecute     │              │  set_tensor() / get_tensor()    │
  │ SchedStore       │              │  execute(schedule)              │
  └──────────────────┘              └────────────┬────────────────────┘
                                                 │
                                                 │ pybind11
                                                 ▼
                                    ┌─────────────────────────────────┐
                                    │        C++ Engine               │
                                    │                                 │
                                    │  ┌───────────┐  ┌────────────┐  │
                                    │  │ SRAMArena │  │TensorStore │  │
                                    │  │ (on-chip) │  │  (DRAM)    │  │
                                    │  │           │  │            │  │
                                    │  │  64KB-    │  │  Padded    │  │
                                    │  │  256KB    │  │  Storage   │  │
                                    │  └─────┬─────┘  └─────┬──────┘  │
                                    │        │              │         │
                                    │        └──────┬───────┘         │
                                    │               │                 │
                                    │        ┌──────▼──────┐          │
                                    │        │  Dispatcher │          │
                                    │        │             │          │
                                    │        │ LOAD: DRAM  │          │
                                    │        │   → SRAM    │          │
                                    │        │             │          │
                                    │        │ EXEC: call  │          │
                                    │        │ matmul_ref  │          │
                                    │        │             │          │
                                    │        │ STORE: SRAM │          │
                                    │        │   → DRAM    │          │
                                    │        └─────────────┘          │
                                    └─────────────────────────────────┘
```

---

## 7) Debugging Notes

### 7.1 Accumulator Clearing Bug

**Symptom:** Multi-tile matmul produced correct results for tile (0,0) but wrong results for subsequent tiles, with errors accumulating.

**Root Cause:** The scheduler reuses accumulator slots after storing completed tiles. The SRAM is only cleared once at execution start, so when an accumulator address is reused, it still contains the previous tile's result.

**Fix:** Clear the accumulator to zero when `k == 0` (first partial product for a new output tile):

```cpp
if (op.k == 0) {
    std::memset(C, 0, TILE_BYTES);
}
```

### 7.2 Strided Tile Copy

**Symptom:** Multi-tile tensors had data corruption in non-(0,0) tiles.

**Root Cause:** Tensor storage uses row-major layout with padding. A 128×128 tensor stored in a 128×128 padded buffer has stride=128, but tiles are stored contiguously in SRAM (stride=32).

**Fix:** Copy tiles row-by-row instead of using a single `memcpy`:

```cpp
for (uint32_t row = 0; row < TILE_DIM; ++row) {
    std::memcpy(dst + row * TILE_DIM,
                src + row * src_stride,
                TILE_DIM * sizeof(float));
}
```

---

## 8) Performance Metrics

| Metric | Value |
|--------|-------|
| **Build time** | ~3 seconds (incremental) |
| **Test suite** | 41 tests in 0.17s |
| **MLP execution** | 99 scheduled ops |
| **Load reuse** | 55.7% (49/88 eliminated) |
| **Peak SRAM** | 52 KiB (20% of 256 KiB budget) |
| **Numerical error** | < 4e-8 vs NumPy |

---

## 9) File Summary

### New Files Created

```
include/mini_runtime/
├── arena.hpp
├── constants.hpp
├── engine.hpp
├── kernels.hpp
├── schedule.hpp
└── tensors.hpp

src/runtime/
├── engine/
│   └── engine.cpp
├── kernels/
│   └── matmul_ref.cpp
└── memory/
    ├── arena.cpp
    └── tensors.cpp

src/bindings/
└── bindings.cpp

src/compiler/
└── runtime.py

tests/
└── test_runtime.py

examples/
└── mlp_runtime.py

CMakeLists.txt
docs/week4_plan.md
docs/week4_report.md
```

### Modified Files

| File | Changes |
|------|---------|
| `setup.py` | Added pybind11 extension build |
| `requirements.txt` | Added pybind11 dependency |
| `.gitignore` | Added `*.egg-info/`, `dist/` |

---

## 10) Look Ahead (Week 5)

With the runtime executing correctly, the next phase is **Performance Optimization**:

- **AVX2/AVX-512 Kernels:** Vectorized 32×32 matmul for 4-8× speedup
- **Cache-Aware Tiling:** Optimize tile traversal order for L1/L2 cache
- **Prefetching:** Software prefetch hints for tile loads
- **Benchmarking:** Compare reference vs optimized kernels

