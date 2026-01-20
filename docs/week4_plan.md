# Week 4 Plan — C++ Runtime & Python Bridge

**Date:** 2026-01-20  
**Phase:** Phase 2 (Backend Execution Engine, C++)  
**Week 4 Goal:** Implement the C++ runtime that consumes the scheduled operations emitted by Week 3's scheduler, along with pybind11 bindings to bridge Python and C++.

---

## 1) Executive Summary

Week 4 transitions from the Python-based compiler frontend to the C++ backend that will actually *execute* the compiled schedules. This week establishes the critical Python↔C++ bridge and the foundational runtime infrastructure:

| Component | Description |
|-----------|-------------|
| **pybind11 Bindings** | Expose C++ runtime to Python for end-to-end execution |
| **Memory Arena** | C++ SRAM simulation using `std::aligned_alloc` |
| **Schedule Executor** | Dispatches `SchedLoad`, `SchedExecute`, `SchedStore` operations |
| **Reference Kernel** | Correctness-first $32\times 32$ matmul implementation |

The key insight is that the Python scheduler emits a **fully static schedule** with concrete SRAM addresses. The C++ runtime is a relatively "dumb" executor — it doesn't make allocation decisions, just follows the script.

---

## 2) Design Decisions & Rationale

### 2.1 Schedule Serialization Format

**Decision:** Use a simple struct-based binary format over JSON.

**Rationale:**
- The schedule is already a flat list of typed operations with fixed fields
- Binary format avoids parsing overhead and is trivial to serialize/deserialize
- pybind11 can pass a `std::vector<SchedOp>` directly if we define the struct properly
- For debugging, we can add an optional JSON dump on the Python side

**Format Structure:**
```cpp
enum class SchedOpType : uint8_t {
    Load    = 0,
    Execute = 1,
    Store   = 2
};

struct SchedOpHeader {
    SchedOpType type;
    uint8_t     padding[3];  // Align to 4 bytes
};

struct SchedLoadOp {
    uint32_t tensor_id;   // Index into tensor name table
    uint32_t coord[2];    // Tile coordinates (e.g., m, k)
    uint32_t dst_addr;    // SRAM address
    uint32_t bytes;       // Always TILE_BYTES (4096)
    int32_t  buffer;      // -1 if not double-buffered
};

struct SchedExecuteOp {
    uint32_t m, n, k;     // Tile indices
    uint32_t a_addr;      // SRAM address of A tile
    uint32_t b_addr;      // SRAM address of B tile
    uint32_t acc_addr;    // SRAM address of accumulator
    int32_t  buffer;      // -1 if not double-buffered
};

struct SchedStoreOp {
    uint32_t tensor_id;   // Index into tensor name table
    uint32_t coord[2];    // Tile coordinates (m, n)
    uint32_t src_addr;    // SRAM address
    uint32_t bytes;       // Always TILE_BYTES (4096)
    uint8_t  activation;  // 0=None, 1=ReLU
    uint8_t  padding[3];
};
```

### 2.2 Memory Architecture

**Decision:** Implement a two-level memory model: SRAM (arena) + DRAM (backing tensors).

**Rationale:**
- Matches the mental model of an accelerator with on-chip SRAM and off-chip DRAM
- The scheduler's addresses refer to SRAM; LOAD/STORE copy to/from DRAM
- Clearly separates "fast" (SRAM) from "slow" (DRAM) paths

**SRAM Arena Design:**
```cpp
class SRAMArena {
public:
    explicit SRAMArena(size_t total_bytes);
    
    // Direct access by pre-computed address (no allocation logic!)
    float* ptr(uint32_t addr);
    const float* ptr(uint32_t addr) const;
    
    // Bounds checking (debug builds)
    void validate_addr(uint32_t addr, size_t bytes) const;
    
private:
    std::unique_ptr<float[], AlignedDeleter> buffer_;
    size_t total_bytes_;
};
```

**Key Insight:** The C++ arena does *not* implement an allocator. The Python scheduler already computed all addresses. The C++ side just provides a raw memory buffer and validates bounds.

**DRAM (Tensor Storage) Design:**
```cpp
class TensorStorage {
public:
    // Register a tensor (called during setup)
    void register_tensor(const std::string& name, 
                         std::array<size_t, 2> shape);
    
    // Get pointer to a specific tile within a tensor
    float* tile_ptr(uint32_t tensor_id, 
                    uint32_t tile_row, uint32_t tile_col);
    
    // Load input data / retrieve output data
    void set_tensor_data(const std::string& name, const float* data);
    void get_tensor_data(const std::string& name, float* data) const;
    
private:
    struct TensorInfo {
        std::string name;
        std::array<size_t, 2> shape;
        std::unique_ptr<float[], AlignedDeleter> data;
    };
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, uint32_t> name_to_id_;
};
```

### 2.3 Schedule Executor Architecture

**Decision:** Single-threaded, synchronous executor for Week 4. Concurrency added in Week 6.

**Rationale:**
- Establishes correctness baseline before adding complexity
- Easier to debug and profile
- Matches the "reference-first" development philosophy from agent.md

**Executor Interface:**
```cpp
class ScheduleExecutor {
public:
    struct Config {
        size_t sram_bytes;
        bool   trace_enabled;  // Emit execution trace
    };
    
    explicit ScheduleExecutor(const Config& config);
    
    // Register tensors before execution
    void register_tensor(const std::string& name,
                         std::array<size_t, 2> shape);
    
    // Set input/parameter data
    void set_tensor(const std::string& name, const float* data);
    
    // Execute the full schedule
    void execute(const std::vector<SchedOp>& schedule);
    
    // Retrieve output data
    void get_tensor(const std::string& name, float* data) const;
    
private:
    SRAMArena sram_;
    TensorStorage tensors_;
    Kernel* kernel_;  // Reference kernel (Week 4) or AVX2 (Week 5)
    
    void dispatch_load(const SchedLoadOp& op);
    void dispatch_execute(const SchedExecuteOp& op);
    void dispatch_store(const SchedStoreOp& op);
};
```

### 2.4 Reference Kernel Design

**Decision:** Implement a naive triple-loop $32\times 32$ matmul as the correctness baseline.

**Rationale:**
- Per agent.md: "Always implement a reference (correct, slow) version before optimizing"
- The reference kernel is used to validate AVX2/AVX-512 kernels in Week 5
- Simple enough to be obviously correct

**Reference Kernel Specification:**
```cpp
// Computes: C[0:32, 0:32] += A[0:32, 0:32] @ B[0:32, 0:32]
// All pointers must be 64-byte aligned
// Dimensions are fixed at 32x32 (TILE_SIZE)
void matmul_32x32_ref(
    float*       C,      // [32x32] accumulator (read-modify-write)
    const float* A,      // [32x32] left operand
    const float* B,      // [32x32] right operand
);

// Optional: fused matmul + ReLU for SchedStore with activation
void relu_inplace_32x32(float* C);
```

**Implementation Notes:**
- The kernel performs an **accumulation** (C += A @ B), matching `SchedExecute` semantics
- The first `SchedExecute` for a given (m,n) pair operates on a zero-initialized accumulator
- Zero-initialization happens implicitly: SRAM arena is zero-filled on allocation
- ReLU is applied during `SchedStore`, not during `SchedExecute`

### 2.5 pybind11 Binding Strategy

**Decision:** Expose a high-level `Runtime` class that encapsulates executor details.

**Rationale:**
- Python side doesn't need to know about internal C++ structures
- Single entry point simplifies the API
- Allows flexible internal refactoring without breaking Python interface

**Python API (from pybind11):**
```python
import mini_runtime

# Create runtime with SRAM size
rt = mini_runtime.Runtime(sram_bytes=256 * 1024)

# Register tensors from graph
rt.register_tensor("x", (128, 128))
rt.register_tensor("w1", (128, 64))
rt.register_tensor("h1", (128, 64))

# Set input/parameter data
rt.set_tensor("x", x_data)   # x_data is numpy array
rt.set_tensor("w1", w1_data)

# Execute schedule (serialized from Python)
rt.execute(schedule_bytes)  # Or rt.execute_ops(schedule_list)

# Get output
output = rt.get_tensor("h1")
```

**Alternative Considered:** Pass schedule ops one-by-one via pybind11. Rejected because:
- High overhead per op (Python→C++ call overhead)
- The schedule is static; passing it as a batch is cleaner

---

## 3) Implementation Plan (Ordered Tasks)

### Phase 3.1: Build System Setup

**Task 3.1.1:** Configure CMakeLists.txt for pybind11

```cmake
cmake_minimum_required(VERSION 3.15)
project(mini_runtime)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
find_package(pybind11 REQUIRED)

# Runtime library (static)
add_library(mini_runtime_core STATIC
    src/runtime/engine/engine.cpp
    src/runtime/engine/dispatcher.cpp
    src/runtime/memory/arena.cpp
    src/runtime/memory/tensors.cpp
    src/runtime/kernels/matmul_ref.cpp
)

target_include_directories(mini_runtime_core PUBLIC include)
target_compile_options(mini_runtime_core PRIVATE -Wall -Wextra -O2)

# pybind11 module
pybind11_add_module(mini_runtime src/bindings/bindings.cpp)
target_link_libraries(mini_runtime PRIVATE mini_runtime_core)
```

**Task 3.1.2:** Update requirements.txt and setup.py

Add pybind11 to requirements.txt:
```
pytest>=7
pybind11>=2.11
numpy>=1.24
```

Update setup.py for C++ extension:
```python
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "mini_runtime",
        sources=[
            "src/bindings/bindings.cpp",
            "src/runtime/engine/engine.cpp",
            # ... other sources
        ],
        include_dirs=["include"],
        cxx_std=17,
    ),
]

setup(
    # ... existing config ...
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
```

**Deliverable:** `pip install -e .` builds the C++ extension.

---

### Phase 3.2: Memory Components

**Task 3.2.1:** Implement SRAM Arena (`src/runtime/memory/arena.cpp`)

```cpp
// include/mini_runtime/arena.hpp
#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace mini_runtime {

constexpr size_t TILE_SIZE = 32;
constexpr size_t TILE_BYTES = TILE_SIZE * TILE_SIZE * sizeof(float);  // 4096
constexpr size_t ALIGNMENT = 64;  // Cache line

class SRAMArena {
public:
    explicit SRAMArena(size_t total_bytes);
    
    float* ptr(uint32_t addr);
    const float* ptr(uint32_t addr) const;
    
    size_t total_bytes() const { return total_bytes_; }
    
    // Zero-fill (called before execution)
    void clear();
    
private:
    size_t total_bytes_;
    std::unique_ptr<float[], decltype(&std::free)> buffer_;
};

}  // namespace mini_runtime
```

**Implementation Details:**
- Use `std::aligned_alloc(64, size)` for allocation
- Custom deleter wraps `std::free`
- `clear()` uses `std::memset` to zero-fill

**Task 3.2.2:** Implement Tensor Storage (`src/runtime/memory/tensors.cpp`)

```cpp
// include/mini_runtime/tensors.hpp
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <array>

namespace mini_runtime {

class TensorStorage {
public:
    uint32_t register_tensor(const std::string& name, 
                             size_t rows, size_t cols);
    
    float* tile_ptr(uint32_t tensor_id, uint32_t tile_row, uint32_t tile_col);
    const float* tile_ptr(uint32_t tensor_id, uint32_t tile_row, uint32_t tile_col) const;
    
    void set_data(const std::string& name, const float* data, size_t size);
    void get_data(const std::string& name, float* data, size_t size) const;
    
    uint32_t name_to_id(const std::string& name) const;
    
private:
    struct TensorInfo {
        std::string name;
        size_t rows, cols;
        size_t tile_rows, tile_cols;  // Padded to tile boundary
        std::unique_ptr<float[], decltype(&std::free)> data;
    };
    
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, uint32_t> name_to_id_;
};

}  // namespace mini_runtime
```

**Implementation Details:**
- Tensors are stored in row-major order with padding to tile boundaries
- `tile_ptr()` computes: `base + tile_row * TILE_SIZE * padded_cols + tile_col * TILE_SIZE`
- Padding ensures all tile accesses are in-bounds

---

### Phase 3.3: Reference Kernel

**Task 3.3.1:** Implement Reference Matmul (`src/runtime/kernels/matmul_ref.cpp`)

```cpp
// include/mini_runtime/kernels.hpp
#pragma once
#include <cstdint>

namespace mini_runtime {

constexpr uint32_t TILE_DIM = 32;

// C[32x32] += A[32x32] @ B[32x32]
void matmul_tile_ref(float* C, const float* A, const float* B);

// Apply ReLU in-place: x = max(0, x)
void relu_tile_inplace(float* C);

// Combined for fused ops (future optimization)
void matmul_tile_relu_ref(float* C, const float* A, const float* B);

}  // namespace mini_runtime
```

**Reference Implementation:**
```cpp
void matmul_tile_ref(float* C, const float* A, const float* B) {
    for (uint32_t i = 0; i < TILE_DIM; ++i) {
        for (uint32_t j = 0; j < TILE_DIM; ++j) {
            float sum = C[i * TILE_DIM + j];  // Accumulate
            for (uint32_t k = 0; k < TILE_DIM; ++k) {
                sum += A[i * TILE_DIM + k] * B[k * TILE_DIM + j];
            }
            C[i * TILE_DIM + j] = sum;
        }
    }
}

void relu_tile_inplace(float* C) {
    for (uint32_t i = 0; i < TILE_DIM * TILE_DIM; ++i) {
        if (C[i] < 0.0f) C[i] = 0.0f;
    }
}
```

**Correctness Tests:**
- Identity matrix multiplication
- Known small matrices
- Accumulation semantics (C starts non-zero)
- Comparison with NumPy reference

---

### Phase 3.4: Schedule Executor

**Task 3.4.1:** Define Schedule Op Structures (`include/mini_runtime/schedule.hpp`)

```cpp
#pragma once
#include <cstdint>
#include <variant>
#include <vector>

namespace mini_runtime {

struct SchedLoad {
    uint32_t tensor_id;
    uint32_t tile_row;
    uint32_t tile_col;
    uint32_t dst_addr;
    uint32_t bytes;
    int32_t  buffer;  // -1 if unused
};

struct SchedExecute {
    uint32_t m, n, k;
    uint32_t a_addr;
    uint32_t b_addr;
    uint32_t acc_addr;
    int32_t  buffer;
};

struct SchedStore {
    uint32_t tensor_id;
    uint32_t tile_row;
    uint32_t tile_col;
    uint32_t src_addr;
    uint32_t bytes;
    bool     apply_relu;
};

using SchedOp = std::variant<SchedLoad, SchedExecute, SchedStore>;

}  // namespace mini_runtime
```

**Task 3.4.2:** Implement Executor (`src/runtime/engine/engine.cpp`)

```cpp
// include/mini_runtime/engine.hpp
#pragma once
#include "arena.hpp"
#include "tensors.hpp"
#include "schedule.hpp"
#include <vector>

namespace mini_runtime {

class Engine {
public:
    struct Config {
        size_t sram_bytes = 256 * 1024;  // 256 KiB default
        bool   trace = false;
    };
    
    explicit Engine(const Config& config);
    
    // Setup
    uint32_t register_tensor(const std::string& name, size_t rows, size_t cols);
    void set_tensor(const std::string& name, const float* data, size_t size);
    void get_tensor(const std::string& name, float* data, size_t size) const;
    
    // Execution
    void execute(const std::vector<SchedOp>& schedule);
    
    // Stats (for testing/debugging)
    struct Stats {
        uint64_t loads = 0;
        uint64_t executes = 0;
        uint64_t stores = 0;
    };
    Stats stats() const { return stats_; }
    
private:
    Config config_;
    SRAMArena sram_;
    TensorStorage tensors_;
    Stats stats_;
    
    void dispatch(const SchedLoad& op);
    void dispatch(const SchedExecute& op);
    void dispatch(const SchedStore& op);
};

}  // namespace mini_runtime
```

**Dispatch Implementation:**

```cpp
void Engine::dispatch(const SchedLoad& op) {
    // Copy tile from DRAM (tensor storage) to SRAM
    const float* src = tensors_.tile_ptr(op.tensor_id, op.tile_row, op.tile_col);
    float* dst = sram_.ptr(op.dst_addr);
    std::memcpy(dst, src, op.bytes);
    stats_.loads++;
}

void Engine::dispatch(const SchedExecute& op) {
    // C += A @ B
    float* C = sram_.ptr(op.acc_addr);
    const float* A = sram_.ptr(op.a_addr);
    const float* B = sram_.ptr(op.b_addr);
    matmul_tile_ref(C, A, B);
    stats_.executes++;
}

void Engine::dispatch(const SchedStore& op) {
    // Copy tile from SRAM to DRAM, optionally apply ReLU
    float* src = sram_.ptr(op.src_addr);
    if (op.apply_relu) {
        relu_tile_inplace(src);
    }
    float* dst = tensors_.tile_ptr(op.tensor_id, op.tile_row, op.tile_col);
    std::memcpy(dst, src, op.bytes);
    stats_.stores++;
}
```

---

### Phase 3.5: pybind11 Bindings

**Task 3.5.1:** Implement Bindings (`src/bindings/bindings.cpp`)

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "mini_runtime/engine.hpp"

namespace py = pybind11;
using namespace mini_runtime;

PYBIND11_MODULE(mini_runtime, m) {
    m.doc() = "Mini-compiler C++ runtime";
    
    // Engine configuration
    py::class_<Engine::Config>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("sram_bytes", &Engine::Config::sram_bytes)
        .def_readwrite("trace", &Engine::Config::trace);
    
    // Engine stats
    py::class_<Engine::Stats>(m, "EngineStats")
        .def_readonly("loads", &Engine::Stats::loads)
        .def_readonly("executes", &Engine::Stats::executes)
        .def_readonly("stores", &Engine::Stats::stores);
    
    // Main engine class
    py::class_<Engine>(m, "Engine")
        .def(py::init<const Engine::Config&>())
        .def("register_tensor", &Engine::register_tensor)
        .def("set_tensor", [](Engine& e, const std::string& name, 
                               py::array_t<float> arr) {
            auto buf = arr.request();
            e.set_tensor(name, static_cast<const float*>(buf.ptr), buf.size);
        })
        .def("get_tensor", [](const Engine& e, const std::string& name,
                               py::array_t<float> arr) {
            auto buf = arr.request();
            e.get_tensor(name, static_cast<float*>(buf.ptr), buf.size);
        })
        .def("execute", &Engine::execute)
        .def("stats", &Engine::stats);
    
    // Schedule op types (for constructing schedules from Python)
    py::class_<SchedLoad>(m, "SchedLoad")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int32_t>(),
             py::arg("tensor_id"), py::arg("tile_row"), py::arg("tile_col"),
             py::arg("dst_addr"), py::arg("bytes"), py::arg("buffer") = -1);
    
    py::class_<SchedExecute>(m, "SchedExecute")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int32_t>(),
             py::arg("m"), py::arg("n"), py::arg("k"),
             py::arg("a_addr"), py::arg("b_addr"), py::arg("acc_addr"),
             py::arg("buffer") = -1);
    
    py::class_<SchedStore>(m, "SchedStore")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, bool>(),
             py::arg("tensor_id"), py::arg("tile_row"), py::arg("tile_col"),
             py::arg("src_addr"), py::arg("bytes"), py::arg("apply_relu") = false);
}
```

**Task 3.5.2:** Add Python Wrapper (`src/compiler/runtime.py`)

A higher-level Python wrapper that converts the Python scheduler output into the C++ format:

```python
"""Python wrapper for C++ runtime."""

from typing import Sequence
import numpy as np

from compiler.scheduler.ops import SchedLoad, SchedStore, SchedExecute, SchedOp
import mini_runtime as _rt


class Runtime:
    """High-level Python interface to the C++ execution engine."""
    
    def __init__(self, sram_bytes: int = 256 * 1024):
        config = _rt.EngineConfig()
        config.sram_bytes = sram_bytes
        self._engine = _rt.Engine(config)
        self._tensor_ids: dict[str, int] = {}
    
    def register_tensor(self, name: str, shape: tuple[int, int]) -> None:
        tid = self._engine.register_tensor(name, shape[0], shape[1])
        self._tensor_ids[name] = tid
    
    def set_tensor(self, name: str, data: np.ndarray) -> None:
        self._engine.set_tensor(name, data.astype(np.float32).ravel())
    
    def get_tensor(self, name: str, shape: tuple[int, int]) -> np.ndarray:
        out = np.zeros(shape, dtype=np.float32)
        self._engine.get_tensor(name, out)
        return out
    
    def execute(self, schedule: Sequence[SchedOp]) -> None:
        cpp_ops = self._convert_schedule(schedule)
        self._engine.execute(cpp_ops)
    
    def _convert_schedule(self, schedule: Sequence[SchedOp]) -> list:
        """Convert Python schedule ops to C++ format."""
        cpp_ops = []
        for op in schedule:
            if isinstance(op, SchedLoad):
                cpp_ops.append(_rt.SchedLoad(
                    self._tensor_ids[op.tensor],
                    op.coord[0], op.coord[1],
                    op.dst_addr, op.bytes,
                    op.buffer if op.buffer is not None else -1
                ))
            elif isinstance(op, SchedExecute):
                cpp_ops.append(_rt.SchedExecute(
                    op.m, op.n, op.k,
                    op.a_addr, op.b_addr, op.acc_addr,
                    op.buffer if op.buffer is not None else -1
                ))
            elif isinstance(op, SchedStore):
                cpp_ops.append(_rt.SchedStore(
                    self._tensor_ids[op.tensor],
                    op.coord[0], op.coord[1],
                    op.src_addr, op.bytes,
                    op.activation == "relu"
                ))
        return cpp_ops
    
    @property
    def stats(self):
        return self._engine.stats()
```

---

## 4) Testing Strategy

### 4.1 C++ Unit Tests (`tests/test_kernels.cpp`)

```cpp
// Using Catch2 or GoogleTest
TEST_CASE("matmul_tile_ref correctness") {
    // Test 1: Identity matrix
    float A[32*32], B[32*32], C[32*32];
    // Initialize A as identity, B as random
    // C should equal B after one tile matmul
    
    // Test 2: Known values
    // Small hand-computed example
    
    // Test 3: Accumulation
    // C starts non-zero, verify += semantics
}

TEST_CASE("relu_tile_inplace") {
    float C[32*32];
    // Fill with positive and negative values
    relu_tile_inplace(C);
    // Verify all negatives are zero
}
```

### 4.2 Python Integration Tests (`tests/test_runtime.py`)

```python
def test_single_matmul_correctness():
    """Execute a single 32x32 matmul and verify against NumPy."""
    from compiler.runtime import Runtime
    
    # Setup
    rt = Runtime(sram_bytes=64 * 1024)
    rt.register_tensor("A", (32, 32))
    rt.register_tensor("B", (32, 32))
    rt.register_tensor("C", (32, 32))
    
    # Random inputs
    A = np.random.randn(32, 32).astype(np.float32)
    B = np.random.randn(32, 32).astype(np.float32)
    rt.set_tensor("A", A)
    rt.set_tensor("B", B)
    
    # Manual schedule for single tile matmul
    schedule = [
        SchedLoad("A", (0, 0), dst_addr=0, bytes=4096),
        SchedLoad("B", (0, 0), dst_addr=4096, bytes=4096),
        SchedExecute(m=0, n=0, k=0, a_addr=0, b_addr=4096, acc_addr=8192),
        SchedStore("C", (0, 0), src_addr=8192, bytes=4096),
    ]
    
    rt.execute(schedule)
    
    # Verify
    C = rt.get_tensor("C", (32, 32))
    expected = A @ B
    np.testing.assert_allclose(C, expected, rtol=1e-5)


def test_mlp_end_to_end():
    """Run the full MLP through Python scheduler + C++ runtime."""
    # Build graph
    g = build_mlp_graph()
    
    # Run compiler passes
    FusionPass().run(g)
    TilingPass().run(g)
    LoweringPass().run(g)
    schedule, stats = Scheduler(config).run_on_graph(g)
    
    # Execute on C++ runtime
    rt = Runtime(sram_bytes=config.total_bytes)
    for tensor in g.tensors.values():
        rt.register_tensor(tensor.name, tensor.shape)
    
    # Set inputs (random)
    rt.set_tensor("x", np.random.randn(*g.tensors["x"].shape).astype(np.float32))
    for name in ["w1", "w2", "w3"]:
        rt.set_tensor(name, np.random.randn(*g.tensors[name].shape).astype(np.float32))
    
    rt.execute(schedule)
    
    # Compare with NumPy reference
    # ...
```

### 4.3 Test Coverage Goals

| Component | Tests |
|-----------|-------|
| SRAMArena | Allocation, bounds checking, clear |
| TensorStorage | Registration, tile_ptr math, data round-trip |
| matmul_tile_ref | Identity, known values, accumulation |
| relu_tile_inplace | Positive/negative values |
| Engine | Single op dispatch, multi-op sequence |
| End-to-end | Single matmul, fused matmul+relu, full MLP |

---

## 5) Deliverables Checklist

| Deliverable | File(s) | Description |
|-------------|---------|-------------|
| CMake build | `CMakeLists.txt` | Build C++ runtime + pybind11 module |
| SRAM Arena | `include/mini_runtime/arena.hpp`, `src/runtime/memory/arena.cpp` | Aligned memory buffer |
| Tensor Storage | `include/mini_runtime/tensors.hpp`, `src/runtime/memory/tensors.cpp` | DRAM tensor management |
| Schedule IR | `include/mini_runtime/schedule.hpp` | C++ schedule op types |
| Reference Kernel | `include/mini_runtime/kernels.hpp`, `src/runtime/kernels/matmul_ref.cpp` | 32x32 matmul |
| Engine | `include/mini_runtime/engine.hpp`, `src/runtime/engine/engine.cpp` | Schedule executor |
| pybind11 Bindings | `src/bindings/bindings.cpp` | Python↔C++ bridge |
| Python Wrapper | `src/compiler/runtime.py` | High-level Python API |
| C++ Tests | `tests/test_kernels.cpp` | Kernel correctness |
| Python Tests | `tests/test_runtime.py` | Integration tests |
| Demo Script | `examples/mlp_runtime.py` | End-to-end execution demo |
| Documentation | `docs/week4_report.md` | Implementation report |

---

## 6) Risk Assessment & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| pybind11 version mismatch | Build failures | Pin version in requirements.txt, test on clean env |
| Memory alignment issues | Segfaults, wrong results | Use `std::aligned_alloc`, enable ASAN in debug builds |
| Tile coordinate mismatch | Wrong data copied | Extensive logging, visualize tile access patterns |
| Accumulator semantics | Wrong results | Test accumulation explicitly, compare with NumPy |
| Large tensor padding | Memory waste | Document trade-off, acceptable for portfolio scope |

---

## 7) File Organization (After Week 4)

```
include/
    mini_runtime/
        arena.hpp
        tensors.hpp
        schedule.hpp
        kernels.hpp
        engine.hpp
src/
    bindings/
        bindings.cpp
    compiler/
        runtime.py      # NEW: Python wrapper
        ...
    runtime/
        engine/
            engine.cpp
        memory/
            arena.cpp
            tensors.cpp
        kernels/
            matmul_ref.cpp
            matmul.hpp
tests/
    test_kernels.cpp
    test_runtime.py     # NEW
examples/
    mlp_runtime.py      # NEW: End-to-end demo
```

---

## 8) Success Criteria

Week 4 is complete when:

1. ✅ `pip install -e .` builds the C++ extension without errors
2. ✅ Reference kernel passes all correctness tests
3. ✅ Single 32x32 matmul matches NumPy output (tolerance: `rtol=1e-5`)
4. ✅ Fused matmul+ReLU produces correct output
5. ✅ Full MLP (3 layers) executes end-to-end and matches NumPy reference
6. ✅ All Python tests pass (`pytest tests/`)
7. ✅ C++ tests pass (via CMake/CTest)

---

## 9) Open Questions for Implementation

1. **Tensor name encoding:** Should we pass tensor names as strings or pre-map to IDs in Python?
   - *Proposed:* Map in Python wrapper, pass IDs to C++

2. **Accumulator initialization:** Should the C++ runtime zero-initialize accumulators, or rely on Python schedule to emit explicit zeroing?
   - *Proposed:* C++ clears SRAM arena before execution; first SchedExecute sees zeros

3. **Error handling:** What happens if a SchedLoad references an unregistered tensor?
   - *Proposed:* Throw `std::runtime_error`, caught as Python exception via pybind11

4. **Tracing hooks:** Add instrumentation now or defer to Week 7?
   - *Proposed:* Add optional `trace` flag to Config, emit nothing if false, defer detailed tracing to Week 7

---

## 10) Timeline Estimate

| Day | Tasks |
|-----|-------|
| Day 1 | CMake setup, pybind11 skeleton, arena implementation |
| Day 2 | Tensor storage, reference kernel |
| Day 3 | Engine dispatcher, basic pybind11 bindings |
| Day 4 | Python wrapper, single matmul test passing |
| Day 5 | Full MLP end-to-end, documentation |
| Day 6 | Buffer/polish, edge cases, week4_report.md |

---

*End of Week 4 Plan*
