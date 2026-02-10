# Tileflow

**An educational, end-to-end deep learning compiler and runtime built from scratch.**

This project implements a complete compilation and execution pipeline for simple neural networks, designed to demonstrate the fundamentals of AI systems without the complexity of production frameworks. It targets a simulated hardware architecture with strict constraints to emphasize memory management, scheduling, and operation overlap.

---

## 🎯 Project Goals & Philosophy

The primary goal is to tell the "full stack" story of how a high-level graph becomes executable code on simulated hardware.

> **IR → Tiling & Fusion → Lowering → Scheduling → Runtime → SIMD Kernels → Overlap (DMA + Compute)**

### Key Constraints (The "Hardware")
To keep the scope manageable while remaining architecturally interesting, the project enforces:
*   **Fixed Tiling**: All operations occur on $32 \times 32$ Float32 tiles.
*   **Simulated SRAM**: A small, explicitly managed on-chip memory arena (default 256 KiB).
*   **No External Math Libs**: No BLAS/MKL/CuBLAS; all kernels (reference and AVX2) are hand-written.
*   **Static Scheduling**: Memory allocation and movement are fully resolved at compile time.

---

## 🏗️ Architecture

The system is divided into a **Python compiler** (frontend and middle-end) and a **C++ runtime** (backend).

### 1. Compiler (Python)
*   **Graph IR**: Tensors and operators (`MatMul`, `ReLU`, `Add`), with shape inference and producer/user tracking.
*   **Passes**:
    *   **Fusion**: Fuses patterns like `MatMul → ReLU` into `FusedMatMulReLU` to reduce memory traffic.
    *   **Tiling**: Validates and annotates tile-aligned dimensions; pads to multiples of 32.
    *   **Lowering**: Converts graph ops into a linear stream of tile-level uOps (`LOAD`, `EXECUTE`, `STORE`).
*   **Scheduler**: Allocates addresses in a virtual SRAM, emits a linear schedule with concrete addresses, eliminates redundant loads via residency tracking, and supports optional double-buffering tags (buffer 0/1 per K-iteration). Full double buffering (separate SRAM regions for ping/pong) is not yet implemented in the allocator.

### 2. Runtime (C++)
*   **Execution modes**: Sequential (single-thread) or **threaded** (pipelined DMA + Compute).
*   **Threaded engine** (opt-in via `threaded=True`):
    *   **DMA thread**: Performs `LOAD` (tensor → SRAM) and `STORE` (SRAM → tensor). Before each LOAD it ensures no in-flight EXEC is still reading that address (address-conflict tracking).
    *   **Compute thread**: Pops work items and runs the matmul kernel; signals the DMA thread when an accumulator is ready to store.
    *   **Communication**: Lock-free SPSC ring buffers (DMA → Compute for work items, Compute → DMA for store notifications). A backward pass over the schedule pairs each STORE with the correct “last K” EXEC so multi-layer graphs with address reuse do not deadlock.
*   **Overlap metrics**: When using the threaded engine, stats report DMA/compute utilization and overlap time (nanoseconds both threads were active simultaneously).
*   **Kernels**:
    *   **Reference**: Triple-loop 32×32 matmul-accumulate and ReLU for correctness.
    *   **AVX2**: Optimized 6×16 micro-kernel with FMA; compile-time detection and optional explicit dispatch for benchmarking.

---

## 📦 Installation

### Prerequisites
*   Linux (or WSL) with a C++17-capable compiler (GCC/Clang)
*   Python 3.10+
*   pybind11 (for the C++ extension)

### Build & Install
The C++ runtime is built as a Python extension via pybind11.

```bash
# Clone and enter the project
git clone https://github.com/re-pixel/tileflow.git
cd tileflow

# Optional: use a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install (builds the mini_runtime extension)
pip install -r requirements.txt
pip install -e .
```

---

## 🚀 Usage

### End-to-end MLP (sequential or threaded)
Run the full pipeline: graph → fusion → tiling → lowering → scheduling → execution.

```bash
# Sequential execution (default)
python3 -m examples.mlp_runtime

# Benchmark: sequential vs threaded, with overlap metrics
python3 -m benchmarks.mlp_threaded
```

### Other examples
*   `examples/mlp_tiled.py` — Graph building and tiling summary
*   `examples/mlp_fused.py` — Fusion pass
*   `examples/mlp_scheduled.py` — Schedule and SRAM stats

### Using the runtime from Python
```python
from compiler.ir import Graph
from compiler.passes import FusionPass, LoweringPass, TilingPass
from compiler.runtime import Runtime
from compiler.scheduler import Scheduler, SRAMConfig

# Build and compile a graph (see examples)
g = Graph("model")
# ... add inputs, params, matmul, relu ...
FusionPass().run(g)
TilingPass().run(g)
LoweringPass().run(g)
schedule, _ = Scheduler(config=SRAMConfig(total_bytes=256*1024)).run_on_graph(g)

# Run (sequential or threaded)
rt = Runtime(sram_bytes=256*1024, threaded=True)
rt.register_graph_tensors(g)
rt.set_tensor("x", x_data)
# ... set other tensors ...
rt.execute(schedule)
result = rt.get_tensor("out", (128, 32))

# Threaded mode: overlap stats (dma_busy_ns, compute_busy_ns, overlap_ns, total_ns)
print(rt.stats.overlap_ns, rt.stats.total_ns)
```

---

## 📂 File Structure

```plaintext
mini-compiler/
├── include/mini_runtime/    # C++ headers (engine, schedule, kernels, ring_buffer, etc.)
├── src/
│   ├── compiler/            # Python compiler
│   │   ├── ir/              # Graph, tensors, operators
│   │   ├── passes/          # Fusion, tiling, lowering
│   │   └── scheduler/       # Virtual SRAM, schedule IR, static scheduler
│   ├── runtime/             # C++ backend
│   │   ├── engine/          # Sequential + threaded engine
│   │   ├── kernels/         # Reference & AVX2 matmul/ReLU
│   │   └── memory/          # SRAM arena, tensor storage
│   └── bindings/            # pybind11 bindings
├── examples/                # MLP demos (tiled, fused, scheduled, runtime)
├── benchmarks/               # Kernel and MLP benchmarks (incl. sequential vs threaded)
├── tests/                   # Pytest + C++ tests
└── docs/                    # Week reports, roofline, architecture
```

---

## 📚 Status

The project is in a **working prototype** state and reflects an end-to-end pipeline from graph to execution.

| Component | Status |
|-----------|--------|
| Graph IR (tensors, MatMul, ReLU, Add) | ✅ |
| Fusion pass (MatMul+ReLU) | ✅ |
| Tiling & lowering to tile uOps | ✅ |
| Static scheduler (virtual SRAM, load elimination) | ✅ |
| Double-buffering tags in schedule | ✅ (allocator ping/pong not yet) |
| C++ runtime (sequential + threaded) | ✅ |
| Lock-free SPSC ring buffer, address-conflict tracking | ✅ |
| Overlap instrumentation (DMA/compute/overlap ns) | ✅ |
| Reference & AVX2 32×32 kernels | ✅ |
| Python bindings & Runtime wrapper | ✅ |
| Tests (pytest + C++ ring buffer) | ✅ |
| Roofline notes & week reports | ✅ |
