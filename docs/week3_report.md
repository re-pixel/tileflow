# Week 3 Report — Memory Planner & Static Scheduler (mini-compiler)

**Date:** 2026-01-10  
**Phase:** Phase 1 (Compiler Frontend, Python)  
**Week 3 Goal:** Implement a virtual SRAM allocator and static scheduler that converts Week 2's logical uOps into executable scheduled operations with concrete memory addresses.

---

## 1) Executive Summary

Week 3 successfully implemented the **Memory Planner** and **Static Scheduler**:

- **VirtualSRAMArena:** A first-fit allocator simulating on-chip SRAM with 64-byte alignment, block coalescing, LRU tracking for eviction, and detailed OOM diagnostics.
- **Schedule IR:** Defined `SchedLoad`, `SchedExecute`, `SchedStore` operations that carry concrete SRAM addresses — directly consumable by the Week 4+ C++ runtime.
- **Liveness Tracking:** Split tracking for operand tiles (A/B) and accumulator tiles (C) with strict invariants: accumulators are never evicted.
- **Redundant Load Elimination:** 55.7% of tile loads eliminated via residency tracking on an MLP workload.
- **Double Buffering:** Optional mode that annotates schedule ops with buffer IDs (`k % 2`) scoped to K-loops for pipelined execution.
- **Graph Integration:** Scheduler integrates cleanly with the graph pipeline via `run_on_graph()`.

This completes the *static planning* phase. Week 4 will implement the C++ runtime that executes these schedules.

---

## 2) Key Implementation Details

### 2.1 VirtualSRAMArena (`src/compiler/scheduler/memory.py`)

A simulated SRAM allocator designed for debuggability:

| Feature | Description |
|---------|-------------|
| **First-fit allocation** | Scans free list for first block that fits |
| **64-byte alignment** | All allocations aligned for SIMD loads |
| **Block coalescing** | Adjacent free blocks merged on `free()` |
| **LRU tracking** | Access order maintained for eviction policy |
| **Peak tracking** | High-water mark for memory pressure analysis |
| **OOM diagnostics** | Detailed error messages listing live allocations |

Configuration via `SRAMConfig(total_bytes, alignment)`.

### 2.2 Schedule IR (`src/compiler/scheduler/ops.py`)

Physical execution operations with concrete addresses:

| SchedOp | Description |
|---------|-------------|
| `SchedLoad(tensor, coord, dst_addr, bytes)` | Load a tile into SRAM at `dst_addr` |
| `SchedExecute(m, n, k, a_addr, b_addr, acc_addr)` | Tile matmul-accumulate with explicit addresses |
| `SchedStore(tensor, coord, src_addr, bytes, activation)` | Store tile from SRAM, optionally apply ReLU |

All ops carry an optional `buffer: int | None` field for double buffering.

### 2.3 Scheduler Algorithm (`src/compiler/scheduler/scheduler.py`)

The scheduling algorithm follows these phases:

1. **Liveness Precompute:** Count remaining uses for each operand tile; accumulators have lifetime = 1 (first EXEC to STORE).
2. **Residency Tracking:** On LOAD, check if tile is already resident. If yes, skip load (increment `loads_eliminated`).
3. **Allocation:** First-fit from arena. If OOM, evict LRU operand tiles (never accumulators).
4. **Accumulator Invariants:**
   - Allocated on first `EXEC(m,n,*)` for each output tile
   - Freed on matching `STORE(m,n)`
   - Never evicted (protected from LRU)
5. **Operand Freeing:** Decrement use count after each EXEC; free when count reaches 0.
6. **Double Buffering:** When enabled, assigns `buffer = k % 2` within each (m,n) K-loop, resetting on (m,n) change.

### 2.4 Statistics Contract (`ScheduleStats`)

Explicit metrics returned by the scheduler:

```python
@dataclass
class ScheduleStats:
    sched_ops: int           # Total scheduled operations
    loads_emitted: int       # LOAD ops in output schedule
    loads_eliminated: int    # Loads skipped via residency
    peak_sram_bytes: int     # High-water SRAM usage
    final_live_bytes: int    # Should be 0 (all freed)
```

---

## 3) Deliverables Check

| Deliverable | Status | Location |
|-------------|--------|----------|
| **VirtualSRAMArena** | ✅ Complete | `src/compiler/scheduler/memory.py` |
| **Schedule IR** | ✅ Complete | `src/compiler/scheduler/ops.py` |
| **Scheduler** | ✅ Complete | `src/compiler/scheduler/scheduler.py` |
| **Double Buffering** | ✅ Complete | `Scheduler(double_buffer=True)` |
| **Graph Integration** | ✅ Complete | `Scheduler.run_on_graph()` |
| **Tests** | ✅ Complete (18 tests) | `tests/test_scheduler.py` |
| **Demo Script** | ✅ Complete | `examples/mlp_scheduled.py` |

---

## 4) Example Output

Running `PYTHONPATH=src python -m examples.mlp_scheduled`:

```text
======================================================================
Week 3 Demo — MLP Scheduling Pipeline
======================================================================

[1] Building MLP graph...
    Original graph: 5 ops
    Tensors: 9

[2] Running Fusion Pass...
    After fusion: 3 ops

[3] Running Tiling Pass...
    Tiling validated: True

[4] Running Lowering Pass...
    Generated 148 micro-ops (uOps)

[5] Running Scheduling Pass...
    SRAM size: 256 KiB

======================================================================
Schedule Statistics (Single Buffer)
======================================================================
  Total scheduled ops: 99
  Loads emitted:       39
  Loads eliminated:    49 (55.7% reuse)
  Peak SRAM usage:     53,248 bytes (52.0 KiB)
  Final live bytes:    0

======================================================================
Sample Schedule (first 15 operations)
======================================================================
  LOAD x(0, 0) -> @0x0000 (4096B)
  LOAD w1(0, 0) -> @0x1000 (4096B)
  EXEC (0,0,0) A=@0x0000 B=@0x1000 ACC=@0x2000
  LOAD x(0, 1) -> @0x3000 (4096B)
  LOAD w1(1, 0) -> @0x4000 (4096B)
  EXEC (0,0,1) A=@0x3000 B=@0x4000 ACC=@0x2000
  ...

======================================================================
Summary
======================================================================
  Graph ops:          3
  Lowered uOps:       148
  Scheduled ops:      99
  Load reuse:         49/88 (55.7%)
  Peak SRAM:          52.0 KiB / 256 KiB
  Final live bytes:   0

  Invariants:
    ✓ Final live bytes = 0: PASS
    ✓ Load reuse > 0:       PASS
    ✓ Peak < SRAM size:     PASS
    ✓ Graph attrs set:      PASS
```

---

## 5) Test Coverage

The `tests/test_scheduler.py` suite validates:

| Test Category | Tests |
|---------------|-------|
| **Arena Basics** | alloc/free roundtrip, alignment, peak tracking, reset |
| **OOM Handling** | Raises with capacity, live bytes, allocation tags |
| **Redundant Load Elimination** | Same tile loaded once, reuse counted |
| **Liveness Freeing** | Final live bytes = 0, peak bytes reasonable |
| **Double Buffering** | Alternating buffer IDs, load buffer matches exec, reset on (m,n) change |
| **End-to-End MLP** | Non-empty schedule, contains EXEC ops, load reuse, graph attrs populated |

All 28 project tests pass.

---

## 6) Architecture Diagram

```
Week 2 Output                   Week 3 Pipeline
─────────────────────────────────────────────────────────────────────
                                          
  ┌─────────────┐                ┌─────────────────────┐
  │   uOps      │                │     Scheduler       │
  │ (logical)   │ ──────────────▶│                     │
  │             │                │  ┌───────────────┐  │
  │ LOAD A(m,k) │                │  │ Liveness      │  │
  │ LOAD B(k,n) │                │  │ Precompute    │  │
  │ EXEC(m,n,k) │                │  └───────┬───────┘  │
  │ STORE C(m,n)│                │          │          │
  └─────────────┘                │  ┌───────▼───────┐  │
                                 │  │ VirtualSRAM   │  │
                                 │  │ Arena         │  │
                                 │  │ (alloc/free)  │  │
                                 │  └───────┬───────┘  │
                                 │          │          │
                                 │  ┌───────▼───────┐  │
                                 │  │ Residency     │  │
                                 │  │ Tracking      │  │
                                 │  └───────┬───────┘  │
                                 │          │          │
                                 └──────────┼──────────┘
                                            │
                                            ▼
                                 ┌─────────────────────┐
                                 │     SchedOps        │
                                 │    (physical)       │
                                 │                     │
                                 │ LOAD x(0,0)→@0x0000 │
                                 │ EXEC A=@0x0000      │
                                 │      B=@0x1000      │
                                 │      ACC=@0x2000    │
                                 │ STORE @0x2000→out   │
                                 └─────────────────────┘
```

---

## 7) Look Ahead (Week 4)

With a physical schedule containing concrete SRAM addresses, the next phase is **C++ Runtime & Python Bridge**:

- **pybind11 bindings:** Expose C++ runtime to Python.
- **Memory Arena:** Implement `std::aligned_alloc`-based SRAM simulation in C++.
- **Schedule Executor:** C++ engine that consumes `SchedLoad`, `SchedExecute`, `SchedStore`.
- **Tile Kernels:** Reference $32\times 32$ matmul kernel for correctness.
