# Week 6 Plan — Virtual Core & Concurrency

**Date:** 2026-01-23  
**Phase:** Phase 2 (Backend Optimization, C++)  
**Week 6 Goal:** Transform the single-threaded runtime into a dual-threaded, pipelined architecture that separates data movement from compute, mirroring real accelerator designs.

---

## 1) Executive Summary

Week 6 introduces **concurrency** to the runtime. The current `Engine::execute()` processes operations sequentially, leaving either the memory subsystem or the compute units idle at any given moment. This week, we split execution into two threads communicating via a lock-free queue.

| Component | Description |
|-----------|-------------|
| **DMA Thread** | Handles `SchedLoad` and `SchedStore` operations |
| **Compute Thread** | Handles `SchedExecute` operations (matmul kernels) |
| **Ring Buffer** | Lock-free SPSC queue for passing work items between threads |
| **Double Buffering** | Ping-pong operand regions to allow overlap |
| **Synchronization** | Barrier mechanism for STORE operations |

**Key Insight:** Real accelerators (TPUs, Tenstorrent Tensix) separate data movement from compute to maximize utilization. By overlapping LOAD operations with EXEC, we can hide memory latency and approach higher sustained throughput.

---

## 2) Background: Why Concurrency Matters

### 2.1 The Sequential Execution Problem

The current Week 5 runtime executes operations in strict order:

```
Time →
┌──────┐     ┌──────┐     ┌──────┐     ┌───────┐     ┌──────┐
│LOAD A│ →   │LOAD B│ →   │ EXEC │ →   │STORE C│ →   │LOAD A│ → ...
└──────┘     └──────┘     └──────┘     └───────┘     └──────┘
  Memory       Memory      Compute       Memory       Memory
    (Compute idle)                          (Compute idle)
```

**Utilization:** Only ~33% of peak (compute active 1/3 of the time).

### 2.2 Pipelined Execution Target

With two threads working in parallel:

```
Time →
DMA Thread:    │LOAD A│LOAD B│LOAD A'│LOAD B'│STORE C│LOAD A''│...
               └──────┴──────┴───────┴───────┴───────┴────────┘
                      ↓       ↓        ↓        ↓
Compute Thread:       │ EXEC  │ EXEC'  │ EXEC'' │ ...
                      └───────┴────────┴────────┘
```

**Utilization:** Approaching 100% (both threads active simultaneously).

---

## 3) Design Decisions & Rationale

### 3.1 Thread Architecture

**Decision:** Two dedicated threads with distinct roles.

| Thread | Role | Operations |
|--------|------|------------|
| **DMA Thread** | Data movement | `SchedLoad`, `SchedStore` |
| **Compute Thread** | Arithmetic | `SchedExecute` |

**Rationale:**
- Clean separation mirrors hardware (DMA engine vs. compute core)
- Avoids complex work-stealing schedulers
- Easier to reason about and debug

**Alternative considered:** Thread pool with dynamic dispatch. Rejected for complexity and unpredictable scheduling.

### 3.2 Inter-Thread Communication

**Decision:** Lock-free Single-Producer Single-Consumer (SPSC) ring buffer.

**Rationale:**
- SPSC avoids all mutex contention (only atomic increments)
- Fixed capacity (8 slots) bounds memory usage
- Well-understood pattern with decades of production use

**Implementation sketch:**
```cpp
template <typename T, size_t Capacity>
class RingBuffer {
    std::array<T, Capacity> buffer_;
    std::atomic<size_t> head_{0};  // Written by producer
    std::atomic<size_t> tail_{0};  // Written by consumer
    
public:
    bool try_push(const T& item);
    bool try_pop(T& item);
};
```

### 3.3 SRAM Partitioning (Double Buffering)

**Decision:** Split operand SRAM into two regions (ping/pong).

```
┌────────────────────────────────────────────────────────────────┐
│  SRAM Layout (256 KiB)                                         │
├────────────────┬────────────────┬──────────────────────────────┤
│  Buffer 0      │  Buffer 1      │  Accumulators                │
│  (Ping)        │  (Pong)        │  (Protected)                 │
│  ~64 KiB       │  ~64 KiB       │  ~128 KiB                    │
│  DMA writes    │  DMA writes    │  Compute reads/writes        │
│  while Compute │  while Compute │                              │
│  reads Buffer 1│  reads Buffer 0│                              │
└────────────────┴────────────────┴──────────────────────────────┘
```

**Rationale:**
- While compute uses Buffer 0, DMA loads into Buffer 1
- On next iteration, roles swap
- Accumulators never move (allocated once, stored once)

**Note:** The scheduler already emits a `buffer` field in `SchedLoad`/`SchedExecute`. Week 6 activates this.

### 3.4 Synchronization for STORE

**Decision:** Use a condition variable to signal accumulator completion.

**Problem:** The DMA thread cannot store an accumulator until the compute thread finishes all partial products for that tile.

**Solution:**
```cpp
// DMA thread (before STORE):
wait_for_accumulator_ready(m, n);  // Blocks until signaled

// Compute thread (after final EXEC for tile m,n):
signal_accumulator_ready(m, n);    // Wakes DMA thread
```

**Implementation:** Per-accumulator flags or a secondary queue of "ready to store" notifications.

---

## 4) Implementation Plan

### 4.1 Phase 1: Lock-Free Ring Buffer

**File:** `include/mini_runtime/ring_buffer.hpp`

```cpp
template <typename T, size_t Capacity>
class RingBuffer {
public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    
    bool try_push(const T& item);
    bool try_pop(T& item);
    bool is_empty() const;
    bool is_full() const;
    size_t size() const;
};
```

**Key properties:**
- Power-of-2 capacity for fast modulo (bitwise AND)
- Acquire/release memory ordering for cross-thread visibility
- No dynamic allocation

### 4.2 Phase 2: Work Item Definitions

**File:** `include/mini_runtime/work_item.hpp`

```cpp
struct ComputeWorkItem {
    uint32_t a_addr;      // SRAM address of A tile
    uint32_t b_addr;      // SRAM address of B tile
    uint32_t acc_addr;    // SRAM address of accumulator
    uint32_t m, n, k;     // Tile coordinates
    bool     is_first_k;  // If true, clear accumulator before compute
    bool     is_last_k;   // If true, signal accumulator ready after compute
};

struct StoreNotification {
    uint32_t m, n;        // Accumulator tile coordinates
    uint32_t acc_addr;    // Where to read from
    uint32_t tensor_id;   // Where to write to
    uint32_t tile_row, tile_col;
    bool     apply_relu;
};
```

### 4.3 Phase 3: Threaded Engine

**File:** `src/runtime/engine/threaded_engine.cpp`

**New class or mode:**
```cpp
class ThreadedEngine {
public:
    void execute(const std::vector<SchedOp>& schedule);
    
private:
    void dma_thread_func(const std::vector<SchedOp>& schedule);
    void compute_thread_func();
    
    RingBuffer<ComputeWorkItem, 8> compute_queue_;
    RingBuffer<StoreNotification, 8> store_queue_;
    std::atomic<bool> done_{false};
};
```

**DMA thread logic:**
1. Iterate through schedule
2. On `SchedLoad`: Copy data from tensor to SRAM (buffer region based on `op.buffer`)
3. On `SchedExecute`: Push `ComputeWorkItem` to `compute_queue_`
4. On `SchedStore`: Wait for `StoreNotification`, then copy SRAM to tensor
5. Set `done_ = true` when schedule exhausted

**Compute thread logic:**
1. Loop: Pop from `compute_queue_` (spin or wait)
2. Execute `matmul_tile()` kernel
3. If `is_last_k`: Push `StoreNotification` to `store_queue_`
4. Exit when `done_` and queue empty

### 4.4 Phase 4: Engine Configuration

**File:** `include/mini_runtime/engine.hpp`

```cpp
struct Engine::Config {
    size_t sram_bytes = 256 * 1024;
    bool   trace = false;
    bool   threaded = false;           // NEW: Enable dual-thread mode
    size_t ring_buffer_capacity = 8;   // NEW: Work queue depth
};
```

### 4.5 Phase 5: Python Bindings Update

**File:** `src/bindings/bindings.cpp`

- Expose `threaded` config option
- Add `set_threaded(bool)` method to Engine

---

## 5) Testing Strategy

### 5.1 Unit Tests

**File:** `tests/test_ring_buffer.cpp`

- Single-threaded push/pop correctness
- Capacity limits (full/empty detection)
- Wrap-around behavior

### 5.2 Integration Tests

**File:** `tests/test_threaded_runtime.py`

```python
def test_threaded_matches_sequential():
    """Threaded engine produces identical output to sequential."""
    schedule = compile_mlp()
    
    # Sequential execution
    engine_seq = Engine(Config(threaded=False))
    result_seq = engine_seq.execute(schedule)
    
    # Threaded execution
    engine_thr = Engine(Config(threaded=True))
    result_thr = engine_thr.execute(schedule)
    
    assert np.allclose(result_seq, result_thr)

def test_threaded_stress():
    """No crashes or hangs under repeated execution."""
    for _ in range(1000):
        engine.execute(schedule)
```

### 5.3 Race Condition Detection

- Run tests under ThreadSanitizer (`-fsanitize=thread`)
- Add to CMake as optional build target

---

## 6) Benchmarking

### 6.1 Metrics to Measure

| Metric | Description |
|--------|-------------|
| **Latency** | Total time for MLP execution |
| **Overlap Ratio** | Time both threads are active / Total time |
| **Queue Depth** | Average items in ring buffer during execution |
| **Speedup** | Sequential time / Threaded time |

### 6.2 Expected Results

For our MLP workload:
- **Sequential:** ~0.15 ms/iteration (Week 5 baseline)
- **Threaded:** ~0.10-0.12 ms/iteration (estimated 20-30% improvement)
- **Overlap Ratio:** 60-80% (limited by schedule structure)

**Note:** Small tiles (32×32) and short schedules may not show dramatic improvement. The benefit scales with larger workloads.

---

## 7) Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-----------|
| Race conditions on SRAM | Medium | High | Double-buffer regions; accumulators isolated |
| Deadlock on STORE barrier | Low | High | Careful ordering; timeout in debug builds |
| Overhead > Benefit | Medium | Medium | Make threading optional; benchmark both |
| Platform differences (atomics) | Low | Medium | Use C++11 `<atomic>` standard library |

---

## 8) Deliverables

| File | Description |
|------|-------------|
| `include/mini_runtime/ring_buffer.hpp` | Lock-free SPSC circular buffer |
| `include/mini_runtime/work_item.hpp` | Work item struct definitions |
| `src/runtime/engine/threaded_engine.cpp` | Dual-thread engine implementation |
| `tests/test_ring_buffer.cpp` | Ring buffer unit tests |
| `tests/test_threaded_runtime.py` | Correctness and stress tests |
| `examples/mlp_threaded.py` | Demo script for threaded execution |
| `docs/week6_report.md` | Design documentation and results |

---

## 9) Success Criteria

1. **Correctness:** Threaded engine produces bit-identical output to sequential engine on all test cases.
2. **Stability:** No crashes, hangs, or ThreadSanitizer warnings under stress testing (1000+ iterations).
3. **Performance:** Measurable speedup (>10%) on MLP benchmark, or clear instrumentation showing overlap.
4. **Code Quality:** Clean separation of concerns; threading is opt-in via config flag.

---

## 10) Look Ahead (Week 7)

With concurrency in place, Week 7 focuses on **observability**:

- **Chrome Tracing:** Export execution traces in `chrome://tracing` format
- **Per-Operation Timing:** Instrument LOAD/EXEC/STORE durations
- **Overlap Visualization:** See DMA and Compute threads on separate lanes
- **Bottleneck Identification:** Find idle gaps and their causes
