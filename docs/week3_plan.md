# Week 3 Plan — Memory Planning & Static Scheduling (mini-compiler)

**Focus:** Turn Week 2’s logical uOps stream into a **real, executable schedule** by adding:
- a **VirtualSRAMArena memory planner** (tile residency, allocation, eviction), and
- a **static scheduler** (uOps → scheduled ops with concrete SRAM addresses, optional double buffering).

**Pre-requisites:** Completed Week 2 (Fusion, Tiling Validator, Lowering to uOps).

---

## 1. Objectives

1. **VirtualSRAMArena:** implement a deterministic SRAM allocator that can place tile buffers into a fixed-size arena and report peak usage.
2. **uOp → Scheduled Op lowering:** translate logical uOps (`LOAD tensor(m,k)`, etc.) into scheduled ops with **concrete SRAM addresses**.
3. **Redundant load elimination:** if a tile is already resident in SRAM, do not reload it.
4. **Liveness & eviction:** free tiles after last use; if SRAM is full, evict **operand tiles only** (never accumulators) using a simple, deterministic policy.
5. **Double buffering (core Week 3 “signal”):** add an optional scheduling mode that alternates between two SRAM buffers to enable overlap of `LOAD` and `EXEC` in future runtime work.
6. **Inspectable artifacts:** produce a human-readable schedule dump + explicit stats (counts, reuse hits, peak SRAM, final live bytes).
7. **Tests:** lock allocator correctness + schedule invariants in pytest.

---

## 2. Ground Rules / Hardware Model (Week 3)

These are the assumptions the Week 3 code should encode explicitly (as constants/config):

- **Tile shape:** fixed $32\times 32$.
- **Tile dtype:** FP32.
- **Tile bytes:** $32 \cdot 32 \cdot 4 = 4096$ bytes (4 KiB) per tile.
- **SRAM is small:** configurable (e.g., 512 KiB–1 MiB). Use a default but make it easy to override.
- **No numeric execution yet:** we’re generating a *schedule*, not running kernels.
- **Schedule is sequential:** Week 3 outputs a linear schedule, but may annotate “buffer id” for future overlap.

---

## 3. Implementation Steps

### 3.1. Define the Week 3 schedule IR (`src/compiler/scheduler/execution.py`)

**Task:** Create an explicit schedule representation that is stable and serializable.

**Why:** Week 2 uOps are logical (“load tensor tile (m,k)”). Week 3 needs **physical placement** and a format that Week 4+ runtime can consume.

**Recommended data model (minimal but future-proof):**

- `SchedOp` (base class)
- `SchedLoad(SchedOp)`
  - fields: `tensor: str`, `coord: tuple[int, ...]`, `dst_addr: int`, `bytes: int`, optional `buffer: int`
- `SchedExecute(SchedOp)`
  - fields: `m: int`, `n: int`, `k: int`, `a_addr: int`, `b_addr: int`, `acc_addr: int`, optional `buffer: int`
- `SchedStore(SchedOp)`
  - fields: `tensor: str`, `coord: tuple[int, ...]`, `src_addr: int`, `bytes: int`, `activation: str | None`

**Notes:**
- Keep schedule ops “flat” (list order = execution order).
- Avoid adding a complex dependency graph yet. If you need correctness gates, add a simple `SchedBarrier(kind=...)` later as an extension.
- Implement `__repr__` for readable debugging (mirroring Week 2 uOps’ style).

**Deliverable:** `execution.py` contains schedule IR + helper to pretty-print a schedule.

---

### 3.2. Implement VirtualSRAMArena + allocator (`src/compiler/scheduler/memory.py`)

**Task:** Build a simple deterministic allocator for a fixed-size SRAM arena.

**Core API (suggested):**

- `SRAMConfig(total_bytes: int, alignment: int = 64)`
- `VirtualSRAMArena(config)`
  - `alloc(bytes: int, tag: str) -> int` (returns base address)
  - `free(addr: int) -> None`
  - `reset() -> None`
  - `peak_bytes` and `live_bytes`

**Implementation approach:**

1. **Start simplest:** bump allocator + whole-reset (useful for debugging), then upgrade.
2. **Then implement reuse:** free-list allocator (first-fit) with alignment.
3. **Track ownership:** maintain `addr -> (bytes, tag)` for diagnostics.
4. **Fail loudly:** on OOM, raise with a message including `total_bytes`, `live_bytes`, and top N live allocations.

**Tile accounting helpers:**
- `tile_bytes(tile_m=32, tile_n=32, dtype_bytes=4) -> int`

**Deliverable:** `memory.py` implements `VirtualSRAMArena` with enough features to support scheduling and tests.

---

### 3.3. Add a Scheduler / MemoryPlanner entrypoint (`src/compiler/scheduler/`)

**Task:** Implement a high-level class that converts Week 2 uOps into a Week 3 schedule.

**Suggested file structure (keep minimal):**
- Implement in `src/compiler/scheduler/execution.py` (or add `src/compiler/scheduler/planner.py` if you want separation).

**API (suggested):**
- `Scheduler(config: SRAMConfig, double_buffer: bool = False)`
  - `run(uops: list[UOp]) -> list[SchedOp]`

**Algorithm (Phase A: correctness-first):**

1. **Identify tile keys:** represent each tile as `TileKey = (tensor_name, coord)`.
2. **Residency map:** `resident: dict[TileKey, addr]`.
3. **Use counting / liveness (be explicit about what “use” means):** precompute remaining-use counters by scanning uOps, but split it into two conceptual classes for clarity.

   - **Operand tiles (A/B):** short-lived. A “use” is consuming the tile in an `EXEC`.
     - Practically in Week 3 (given Week 2 lowering), you can treat each `UOpLoad(A[m,k])` / `UOpLoad(B[k,n])` as a request for an operand tile and count that as one *operand use*.
     - Track as: `remaining_uses_operands: dict[TileKey, int]`.

   - **Accumulator tiles (C[m,n]):** long-lived across the entire K loop.
     - Do **not** count per-`EXEC` uses the same way as operands; instead treat accumulator lifetime structurally.
     - Track as: `remaining_uses_accumulators: dict[TileKey, int]` (often 1, meaning “must survive until STORE”).

   This is not “full dataflow analysis”; it’s a naming + intent distinction that prevents bugs later.

**Important design choice:** Week 2’s uOps currently encode loads explicitly before each execute. In Week 3, treat loads as *requests* that can be skipped if already resident.

4. **Scheduling rules:**
   - For each `UOpLoad(tile)`:
     - If `tile` is resident → skip.
     - Else allocate SRAM + emit `SchedLoad(..., dst_addr=addr)`.
   - For each `UOpExecute(m,n,k)`:
     - Look up addresses for A/B tiles that the lowering just referenced.
     - **Accumulator invariant (codify this as a rule):**
       - Accumulator tiles `C[m,n]` are allocated on the **first** `SchedExecute(m,n,*)`.
       - Accumulator tiles `C[m,n]` are freed on the matching `SchedStore(m,n)`.
       - Accumulators must not be evicted in Week 3.
     - Emit `SchedExecute` with the concrete addresses (note: `acc_addr` is read/write).
   - For each `UOpStore(out, (m,n), activation)`:
     - Emit `SchedStore` from accumulator SRAM addr.
     - Free accumulator when done (per invariant).

5. **Free policy (Phase A):**
  - For **operand tiles only**: decrement `remaining_uses_operands` when their corresponding `EXEC` is scheduled (or equivalently when their load-request is consumed, given Week 2’s uOp structure).
  - If `remaining_uses_operands[tile]` hits 0 → free SRAM and remove from `resident`.
  - For **accumulators**: do not decrement per-op. Allocate on first execute, free on store.

**Phase B: handle OOM with eviction (keep Week 3 scope tight):**

- If `alloc()` fails:
  - First, sanity-check that tiles with `remaining_uses_operands == 0` have been freed (they should not remain resident).
  - Then evict **operand tiles only** (never accumulators), using deterministic tiers:
    1. **Operand tiles** with `remaining_uses_operands > 0` are eviction candidates.
    2. Choose **LRU among operand tiles** as the Week 3 policy.
  - Document “Belady / next-use eviction” as a Week 4+ optimization (too much bookkeeping for Week 3).

**Deliverable:** a scheduler that can produce a valid scheduled op stream for the MLP example without exceeding SRAM in typical configs.

---

### 3.4. Double buffering mode (optional flag, but part of Week 3 deliverable)

**Task:** Add a mode that alternates between two SRAM regions (“ping/pong”) for *streaming* operands.

**Goal:** Encode the concept that while compute uses buffer 0, the next tiles can be loaded into buffer 1, and vice versa.

**Constraints (keep simple):**
- Start by double-buffering only **A and B operand tiles**.
- Keep accumulator tiles single-buffered initially (accumulators are typically long-lived over K-loop).
- Restrict double buffering to inside a single `(m,n)` tile’s K-loop; reset ping/pong when `(m,n)` changes.

**Implementation idea:**

- Partition SRAM logically:
  - `bank0` range and `bank1` range (each is a `VirtualSRAMArena` instance with half the bytes), OR
  - one allocator but enforce address ranges by “tagging” allocations.
- Only when scheduling a sequence of `EXEC`s for the same `(m,n)`, choose `buf = k % 2` for operand loads.
- Reset `buf = 0` when `(m,n)` changes (keeps reasoning simple and prevents cross-output-tile interactions).
- Emit schedule in this pattern:
  1. `LOAD A(m,k) -> buf`
  2. `LOAD B(k,n) -> buf`
  3. `EXEC using buf`

**Note:** True overlap requires a runtime with separate DMA/compute engines; Week 3 only establishes the structure. Still, the schedule must remain valid if executed sequentially.

**Deliverable:** double-buffering annotations in schedule ops (`buffer` field) and a simple “ping/pong” strategy that doesn’t violate SRAM capacity.

---

### 3.5. Graph integration & pass ordering

**Task:** Make scheduling easy to run from examples.

**Target pipeline (Week 3):**

1. Build graph
2. `FusionPass` (Week 2)
3. `TilingPass` / `HardwareConstraintValidator` (Week 1/2)
4. `LoweringPass` (Week 2) → `graph.attrs["lowered"] = list[UOp]`
5. **Scheduler (Week 3)** → `graph.attrs["schedule"] = list[SchedOp]` and `graph.attrs["schedule_stats"] = {...}`

**Schedule stats (explicit contract):**

```python
schedule_stats = {
  "sched_ops": int,
  "loads_emitted": int,
  "loads_eliminated": int,
  "peak_sram_bytes": int,
  "final_live_bytes": int,
}
```

**Deliverable:** a single call site in examples can run the full pipeline and print schedule stats.

---

## 4. Testing & Verification

### 4.1. Unit tests (`tests/`)

**Update / implement:** `tests/test_scheduler.py`.

Recommended tests:

1. **VirtualSRAMArena basics**
   - alloc/free round-trip
   - alignment respected
   - peak_bytes tracked
  - OOM raises with good message

1b. **Intentional OOM test (debuggability)**
  - Configure SRAM to something tiny (e.g., 8 KiB).
  - Attempt to allocate/load enough distinct tiles to exceed capacity.
  - Assert the exception message mentions total bytes, live bytes, and includes some allocation tags/sizes.

2. **Redundant load elimination**
   - Construct a small uOp sequence that loads the same tile twice without eviction.
   - Assert schedule contains only one `SchedLoad` for that tile.

3. **Liveness frees memory**
   - Build a uOp sequence where tiles are not used after a point.
   - Assert that allocator live_bytes returns to 0 at end, and peak_bytes is reasonable.

4. **Double buffering annotations**
   - Run scheduler with `double_buffer=True`.
   - Assert alternating buffer ids for operand loads.

5. **End-to-end smoke (MLP)**
   - Run: build MLP → fusion → tiling → lowering → scheduling.
   - Assert:
     - schedule is non-empty
     - contains `SchedExecute` ops
     - total `SchedLoad` count is strictly less than naive uOp load count (sanity that reuse is happening)
    - `graph.attrs["schedule_stats"]` exists and has the required keys

### 4.2. Debug artifacts

- Add a schedule “pretty print” function and ensure examples print:
  - total scheduled ops
  - load reuse hits / eliminated loads
  - peak SRAM bytes

---

## 5. Example / Demo (Week 3)

**Task:** Add a new demo script (recommended): `examples/mlp_scheduled.py`.

**Behavior:**
- Build the MLP graph
- Run: Fusion → Tiling → Lowering → Scheduling
- Print:
  - Graph summary (existing)
  - uOp count (existing)
  - schedule op count
  - peak SRAM usage
  - first N scheduled ops

---

## 6. Deliverables

1. `src/compiler/scheduler/memory.py` — `VirtualSRAMArena` allocator + tile sizing helpers.
2. `src/compiler/scheduler/execution.py` — schedule IR (`SchedOp` variants) + scheduling logic.
3. `tests/test_scheduler.py` — allocator + scheduling tests.
4. (Recommended) `examples/mlp_scheduled.py` — end-to-end demo printing schedule + stats.

---

## 7. Design Decisions (Write Down in Code / Docstrings)

- **Determinism:** scheduling must be repeatable; eviction policy should not depend on hash iteration order.
- **Correctness first:** sequentially executable schedule is the baseline; overlap is represented via metadata (buffer ids), not actual concurrency yet.
- **Small stable IR:** keep schedule ops minimal so Week 4 runtime can bind to it cleanly.
- **Fail loudly:** if scheduling cannot fit in SRAM, raise an error that tells the user which tiles were live.

---

## 8. Acceptance Checklist (Done = Week 3 complete)

- Scheduling produces a schedule with concrete SRAM addresses.
- Schedule can be printed and inspected easily.
- Redundant loads are eliminated when tiles stay resident.
- Peak SRAM usage is computed and reported.
- Double buffering mode exists and is tested.
- Pytests pass (including new scheduler tests).
