# Week 2 Plan — Operator Fusion & Lowering (mini-compiler)

**Focus:** Graph Transformations (Fusion) and Lowering to Micro-Ops.
**Pre-requisites:** Completed Week 1 (Graph IR, Shape Inference, Tiling Validator).

---

## 1. Objectives

1.  **Operator Fusion:** implement a pass to detect `MatMul` followed immediately by `ReLU` and replace them with a single `FusedMatMulReLU` op. This reduces memory traffic by performing the activation in registers/cache before writing back to memory.
2.  **IR Support:** Extend the IR to support fused operators.
3.  **Tiling Support:** Ensure the Tiling Validator correctly handles fused ops (inherits tiling logic from MatMul).
4.  **Lowering:** Implement the first stage of lowering—translating high-level Graph Ops into a sequence of abstract **Micro-Ops (uOps)** representing tile-level operations (Load, Compute, Store).

---

## 2. Implementation Steps

### 2.1. IR Extensions (`src/compiler/ir/op.py`)

*   **Task:** Add `FusedMatMulReLU` class.
*   **Details:**
    *   Inherit from `MatMul` (or `Op` if inheritance is messy, but it shares `MatMul` logic).
    *   Implement `infer_output` (same as MatMul).
    *   Convention: `FusedMatMulReLU` represents $(A \times B).relu()$.

### 2.2. Fusion Pass (`src/compiler/passes/fusion.py`)

*   **Task:** Implement `FusionPass` class.
*   **Algorithm:**
    1.  Iterate over the graph ops.
    2.  Look for the pattern:
        *   Producer `P` is `MatMul`.
        *   Producer's output `T` has exactly **one** user.
        *   User `U` is `ReLU`.
    3.  **Rewrite:**
        *   Create `NewOp = FusedMatMulReLU(inputs=P.inputs)`.
        *   Wire `NewOp` to output to `U.output`.
        *   Update `U.output.producer` to `NewOp`.
        *   Update `P.inputs` users to point to `NewOp` instead of `P`.
        *   Remove `P` and `U` from the graph, insert `NewOp`.
*   **Validation:** Ensure `Tensor` producer/user links remain consistent.

### 2.3. Update Tiling Validator (`src/compiler/passes/tiling.py`)

*   **Task:** Update `HardwareConstraintValidator` to recognize `FusedMatMulReLU`.
*   **Details:**
    *   The tiling logic for `FusedMatMulReLU` is identical to `MatMul`.
    *   Ensure `op.attrs["matmul"]` and `op.attrs["tile"]` are populated for fused ops too.

### 2.4. Lowering Definitions & Pass (`src/compiler/passes/lowering.py`)

*   **Task:** Define Micro-Ops (uOps) and implement lowering.
*   **uOp Definitions:**
    *   `UOp`: Base class.
    *   `UOpLoad`: Logical load of a tile (e.g., "Load Tile (0,0) of Tensor A").
    *   `UOpStore`: Logical store of a tile.
    *   `UOpExecute`: Logical compute (MatMul) on tiles.
*   **Pass Logic (`LoweringPass`):**
    *   Input: `Graph` (fused and tiled).
    *   Output: A list of `uOps` (flat sequence) or attached to the graph.
    *   **Loop Generation:**
        *   For each `FusedMatMulReLU` (or `MatMul`):
        *   Read `op.attrs["matmul"]` for grid dimensions $(M_{tiles}, N_{tiles}, K_{tiles})$.
        *   Generate a triple loop over tiles ($m, n, k$).
        *   Emit uOps:
            *   `Load A[m, k]`
            *   `Load B[k, n]`
            *   `Execute(accumulating into C[m, n])`
            *   (After k-loop) `Store C[m, n]`
*   **Note:** We are **not** doing memory allocation (SRAM addresses) yet. That is Week 3 (Scheduler). For now, use logical coordinates or placeholder IDs.

---

## 3. Testing & Verification

### 3.1. Unit Tests (`tests/`)

*   `tests/test_fusion.py`:
    *   Construct a graph with `MatMul -> ReLU`.
    *   Run `FusionPass`.
    *   Assert graph has 1 op (`FusedMatMulReLU`) instead of 2.
    *   Assert shapes and connections are preserved.
    *   Test case: `MatMul -> ReLU` where `MatMul` output is used by *two* ops (should **not** fuse).
*   `tests/test_lowering.py`:
    *   Run lowering on a single Tiled MatMul.
    *   Assert the number of generated uOps matches expectation ($Tiles_M \times Tiles_N \times Tiles_K \times \dots$).

### 3.2. Integration Demo (`examples/mlp_fused.py`)

*   Create `examples/mlp_fused.py`.
*   Build the 3-layer MLP.
*   Run: `FusionPass` -> `TilingPass` -> `LoweringPass`.
*   Print:
    *   Fused Graph Summary.
    *   Total uOps generated.

---

## 4. Deliverables

1.  `src/compiler/passes/fusion.py` (Implementation)
2.  `src/compiler/passes/lowering.py` (Implementation)
3.  `tests/test_fusion.py` (Verification)
4.  `examples/mlp_fused.py` (Demo)

---

## 5. Design Decisions

*   **Fusion Strategy:** We only fuse if the intermediate tensor is not used elsewhere ("single-user constraint"). This is a simplification to avoid re-computing data.
*   **Lowering Output:** We will produce a flat list of uOps. This list assumes infinite registers/memory for now. The Week 3 Scheduler will take this list (or the Ops) and assign real SRAM locations and handle eviction (double buffering).
*   **uOp Level:** uOps are "Tile Ops", not scalar instructions. 1 `UOpExecute` = $32\times 32$ MACs.

