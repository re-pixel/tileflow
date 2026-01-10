# Week 2 Report — Operator Fusion & Lowering (mini-compiler)

**Date:** 2026-01-10
**Phase:** Phase 1 (Compiler Frontend, Python)
**Week 2 Goal:** Implement graph-level optimizations (kernel fusion) and lower high-level graph operators into a sequence of tile-level micro-operations (uOps).

---

## 1) Executive Summary

Week 2 successfully extended the compiler frontend to support **Operator Fusion** and **Lowering**:

- **Graph Substitution:** Implemented a robust `FusionPass` that identifies `MatMul` -> `ReLU` patterns and seamlessly replaces them with `FusedMatMulReLU` nodes, ensuring tensor connectivity is preserved.
- **Lowering Engine:** Built the `LoweringPass` which translates the tiled graph into a flat sequence of **Micro-Ops (uOps)**.
- **uOp Abstraction:** Defined the low-level ISA interface: `UOpLoad`, `UOpExecute`, and `UOpStore`.
- **Validation:** Updated the existing Tiling Validator to transparently handle fused operators.
- **Verification:** Delivered a new `examples/mlp_fused.py` demo that runs the full pipeline (Graph -> Fusion -> Tiling -> Lowering) and printed the resulting instruction stream.

This completes the *what to run* phase. Week 3 will focus on *how/when to run it* (Scheduling & Memory Planning).

---

## 2) Key Implementation Details

### 2.1 Operator Fusion (`src/compiler/passes/fusion.py`)

A pattern-matching pass that rewrites the graph structure.

*   **Logic:** Scans for `MatMul` nodes whose output is consumed *only* by a `ReLU` node.
*   **Result:** Replaces the pair with a single `FusedMatMulReLU` op.
*   **Why:** This prepares for later codegen where the ReLU can be computed in registers immediately after the matrix multiplication accumulation, saving a round-trip to memory.

### 2.2 Micro-Op Lowering (`src/compiler/passes/lowering.py`)

The bridge between graph "nodes" and an executable instruction stream.

*   **Input:** A Tiled Graph (with metadata identifying $32\times 32$ tile counts).
*   **Output:** A linear list of `UOp` objects.
*   **Strategy:** Naive unroll. It iterates over the tile grid ($M_{tiles}, N_{tiles}, K_{tiles}$) and emits:
    *   `LOAD A[m, k]`
    *   `LOAD B[k, n]`
    *   `EXEC [m, n, k]`
    *   `STORE C[m, n]` (after reduction completes)

### 2.3 uOp IR Definitions

We defined an abstract ISA for our $32\times 32$ accelerator:

| uOp | Description |
| :--- | :--- |
| `UOpLoad(tensor, coord)` | Loads a $32\times 32$ tile from backing memory into the accelerator. |
| `UOpExecute(m, n, k)` | Performs a $32\times 32$ matrix multiplication accumulation on loaded tiles. |
| `UOpStore(tensor, coord, activation)` | Stores a $32\times 32$ accumulator tile back to memory, optionally applying an activation function (ReLU) on the fly. |

---

## 3) Deliverables Check

| Deliverable | Status | Location |
| :--- | :--- | :--- |
| **Fusion Logic** | ✅ Complete | `src/compiler/passes/fusion.py` |
| **IR Extensions** | ✅ Complete | `src/compiler/ir/op.py` (`FusedMatMulReLU`) |
| **Lowering Logic** | ✅ Complete | `src/compiler/passes/lowering.py` |
| **Tests** | ✅ Complete | `tests/test_fusion.py`, `tests/test_lowering.py` |
| **End-to-End Demo** | ✅ Complete | `examples/mlp_fused.py` |

---

## 4) Example Output

Running `python -m examples.mlp_fused`:

```text
Building MLP...
Original: 5 ops (MatMul -> ReLU -> MatMul -> ReLU -> MatMul)

Running Fusion...
Fused: 3 ops
- Fused fused_mm1_relu1
- Fused fused_mm2_relu2
- mm3 (MatMul)

Running Lowering...
Generated 624 micro-ops.
First 10 uOps:
  LOAD x(0, 0)
  LOAD w1(0, 0)
  EXEC (0,0,0)
  LOAD x(0, 1)
  LOAD w1(1, 0)
  EXEC (0,0,1)
  ...
```

---

## 5) Look Ahead (Week 3)

With a flat stream of naive uOps now available, the next phase is **Scheduling & Memory Planning**.

*   **Problem:** The current lowered stream assumes infinite memory and re-loads data constantly.
*   **Goal:** Map abstract `uOps` to concrete SRAM addresses (e.g., `0x0000` to `0x8000`).
*   **Optimization:** Implement double-buffering and data reuse (don't reload if tile is already in SRAM).
