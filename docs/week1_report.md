# Week 1 Report — Graph IR & Tiling API (mini-compiler)

**Date:** 2026-01-07  
**Phase:** Phase 1 (Compiler Frontend, Python)  
**Week 1 Goal:** Define a graph IR and implement a hardware-constraint-aware tiling API for a fixed $32\times 32$ FP32 accelerator.

---

## 1) Executive Summary

Week 1 delivered a complete, runnable “vertical slice” of the Python frontend:

- A minimal **Graph IR** with `Tensor` and `Op` nodes (`MatMul`, `ReLU`, `Add`).
- Strict **shape inference** (and strict validation) for correctness-first iteration.
- A **Hardware Constraint Validator** that:
  - enforces the $32\times 32$ tiling worldview,
  - automatically computes padding to multiples of 32,
  - annotates ops with stable tiling metadata.
- A demo script that builds a **3-layer MLP graph** and prints a “tiled graph” summary.
- A small pytest suite that locks in shape inference, tiling metadata, and graph invariants.

This establishes the foundation for Week 2 fusion + lowering.

---

## 2) Hard Constraints (Implemented / Reflected)

These constraints are treated as **design axioms** in the Week 1 code:

- **Tile size:** fixed $32\times 32$.
- **DType:** FP32 only.
- **No external math libs:** no BLAS; kernels will be implemented later.

You can see these constraints reflected directly in:

- `float32` being the only `DType` in `src/compiler/ir/dtypes.py`
- `MatMul.infer_output()` rejecting non-float32 dtypes
- tiling pass computing padding and tile counts based on tile size 32

---

## 3) What Was Implemented (Complete Inventory)

### 3.1 Python packaging / import ergonomics

**Motivation:** a compiler project moves faster when examples/tests can import modules consistently without fiddling with `PYTHONPATH`.

Implemented:

- `setup.py` (minimal `setuptools` package config)
  - Allows `pip install -e .` workflows later.
- `tests/conftest.py`
  - Adds `src/` to `sys.path` for test runs without installation.
- Package init modules:
  - `src/compiler/__init__.py`
  - `src/compiler/ir/__init__.py`
  - `src/compiler/passes/__init__.py`
  - `src/compiler/scheduler/__init__.py` (placeholder for future weeks)
- `requirements.txt`
  - Includes `pytest>=7` so the current tests run in a fresh env.

### 3.2 Core IR: tensors + ops + graph

#### `src/compiler/ir/dtypes.py`
- `DType(name, itemsize)` dataclass.
- `float32` constant (only supported dtype).

#### `src/compiler/ir/tensor.py`
- `Tensor(graph, name, shape, dtype)`
  - Tracks:
    - `producer: Op | None`
    - `users: list[Op]`
  - This is foundational for Week 2 fusion (pattern matching) and later scheduling.
- Helpers:
  - `Shape = tuple[int, ...]`
  - `as_shape(...)` coerces dims to `tuple[int, ...]`

#### `src/compiler/ir/op.py`
- Base `Op` with:
  - `name`, `inputs`, `attrs`, `output`
  - `infer_output()` API (shape inference + dtype inference)
- `IRValidationError` for validation failures.
- Ops implemented:
  - `MatMul`: supports only rank-2 tensors, checks $(M,K)@(K,N)$, dtype equality, and FP32 constraint.
  - `Add`: elementwise add, requires same shapes (no broadcasting yet).
  - `ReLU`: elementwise, output spec = input spec.

Important note documented in code:
- `Op.output` may be `None` for ops constructed manually.
- When building through `Graph`, the graph creates the output tensor and assigns it.

#### `src/compiler/ir/graph.py`
A “builder-style” graph, designed to be explicit and friendly to passes:

- Stores:
  - `ops: list[Op]` (append-only creation order)
  - `tensors: list[Tensor]`
  - `attrs: dict[str, object]` (graph-level metadata written by passes)
- Tensor creation:
  - `input(name, shape, dtype=float32)`
  - `param(name, shape, dtype=float32)`
- Op creation:
  - `matmul(a, b)`
  - `add(a, b)`
  - `relu(x)`

Key invariants enforced:

1) **No cross-graph mixing**
- `Graph._add_op` validates `t.graph is self` for every input tensor.
- Prevents subtle “two graphs accidentally fused” bugs.

2) **Better error context for shape inference**
- `Graph._add_op` wraps `op.infer_output()` failures and re-throws as `IRValidationError` with context:
  - graph name
  - op kind + op name
  - each input’s shape and dtype

3) Producer/user links
- When adding an op:
  - each input tensor records `add_user(op)`
  - output tensor sets `producer = op`

Debugging utility:
- `Graph.summary()` prints a readable listing of ops with input/output shapes.

### 3.3 Week 1 hardware constraint validator (tiling)

#### `src/compiler/passes/tiling.py`

Implemented:

- `_pad_to_multiple(x, multiple) -> (padded_x, pad_amount)`
  - Uses `pad = (-x) % multiple` to compute minimal non-negative padding.
- `_ceil_div(a, b)`
- `HardwareConstraintValidator(tile_m=32, tile_n=32, tile_k=32)`
  - Iterates graph ops and for each `MatMul`:
    - reads shapes $(M,K)$ and $(K,N)$ from inputs
    - computes padded dims to multiples of 32
    - computes tile grid counts:
      - `tiles_m = ceil(m_padded / 32)`
      - `tiles_n = ceil(n_padded / 32)`
      - `tiles_k = ceil(k_padded / 32)`
    - writes metadata into `op.attrs` (see “Stable metadata contract” below)
  - Marks the graph as validated:
    - `graph.attrs["tiling"] = {"validated": True, "tile": {...}}`

Design decision: this pass is **analysis/annotation** only.
- It does not rewrite the graph yet.
- It computes metadata needed for Week 2 lowering.

Important coupling decisions addressed:
- The tiling validator does **not** require `op.output` to exist.
  - It only uses input shapes.
  - This makes it robust even if ops are constructed manually (though typical usage is via `Graph`).

### 3.4 Week 1 deliverable example

#### `examples/mlp_tiled.py`

A runnable demo that:

- Builds a 3-layer MLP-like graph:
  - `MatMul -> ReLU -> MatMul -> ReLU -> MatMul`
- Prints:
  - `Graph.summary()`
  - “Tiled Graph” lines from the tiling pass

The example currently demonstrates both:
- a clean multiple-of-32 case (e.g. 128)
- a padding-required case (e.g. 130)

### 3.5 Tests (Week 1 correctness locks)

#### `tests/test_ir.py`
Covers:
- `MatMul` shape inference
- elementwise `Add`/`ReLU` shape preservation
- graph invariant: cannot mix tensors from different graphs

#### `tests/test_integration.py`
Covers:
- tiling metadata for a 128×128×128 matmul:
  - no padding
  - expected tile counts
  - graph-level `attrs["tiling"]` flag
- tiling metadata for a non-multiple case (33×65 @ 65×35):
  - expected padded dims
  - expected pad amounts
  - expected tile counts

Test infrastructure:
- `tests/conftest.py` adds `src/` to `sys.path`.

---

## 4) Stable Metadata Contracts (Important for Week 2+)

### 4.1 Per-op tiling metadata

For each `MatMul` op, the validator writes:

- `op.attrs["tile"]`:

```python
{"m": 32, "n": 32, "k": 32}
```

- `op.attrs["matmul"]` (shape + tiling data):

```python
{
  "m": M,
  "n": N,
  "k": K,
  "m_padded": M_pad,
  "n_padded": N_pad,
  "k_padded": K_pad,
  "pad": {"m": dM, "n": dN, "k": dK},
  "tiles": {"m": tM, "n": tN, "k": tK},
  "output_tile_ops": tM * tN,
  "mac_tile_ops": tM * tN * tK,
}
```

Interpretation:
- `output_tile_ops` = how many $32\times 32$ output tiles exist.
- `mac_tile_ops` = how many $32\times 32$ tile-level compute ops exist if you unroll the K loop.

This is intentionally the information Week 2 lowering will consume.

### 4.2 Graph-level validation metadata

After running the validator, the graph is marked:

```python
graph.attrs["tiling"] = {
  "validated": True,
  "tile": {"m": 32, "n": 32, "k": 32}
}
```

This enables future passes to check preconditions:
- Lowering can refuse to run if `graph.attrs.get("tiling")` is missing.

---

## 5) Important Design Decisions (and Why)

### Decision A: Keep IR small and strict
- No broadcasting, no rank-N matmul, no multiple dtypes.
- This reduces degrees of freedom so tiling/lowering/scheduling can be built cleanly.

### Decision B: Track producer/users explicitly
- Makes pattern matching for fusion straightforward (Week 2).
- Provides a natural bridge to liveness/scheduling concepts later.

### Decision C: Use `attrs` for pass outputs
- Early pipeline stages are easier when passes annotate instead of rewriting.
- Rewriting is reserved for when it adds real value (fusion, lowering).

### Decision D: Fail loudly with context
- Shape inference failures now report which op and which inputs were involved.
- Graph prevents cross-graph tensor mixing.

### Decision E: Tiling validator doesn’t require `op.output`
- This pass is purely shape-driven.
- Keeping it independent of output construction makes it more reusable and less brittle.

---

## 6) How To Run (Week 1)

### Run the demo
From repo root:

```bash
python -m examples.mlp_tiled
```

### Run tests

```bash
python -m pytest -q
```

---

## 7) Known Limitations (Intentional for Week 1)

- No real tensor storage or numeric execution in Python.
- No bias add in the MLP example (easy Week 1/2 extension).
- No graph rewriting passes yet (`fusion.py` and `lowering.py` are still placeholders).
- No scheduler, schedule format, runtime, or kernels implemented yet.

These are deferred to Weeks 2–6 by design.

---

## 8) Week 2 Entry Point (What Week 1 Enables)

Week 1’s work establishes the preconditions for Week 2:

- **Fusion** can use `Tensor.producer` and `Tensor.users` to identify patterns like `MatMul -> ReLU`.
- **Lowering** can consume the tiling metadata in `op.attrs["matmul"]` to emit $32\times 32$ micro-ops.
- **Pass ordering** can use `graph.attrs["tiling"]["validated"]` to ensure tiling ran before lowering.

---

## 9) Files Touched / Added (At a Glance)

- Packaging / infra:
  - `setup.py`
  - `requirements.txt`
  - `tests/conftest.py`
  - `src/compiler/__init__.py`
  - `src/compiler/ir/__init__.py`
  - `src/compiler/passes/__init__.py`
  - `src/compiler/scheduler/__init__.py`

- IR:
  - `src/compiler/ir/dtypes.py`
  - `src/compiler/ir/tensor.py`
  - `src/compiler/ir/op.py`
  - `src/compiler/ir/graph.py`

- Passes:
  - `src/compiler/passes/tiling.py`

- Demo:
  - `examples/mlp_tiled.py`
  - `examples/__init__.py`

- Tests:
  - `tests/test_ir.py`
  - `tests/test_integration.py`

- Meta context:
  - `agent.md`

---

## 10) Notes for Future Agents

When extending this repo:

- Preserve the tiling metadata schema; add fields only if needed and keep them backward-compatible.
- Prefer writing tests alongside new passes.
- Keep Week 2 fusion/lowering minimal: correctness and inspectability first.
