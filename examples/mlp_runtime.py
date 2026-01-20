#!/usr/bin/env python3
"""Week 4 Demo — End-to-End MLP Execution with C++ Runtime.

This script demonstrates the complete Week 4 pipeline:
1. Build graph (IR)
2. Fusion pass (Week 2)
3. Tiling pass (Week 1)
4. Lowering pass (Week 2)
5. Scheduling pass (Week 3)
6. C++ Runtime Execution (Week 4)

Run with:
    python -m examples.mlp_runtime
"""

import numpy as np

from compiler.ir import Graph
from compiler.passes import FusionPass, LoweringPass, TilingPass
from compiler.runtime import Runtime, TILE_DIM, TILE_BYTES
from compiler.scheduler import (
    Scheduler,
    SRAMConfig,
    format_schedule,
    format_stats,
)


def build_mlp_graph() -> Graph:
    """Build a 3-layer MLP graph for demonstration."""
    g = Graph("mlp_3layer")
    
    # Input: batch=128, features=128
    x = g.input("x", (128, 128))
    
    # Layer 1: 128x128 @ 128x64 -> 128x64, with ReLU
    w1 = g.param("w1", (128, 64))
    h1 = g.matmul(x, w1, output_name="h1")
    a1 = g.relu(h1, output_name="a1")
    
    # Layer 2: 128x64 @ 64x32 -> 128x32, with ReLU
    w2 = g.param("w2", (64, 32))
    h2 = g.matmul(a1, w2, output_name="h2")
    a2 = g.relu(h2, output_name="a2")
    
    # Layer 3: 128x32 @ 32x32 -> 128x32 (no ReLU, output layer)
    w3 = g.param("w3", (32, 32))
    out = g.matmul(a2, w3, output_name="out")
    
    return g


def numpy_reference(x: np.ndarray, w1: np.ndarray, w2: np.ndarray, w3: np.ndarray) -> np.ndarray:
    """Compute MLP output using NumPy for reference."""
    h1 = np.maximum(0, x @ w1)   # Layer 1 with ReLU
    h2 = np.maximum(0, h1 @ w2)  # Layer 2 with ReLU
    out = h2 @ w3                # Layer 3 (no ReLU)
    return out


def main():
    print("=" * 70)
    print("Week 4 Demo — MLP Execution with C++ Runtime")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 1: Build Graph
    # -------------------------------------------------------------------------
    print("\n[1] Building MLP graph...")
    g = build_mlp_graph()
    print(f"    Original graph: {len(g.ops)} ops")
    print(f"    Tensors: {len(g.tensors)}")
    
    # -------------------------------------------------------------------------
    # Step 2: Fusion Pass
    # -------------------------------------------------------------------------
    print("\n[2] Running Fusion Pass...")
    fusion = FusionPass()
    fusion.run(g)
    print(f"    After fusion: {len(g.ops)} ops")
    
    for op in g.ops:
        print(f"      - {op.name} ({op.__class__.__name__})")
    
    # -------------------------------------------------------------------------
    # Step 3: Tiling Pass
    # -------------------------------------------------------------------------
    print("\n[3] Running Tiling Pass...")
    tiling = TilingPass()
    tiling.run(g)
    print(f"    Tiling validated: {g.attrs.get('tiling', {}).get('validated', False)}")
    
    # -------------------------------------------------------------------------
    # Step 4: Lowering Pass
    # -------------------------------------------------------------------------
    print("\n[4] Running Lowering Pass...")
    lowering = LoweringPass()
    uops = lowering.run(g)
    print(f"    Generated {len(uops)} micro-ops (uOps)")
    
    # -------------------------------------------------------------------------
    # Step 5: Scheduling Pass
    # -------------------------------------------------------------------------
    print("\n[5] Running Scheduling Pass...")
    
    sram_size_kb = 256
    config = SRAMConfig(total_bytes=sram_size_kb * 1024)
    print(f"    SRAM size: {sram_size_kb} KiB")
    
    scheduler = Scheduler(config=config, double_buffer=False)
    schedule, stats = scheduler.run_on_graph(g)
    
    print("\n" + "=" * 70)
    print("Schedule Statistics")
    print("=" * 70)
    print(format_stats(stats))
    
    # -------------------------------------------------------------------------
    # Step 6: C++ Runtime Execution (Week 4!)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("C++ Runtime Execution")
    print("=" * 70)
    
    # Create runtime
    rt = Runtime(sram_bytes=config.total_bytes)
    print(f"\n  Created runtime with {rt.sram_bytes // 1024} KiB SRAM")
    
    # Register all tensors
    print("\n  Registering tensors...")
    rt.register_graph_tensors(g)
    for name, shape in [("x", (128, 128)), ("w1", (128, 64)), 
                        ("w2", (64, 32)), ("w3", (32, 32))]:
        print(f"    - {name}: {shape}")
    
    # Set random inputs (with seed for reproducibility)
    print("\n  Setting input data...")
    np.random.seed(42)
    x_data = np.random.randn(128, 128).astype(np.float32) * 0.1
    w1_data = np.random.randn(128, 64).astype(np.float32) * 0.1
    w2_data = np.random.randn(64, 32).astype(np.float32) * 0.1
    w3_data = np.random.randn(32, 32).astype(np.float32) * 0.1
    
    rt.set_tensor("x", x_data)
    rt.set_tensor("w1", w1_data)
    rt.set_tensor("w2", w2_data)
    rt.set_tensor("w3", w3_data)
    
    # Execute schedule
    print(f"\n  Executing schedule ({len(schedule)} ops)...")
    rt.execute(schedule)
    
    # Get execution stats
    exec_stats = rt.stats
    print(f"\n  Execution complete!")
    print(f"    LOADs executed:  {exec_stats.loads}")
    print(f"    EXECs executed:  {exec_stats.executes}")
    print(f"    STOREs executed: {exec_stats.stores}")
    
    # -------------------------------------------------------------------------
    # Step 7: Verify Correctness
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Correctness Verification")
    print("=" * 70)
    
    # Get output from runtime
    output = rt.get_tensor("out", (128, 32))
    
    # Compute reference
    expected = numpy_reference(x_data, w1_data, w2_data, w3_data)
    
    # Compare
    max_diff = np.max(np.abs(output - expected))
    mean_diff = np.mean(np.abs(output - expected))
    
    print(f"\n  Output shape: {output.shape}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    
    # Check tolerance
    rtol = 1e-4
    atol = 1e-5
    is_close = np.allclose(output, expected, rtol=rtol, atol=atol)
    
    print(f"\n  Tolerance check (rtol={rtol}, atol={atol}):")
    print(f"    {'✓ PASS' if is_close else '✗ FAIL'}")
    
    # Sample outputs
    print("\n  Sample output values (first 5 elements of first row):")
    print(f"    Runtime: {output[0, :5]}")
    print(f"    NumPy:   {expected[0, :5]}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    print(f"\n  Pipeline stages:")
    print(f"    1. Graph ops:      {len(g.ops)}")
    print(f"    2. Lowered uOps:   {len(uops)}")
    print(f"    3. Scheduled ops:  {stats.sched_ops}")
    print(f"    4. C++ execution:  {exec_stats.loads + exec_stats.executes + exec_stats.stores} ops")
    
    print(f"\n  Memory efficiency:")
    print(f"    Load reuse:        {stats.loads_eliminated}/{stats.loads_emitted + stats.loads_eliminated} " +
          f"({100*stats.loads_eliminated/(stats.loads_emitted + stats.loads_eliminated):.1f}%)")
    print(f"    Peak SRAM:         {stats.peak_sram_bytes / 1024:.1f} KiB / {sram_size_kb} KiB")
    
    print(f"\n  Correctness: {'✓ VERIFIED' if is_close else '✗ FAILED'}")
    
    print("\n" + "=" * 70)
    print("Week 4 Demo Complete!")
    print("=" * 70)
    
    return 0 if is_close else 1


if __name__ == "__main__":
    exit(main())
