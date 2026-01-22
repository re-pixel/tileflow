#!/usr/bin/env python3
"""
End-to-end MLP benchmark.

This benchmark measures the full pipeline performance, including:
- Python compilation overhead (graph building, passes, scheduling)
- C++ execution time

Usage:
    python benchmarks/mlp_e2e.py

Output:
    - Compilation time breakdown
    - Execution time per iteration
    - Total throughput analysis
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compiler.ir import Graph
from compiler.passes import FusionPass, LoweringPass, TilingPass
from compiler.runtime import Runtime
from compiler.scheduler import Scheduler, SRAMConfig

try:
    import mini_runtime as rt
except ImportError as e:
    print(f"Error: Could not import mini_runtime: {e}")
    print("Make sure to run 'pip install -e .' first")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

WARMUP_ITERS = 10
BENCH_ITERS = 100
SRAM_SIZE_KB = 256


def build_mlp_graph() -> Graph:
    """Build a 3-layer MLP graph for benchmarking."""
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


def compile_graph(graph: Graph, sram_kb: int) -> tuple:
    """
    Run the compiler pipeline and return schedule.
    
    Returns:
        (schedule, compilation_times_dict)
    """
    times = {}
    
    # Fusion pass
    start = time.perf_counter()
    fusion = FusionPass()
    fusion.run(graph)
    times["fusion"] = time.perf_counter() - start
    
    # Tiling pass
    start = time.perf_counter()
    tiling = TilingPass()
    tiling.run(graph)
    times["tiling"] = time.perf_counter() - start
    
    # Lowering pass
    start = time.perf_counter()
    lowering = LoweringPass()
    uops = lowering.run(graph)
    times["lowering"] = time.perf_counter() - start
    
    # Scheduling pass
    start = time.perf_counter()
    config = SRAMConfig(total_bytes=sram_kb * 1024)
    scheduler = Scheduler(config=config, double_buffer=False)
    schedule, stats = scheduler.run_on_graph(graph)
    times["scheduling"] = time.perf_counter() - start
    
    return schedule, stats, times


def benchmark_execution(
    runtime: Runtime,
    schedule: list,
    graph: Graph,
    iters: int,
    warmup: int,
) -> dict:
    """
    Benchmark schedule execution.
    
    Returns:
        Dictionary with timing metrics
    """
    # Generate random inputs
    np.random.seed(42)
    x_data = np.random.randn(128, 128).astype(np.float32) * 0.1
    w1_data = np.random.randn(128, 64).astype(np.float32) * 0.1
    w2_data = np.random.randn(64, 32).astype(np.float32) * 0.1
    w3_data = np.random.randn(32, 32).astype(np.float32) * 0.1
    
    # Warmup
    for _ in range(warmup):
        runtime.set_tensor("x", x_data)
        runtime.execute(schedule)
    
    # Benchmark
    times = []
    for i in range(iters):
        # Generate new random input each iteration
        x_data = np.random.randn(128, 128).astype(np.float32) * 0.1
        
        start = time.perf_counter()
        runtime.set_tensor("x", x_data)
        runtime.execute(schedule)
        end = time.perf_counter()
        
        times.append(end - start)
    
    times_ms = np.array(times) * 1000
    
    return {
        "total_ms": np.sum(times_ms),
        "mean_ms": np.mean(times_ms),
        "std_ms": np.std(times_ms),
        "min_ms": np.min(times_ms),
        "max_ms": np.max(times_ms),
        "p50_ms": np.percentile(times_ms, 50),
        "p99_ms": np.percentile(times_ms, 99),
    }


def compute_flops(graph: Graph) -> int:
    """Compute total FLOPs for the MLP graph."""
    # Layer 1: (128, 128) @ (128, 64) -> 2 * 128 * 128 * 64 = 2,097,152 FLOPs
    # Layer 2: (128, 64) @ (64, 32) -> 2 * 128 * 64 * 32 = 524,288 FLOPs  
    # Layer 3: (128, 32) @ (32, 32) -> 2 * 128 * 32 * 32 = 262,144 FLOPs
    # Total: 2,883,584 FLOPs
    # Note: ReLU ops add negligible FLOPs (128*64 + 128*32 comparisons)
    
    total = 0
    total += 2 * 128 * 128 * 64  # Layer 1
    total += 2 * 128 * 64 * 32   # Layer 2
    total += 2 * 128 * 32 * 32   # Layer 3
    return total


def main():
    """Run the end-to-end benchmark."""
    print("=" * 70)
    print("End-to-End MLP Benchmark")
    print("=" * 70)
    
    # Kernel info
    print(f"\nActive kernel: {rt.get_active_kernel_name()}")
    print(f"AVX2 available: {rt.is_avx2_available()}")
    print(f"SRAM size: {SRAM_SIZE_KB} KiB")
    
    # =========================================================================
    # Compilation Phase
    # =========================================================================
    print("\n" + "-" * 70)
    print("Compilation Phase")
    print("-" * 70)
    
    print("\nBuilding graph...")
    graph = build_mlp_graph()
    print(f"  Original ops: {len(graph.ops)}")
    
    print("\nRunning compiler passes...")
    schedule, sched_stats, compile_times = compile_graph(graph, SRAM_SIZE_KB)
    
    total_compile = sum(compile_times.values())
    print(f"\n  Compilation time breakdown:")
    print(f"    Fusion:     {compile_times['fusion']*1000:6.3f} ms")
    print(f"    Tiling:     {compile_times['tiling']*1000:6.3f} ms")
    print(f"    Lowering:   {compile_times['lowering']*1000:6.3f} ms")
    print(f"    Scheduling: {compile_times['scheduling']*1000:6.3f} ms")
    print(f"    ─────────────────────────")
    print(f"    Total:      {total_compile*1000:6.3f} ms")
    
    print(f"\n  Schedule statistics:")
    print(f"    Scheduled ops:    {sched_stats.sched_ops}")
    print(f"    Loads emitted:    {sched_stats.loads_emitted}")
    print(f"    Loads eliminated: {sched_stats.loads_eliminated}")
    print(f"    Peak SRAM:        {sched_stats.peak_sram_bytes / 1024:.1f} KiB")
    
    # =========================================================================
    # Runtime Setup
    # =========================================================================
    print("\n" + "-" * 70)
    print("Runtime Setup")
    print("-" * 70)
    
    runtime = Runtime(sram_bytes=SRAM_SIZE_KB * 1024)
    print(f"\n  Created runtime with {runtime.sram_bytes // 1024} KiB SRAM")
    
    # Register tensors
    runtime.register_graph_tensors(graph)
    
    # Set weight data (constant across iterations)
    np.random.seed(42)
    w1_data = np.random.randn(128, 64).astype(np.float32) * 0.1
    w2_data = np.random.randn(64, 32).astype(np.float32) * 0.1
    w3_data = np.random.randn(32, 32).astype(np.float32) * 0.1
    
    runtime.set_tensor("w1", w1_data)
    runtime.set_tensor("w2", w2_data)
    runtime.set_tensor("w3", w3_data)
    
    # =========================================================================
    # Execution Benchmark
    # =========================================================================
    print("\n" + "-" * 70)
    print("Execution Benchmark")
    print("-" * 70)
    
    print(f"\n  Warmup iterations: {WARMUP_ITERS}")
    print(f"  Benchmark iterations: {BENCH_ITERS}")
    print("\n  Running benchmark...")
    
    exec_metrics = benchmark_execution(
        runtime, schedule, graph, BENCH_ITERS, WARMUP_ITERS
    )
    
    print(f"\n  Execution time per iteration:")
    print(f"    Mean:   {exec_metrics['mean_ms']:6.3f} ms")
    print(f"    Std:    {exec_metrics['std_ms']:6.3f} ms")
    print(f"    Min:    {exec_metrics['min_ms']:6.3f} ms")
    print(f"    Max:    {exec_metrics['max_ms']:6.3f} ms")
    print(f"    p50:    {exec_metrics['p50_ms']:6.3f} ms")
    print(f"    p99:    {exec_metrics['p99_ms']:6.3f} ms")
    
    # =========================================================================
    # Performance Analysis
    # =========================================================================
    print("\n" + "-" * 70)
    print("Performance Analysis")
    print("-" * 70)
    
    total_flops = compute_flops(graph)
    throughput_gflops = (total_flops / (exec_metrics['mean_ms'] / 1000)) / 1e9
    
    print(f"\n  Total FLOPs per forward pass: {total_flops:,}")
    print(f"  Throughput: {throughput_gflops:.2f} GFLOP/s")
    
    # Execution stats from runtime
    stats = runtime.stats
    print(f"\n  Runtime statistics (last iteration):")
    print(f"    LOADs:  {stats.loads}")
    print(f"    EXECs:  {stats.executes}")
    print(f"    STOREs: {stats.stores}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    print(f"\n  Pipeline:")
    print(f"    Compilation time:     {total_compile*1000:6.2f} ms (one-time)")
    print(f"    Execution time (avg): {exec_metrics['mean_ms']:6.2f} ms/iter")
    print(f"    Effective throughput: {throughput_gflops:.2f} GFLOP/s")
    
    print(f"\n  Schedule efficiency:")
    total_loads = sched_stats.loads_emitted + sched_stats.loads_eliminated
    reuse_pct = 100 * sched_stats.loads_eliminated / total_loads if total_loads > 0 else 0
    print(f"    Load reuse: {sched_stats.loads_eliminated}/{total_loads} ({reuse_pct:.1f}%)")
    print(f"    Peak SRAM utilization: {100*sched_stats.peak_sram_bytes/(SRAM_SIZE_KB*1024):.1f}%")
    
    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
