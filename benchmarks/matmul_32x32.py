#!/usr/bin/env python3
"""
Isolated 32x32 tile matmul kernel benchmark.

This benchmark measures the raw throughput of the tile matmul kernels,
comparing reference and AVX2 implementations.

Usage:
    python benchmarks/matmul_32x32.py

Output:
    - Average time per 1000 iterations
    - Throughput in GFLOP/s
    - Speedup of AVX2 vs reference
    - Efficiency relative to theoretical peak
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import mini_runtime as rt
except ImportError as e:
    print(f"Error: Could not import mini_runtime: {e}")
    print("Make sure to run 'pip install -e .' first")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

TILE_DIM = 32
WARMUP_ITERS = 100
BENCH_ITERS = 1000

# Theoretical peak calculation (adjust for your CPU)
# Configured for AMD Ryzen 7 7730U (Zen 3) - Base Clock used for realistic laptop expectations
# Base Clock: 2.0 GHz, 2 FMA units (256-bit), 8 floats/vector, 2 FLOPs/FMA
CPU_FREQ_GHZ = 2.0
FMA_UNITS = 2
FLOATS_PER_VECTOR = 8
FLOPS_PER_FMA = 2

THEORETICAL_PEAK_GFLOPS = CPU_FREQ_GHZ * FMA_UNITS * FLOATS_PER_VECTOR * FLOPS_PER_FMA


def get_cpu_info() -> dict:
    """Get basic CPU information from /proc/cpuinfo on Linux."""
    info = {"model": "Unknown", "freq_mhz": 0.0}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["model"] = line.split(":")[1].strip()
                elif line.startswith("cpu MHz"):
                    info["freq_mhz"] = float(line.split(":")[1].strip())
                    break
    except (FileNotFoundError, PermissionError):
        pass
    return info


def benchmark_kernel(impl: "rt.KernelImpl", iters: int) -> float:
    """
    Benchmark a kernel implementation.
    
    Args:
        impl: Kernel implementation to benchmark
        iters: Number of iterations
        
    Returns:
        Total time in seconds
    """
    # Allocate aligned arrays
    A = np.ascontiguousarray(np.random.randn(TILE_DIM, TILE_DIM).astype(np.float32))
    B = np.ascontiguousarray(np.random.randn(TILE_DIM, TILE_DIM).astype(np.float32))
    C = np.ascontiguousarray(np.zeros((TILE_DIM, TILE_DIM), dtype=np.float32))
    
    # Warmup
    for _ in range(WARMUP_ITERS):
        rt.matmul_tile_bench(C, A, B, impl)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        rt.matmul_tile_bench(C, A, B, impl)
    end = time.perf_counter()
    
    return end - start


def compute_metrics(total_time: float, iters: int) -> dict:
    """
    Compute performance metrics from benchmark results.
    
    Args:
        total_time: Total time in seconds
        iters: Number of iterations
        
    Returns:
        Dictionary with performance metrics
    """
    # FLOPs per tile matmul: 2 * 32 * 32 * 32 = 65536
    flops_per_tile = 2 * TILE_DIM * TILE_DIM * TILE_DIM
    total_flops = flops_per_tile * iters
    
    time_per_iter_us = (total_time / iters) * 1e6
    throughput_gflops = (total_flops / total_time) / 1e9
    
    return {
        "total_time_ms": total_time * 1000,
        "time_per_iter_us": time_per_iter_us,
        "throughput_gflops": throughput_gflops,
        "efficiency": throughput_gflops / THEORETICAL_PEAK_GFLOPS * 100,
    }


def verify_correctness():
    """Verify that AVX2 kernel produces correct results."""
    print("Verifying AVX2 kernel correctness...")
    
    A = np.ascontiguousarray(np.random.randn(TILE_DIM, TILE_DIM).astype(np.float32))
    B = np.ascontiguousarray(np.random.randn(TILE_DIM, TILE_DIM).astype(np.float32))
    
    # Reference result
    C_ref = np.zeros((TILE_DIM, TILE_DIM), dtype=np.float32)
    rt.matmul_tile_bench(C_ref, A, B, rt.KernelImpl.Reference)
    
    # AVX2 result
    C_avx2 = np.zeros((TILE_DIM, TILE_DIM), dtype=np.float32)
    rt.matmul_tile_bench(C_avx2, A, B, rt.KernelImpl.AVX2)
    
    # Compare
    max_diff = np.max(np.abs(C_ref - C_avx2))
    rel_diff = max_diff / (np.max(np.abs(C_ref)) + 1e-10)
    
    if rel_diff < 1e-5:
        print(f"  ✓ Results match (max relative diff: {rel_diff:.2e})")
        return True
    else:
        print(f"  ✗ Results differ (max relative diff: {rel_diff:.2e})")
        return False


def main():
    """Run the benchmark suite."""
    print("=" * 60)
    print("32x32 Tile Matmul Kernel Benchmark")
    print("=" * 60)
    
    # System info
    cpu_info = get_cpu_info()
    print(f"\nCPU: {cpu_info['model']}")
    if cpu_info["freq_mhz"] > 0:
        print(f"Current frequency: {cpu_info['freq_mhz']:.0f} MHz")
    print(f"Theoretical peak: {THEORETICAL_PEAK_GFLOPS:.1f} GFLOP/s "
          f"(assuming {CPU_FREQ_GHZ} GHz, {FMA_UNITS} FMA units)")
    
    # Kernel info
    print(f"\nActive kernel: {rt.get_active_kernel_name()}")
    print(f"AVX2 available: {rt.is_avx2_available()}")
    
    # Verify correctness first
    print()
    if not verify_correctness():
        print("\nAborting benchmark due to correctness failure.")
        sys.exit(1)
    
    # Benchmark reference kernel
    print(f"\nBenchmarking Reference kernel ({BENCH_ITERS} iterations)...")
    ref_time = benchmark_kernel(rt.KernelImpl.Reference, BENCH_ITERS)
    ref_metrics = compute_metrics(ref_time, BENCH_ITERS)
    
    print(f"  Time: {ref_metrics['total_time_ms']:.2f} ms total, "
          f"{ref_metrics['time_per_iter_us']:.2f} µs/iter")
    print(f"  Throughput: {ref_metrics['throughput_gflops']:.2f} GFLOP/s")
    print(f"  Efficiency: {ref_metrics['efficiency']:.1f}% of theoretical peak")
    
    # Benchmark AVX2 kernel
    print(f"\nBenchmarking AVX2 kernel ({BENCH_ITERS} iterations)...")
    avx2_time = benchmark_kernel(rt.KernelImpl.AVX2, BENCH_ITERS)
    avx2_metrics = compute_metrics(avx2_time, BENCH_ITERS)
    
    print(f"  Time: {avx2_metrics['total_time_ms']:.2f} ms total, "
          f"{avx2_metrics['time_per_iter_us']:.2f} µs/iter")
    print(f"  Throughput: {avx2_metrics['throughput_gflops']:.2f} GFLOP/s")
    print(f"  Efficiency: {avx2_metrics['efficiency']:.1f}% of theoretical peak")
    
    # Summary
    speedup = ref_time / avx2_time
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Reference:  {ref_metrics['throughput_gflops']:6.2f} GFLOP/s")
    print(f"AVX2:       {avx2_metrics['throughput_gflops']:6.2f} GFLOP/s")
    print(f"Speedup:    {speedup:.2f}x")
    print(f"\nTheoretical peak: {THEORETICAL_PEAK_GFLOPS:.1f} GFLOP/s")
    print(f"AVX2 efficiency:  {avx2_metrics['efficiency']:.1f}%")
    
    # Performance analysis
    print("\n" + "-" * 60)
    print("Performance Analysis")
    print("-" * 60)
    
    # Arithmetic intensity
    flops = 2 * TILE_DIM ** 3  # 65536
    bytes_loaded = 3 * TILE_DIM * TILE_DIM * 4  # A + B + C (initial read)
    bytes_stored = TILE_DIM * TILE_DIM * 4  # C (write back)
    total_bytes = bytes_loaded + bytes_stored
    arith_intensity = flops / total_bytes
    
    print(f"FLOPs per tile:        {flops:,}")
    print(f"Memory traffic:        {total_bytes:,} bytes")
    print(f"Arithmetic intensity:  {arith_intensity:.2f} FLOP/byte")
    print(f"Tile fits in L1:       Yes (4 KiB per tile, 12 KiB total)")
    print(f"Bound:                 Compute-bound (AI > 1.4 ridge point)")


if __name__ == "__main__":
    main()
