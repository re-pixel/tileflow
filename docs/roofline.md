# Roofline Model & Performance Analysis

This document explains the performance characteristics of the mini-compiler's SIMD kernels using the roofline model framework.

---

## 1. What is the Roofline Model?

The roofline model is a visual performance analysis framework that helps understand whether a kernel is:
- **Compute-bound**: Limited by the CPU's floating-point throughput
- **Memory-bound**: Limited by memory bandwidth

The model plots achievable performance (FLOP/s) against **arithmetic intensity** (FLOP/byte), creating a "roofline" that represents the maximum achievable performance at any given arithmetic intensity.

```
Performance (GFLOP/s)
    ^
    |          ______________________ Peak Compute (56 GFLOP/s)
    |         /
    |        /  Memory-Bound Region
    |       /
    |      /   Compute-Bound Region
    |     /
    |    /
    +---/---------------------------------> Arithmetic Intensity (FLOP/byte)
       Ridge Point
```

**Key Concepts:**

- **Peak Compute** ($\pi$): Maximum FLOP/s the CPU can sustain (horizontal ceiling)
- **Memory Bandwidth** ($\beta$): Maximum bytes/second the memory system can deliver
- **Arithmetic Intensity** (AI): Ratio of FLOPs to bytes transferred ($\text{AI} = \frac{\text{FLOPs}}{\text{Bytes}}$)
- **Ridge Point**: The AI where compute and memory constraints meet ($\text{Ridge} = \frac{\pi}{\beta}$)

---

## 2. Hardware Parameters

### 2.1 Compute Peak (AVX2 + FMA)

For a modern x86 CPU with AVX2 and FMA support:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Vector width | 256 bits | 8 × float32 per YMM register |
| FMA units | 2 | Most Haswell+ CPUs have 2 FMA ports |
| FLOPs per FMA | 2 | One multiply + one add |
| FLOPs per cycle | $2 \times 8 \times 2 = 32$ | Per core |

**Peak compute at 4.5 GHz:**
$$\pi = 4.5 \times 10^9 \times 32 = 144 \text{ GFLOP/s (single core)}$$

In practice, ~50-70% of theoretical peak is achievable due to:
- Instruction latency and dependencies
- Register pressure
- Memory access overhead

**Realistic target:** 56-80 GFLOP/s

### 2.2 Memory Bandwidth

| Memory Type | Theoretical BW | Practical BW |
|-------------|----------------|--------------|
| L1 Cache (per core) | ~1000 GB/s | ~500 GB/s |
| L2 Cache (per core) | ~300 GB/s | ~200 GB/s |
| L3 Cache (shared) | ~150 GB/s | ~100 GB/s |
| DDR4-2400 (dual channel) | 38.4 GB/s | ~30 GB/s |

For our 32×32 tiles fitting entirely in L1:
- **Effective bandwidth:** ~500 GB/s (L1 bound)
- **Ridge point (L1):** $\frac{80}{500} \approx 0.16$ FLOP/byte

Since our tiles fit in L1, we're firmly in the **compute-bound** region.

---

## 3. 32×32 Tile Matmul Analysis

### 3.1 Operation Counts

For a 32×32 tile matmul ($C_{32 \times 32} \mathrel{+}= A_{32 \times 32} \times B_{32 \times 32}$):

| Metric | Value | Calculation |
|--------|-------|-------------|
| Multiply-accumulate ops | 32,768 | $32 \times 32 \times 32$ |
| FLOPs (multiply + add) | 65,536 | $2 \times 32^3$ |

### 3.2 Memory Traffic

| Data Movement | Bytes | Notes |
|---------------|-------|-------|
| Load A | 4,096 | $32 \times 32 \times 4$ |
| Load B | 4,096 | $32 \times 32 \times 4$ |
| Load C (initial) | 4,096 | Read-modify-write |
| Store C (final) | 4,096 | Write back result |
| **Total** | **16,384** | |

### 3.3 Arithmetic Intensity

$$\text{AI} = \frac{65{,}536 \text{ FLOPs}}{16{,}384 \text{ bytes}} = 4.0 \text{ FLOP/byte}$$

This is well above the L1 ridge point (~0.16 FLOP/byte), confirming the kernel is **compute-bound**.

### 3.4 Cache Behavior

Each tile is 4 KiB ($32 \times 32 \times 4$ bytes). With 3 tiles active (A, B, C):
- Total working set: 12 KiB
- L1 cache size: 32-48 KiB (typical)
- **Result:** All tiles fit in L1 with room to spare

---

## 4. Micro-kernel Design

### 4.1 Register Allocation (6×16 Micro-kernel)

AVX2 provides 16 YMM registers (256 bits each, holding 8 floats).

| Purpose | Registers | Count |
|---------|-----------|-------|
| Accumulators (6 rows × 2 vectors) | c00-c01, c10-c11, ..., c50-c51 | 12 |
| A broadcasts | a0 (reused) | 1 |
| B loads | b0, b1 | 2 |
| Spare | (address computation) | 1 |
| **Total** | | **16** |

This design maximizes register utilization without spilling to memory.

### 4.2 Instruction Mix per K-iteration

For each value of k in the reduction loop:

| Instruction Type | Count | Notes |
|------------------|-------|-------|
| `vbroadcastss` | 6 | Broadcast A[i,k] for each row |
| `vmovups` (load) | 2 | Load B[k, 0:8] and B[k, 8:16] |
| `vfmadd231ps` | 12 | 6 rows × 2 accumulators |
| **Total** | **20** | Per k-iteration |

With k-loop unrolling by 4, the inner loop has 80 instructions per 4 k-values.

### 4.3 Expected Throughput

- 32 k-iterations × 12 FMAs = 384 FMA instructions per 6×16 block
- Each FMA = 16 FLOPs (8 elements × 2 FLOPs)
- Total FLOPs per 6×16 block = 6,144

Processing the full 32×32 tile:
- 2 column blocks (0-15, 16-31)
- ~5.33 row blocks (32/6)
- Total FMAs ≈ 2,048

**Theoretical minimum cycles:** 2,048 / 2 = 1,024 (2 FMA units)

At 4.5 GHz: $1{,}024 / 4.5 \times 10^9 \approx 0.23 \mu s$ per tile

**Realistic expectation:** 0.5-1.0 µs per tile (accounting for overhead)

---

## 5. Performance Expectations

### 5.1 Reference Kernel

The reference kernel uses scalar operations:
- 1 FLOP per cycle (no vectorization)
- Expected: ~4.5 GFLOP/s at 4.5 GHz

### 5.2 AVX2 Kernel

With the 6×16 micro-kernel design:
- 16 FLOPs per FMA instruction
- 2 FMA instructions per cycle (dual FMA units)
- Expected: **30-50 GFLOP/s** (realistic)
- Theoretical max: 144 GFLOP/s

### 5.3 Speedup

$$\text{Speedup} = \frac{\text{AVX2 throughput}}{\text{Reference throughput}} \approx \frac{40}{4.5} \approx 9\times$$

Realistic range: **4-10× speedup** depending on:
- CPU model and microarchitecture
- Compiler optimization level
- Memory subsystem state

---

## 6. Measurement Methodology

### 6.1 Isolated Kernel Benchmark

To measure kernel performance accurately:

1. **Allocate aligned memory** (64-byte alignment for cache lines)
2. **Warm up** the cache (10+ iterations)
3. **Time many iterations** (1000+) to amortize overhead
4. **Use high-resolution timer** (`std::chrono::high_resolution_clock`)

```python
# Example timing code
for _ in range(WARMUP):
    matmul_tile_bench(C, A, B, impl)

start = time.perf_counter()
for _ in range(ITERS):
    matmul_tile_bench(C, A, B, impl)
elapsed = time.perf_counter() - start

gflops = (FLOPS_PER_TILE * ITERS) / elapsed / 1e9
```

### 6.2 End-to-End Benchmark

For full pipeline performance:

1. **Separate compilation from execution** (compile once, run many times)
2. **Measure with different input data** each iteration
3. **Report percentiles** (p50, p99) not just mean

---

## 7. Optimization Opportunities

### 7.1 Current Implementation

| Technique | Status | Benefit |
|-----------|--------|---------|
| AVX2 vectorization | ✓ | 8× parallelism |
| FMA instructions | ✓ | 2× FLOPs/instruction |
| Register blocking | ✓ | Minimizes loads |
| K-loop unrolling | ✓ | Hides latency |

### 7.2 Future Optimizations (Not in Week 5)

| Technique | Potential Benefit | Complexity |
|-----------|-------------------|------------|
| AVX-512 | 2× vector width | Medium |
| B-panel packing | Better memory access | Medium |
| Software prefetching | Hide L2/L3 latency | Low |
| Multi-threading | Linear scaling | High |

---

## 8. References

1. **Williams, S., Waterman, A., & Patterson, D. (2009).** "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *Communications of the ACM*.

2. **Agner Fog's Optimization Manuals.** https://www.agner.org/optimize/

3. **Intel Intrinsics Guide.** https://www.intel.com/content/www/us/en/docs/intrinsics-guide/

4. **BLIS Framework.** https://github.com/flame/blis - Reference for micro-kernel design.

---

## 9. Appendix: Key Formulas

### Roofline Equation
$$P = \min(\pi, \beta \times \text{AI})$$

Where:
- $P$ = Achieved performance (FLOP/s)
- $\pi$ = Peak compute (FLOP/s)
- $\beta$ = Memory bandwidth (byte/s)
- $\text{AI}$ = Arithmetic intensity (FLOP/byte)

### Efficiency
$$\text{Efficiency} = \frac{P_{\text{achieved}}}{\pi} \times 100\%$$

### Arithmetic Intensity (General Matrix Multiply)
For $C_{M \times N} \mathrel{+}= A_{M \times K} \times B_{K \times N}$:

$$\text{AI} = \frac{2MNK}{4(MK + KN + MN)} = \frac{MNK}{2(MK + KN + MN)}$$

For square matrices ($M = N = K$):
$$\text{AI} = \frac{N^3}{6N^2} = \frac{N}{6}$$

For our 32×32 tiles: $\text{AI} = 32/6 \approx 5.3$ (close to our measured 4.0 when accounting for load/store of C).
