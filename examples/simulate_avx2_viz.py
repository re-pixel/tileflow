#!/usr/bin/env python3
"""
AVX2 Kernel Simulation / Visualization
This script mimics the exact logic of src/runtime/kernels/matmul_avx2.cpp
but prints out the "register states" to help visualize the data flow.
"""

import numpy as np

# Constants from C++ code
TILE_DIM = 32

class AVX2Simulator:
    def __init__(self):
        # Create simple 32x32 matrices with integers for readability
        self.A = np.random.randint(1, 4, (TILE_DIM, TILE_DIM)).astype(np.float32)
        self.B = np.random.randint(1, 4, (TILE_DIM, TILE_DIM)).astype(np.float32)
        self.C = np.zeros((TILE_DIM, TILE_DIM), dtype=np.float32)
        
        # Colors for terminal output
        self.BLUE = '\033[94m'
        self.GREEN = '\033[92m'
        self.YELLOW = '\033[93m'
        self.RESET = '\033[0m'

    def format_vec(self, vec):
        """Format a list/array as a vector string [x, x, x...]"""
        return "[" + ", ".join(f"{x:2.0f}" for x in vec) + "]"

    def microkernel_sim(self, start_row, start_col, num_rows):
        """
        Simulates: template <int NumRows> inline void microkernel_avx2(...)
        """
        print(f"\n{self.BLUE}=== Micro-kernel Start (Rows {start_row}-{start_row+num_rows-1}, Cols {start_col}-{start_col+15}) ==={self.RESET}")
        
        # 1. LOAD ACCUMULATORS
        # In C++: __m256 c00 = _mm256_loadu_ps(...)
        # We simulate 16-wide columns by splitting into two 8-wide "registers"
        # c_regs[row][0] is the first 8 floats, c_regs[row][1] is next 8
        c_regs = [] 
        for r in range(num_rows):
            # Load from C
            row_idx = start_row + r
            # Vector 0 (cols 0-7 relative to start_col)
            v0 = self.C[row_idx, start_col : start_col+8].copy()
            # Vector 1 (cols 8-15 relative to start_col)
            v1 = self.C[row_idx, start_col+8 : start_col+16].copy()
            c_regs.append([v0, v1])

        # 2. K-LOOP (Accumulation)
        # In C++: for (uint32_t k = 0; k < TILE_DIM; k += 4)
        for k in range(0, TILE_DIM, 4):
            # We will just visualize the first step of the K loop to avoid spam
            verbose = (start_row == 0 and start_col == 0 and k == 0)
            
            if verbose:
                print(f"\n  {self.YELLOW}--- Processing K-block {k} to {k+3} ---{self.RESET}")

            # Unroll loop (kk=0..3)
            for kk in range(4):
                current_k = k + kk
                
                # Load B vectors
                # In C++: __m256 b0 = _mm256_loadu_ps(&B[k_idx * TILE_DIM + j_start]);
                #         __m256 b1 = _mm256_loadu_ps(&B[k_idx * TILE_DIM + j_start + 8]);
                b_vec0 = self.B[current_k, start_col : start_col+8]
                b_vec1 = self.B[current_k, start_col+8 : start_col+16]

                if verbose:
                    print(f"    [k={current_k}] Loaded B vectors (Row {current_k} of B)")
                    print(f"      b0 (cols {start_col}-{start_col+7}):   {self.format_vec(b_vec0)}")
                    print(f"      b1 (cols {start_col+8}-{start_col+15}):  {self.format_vec(b_vec1)}")

                # Process each row
                for r in range(num_rows):
                    row_idx = start_row + r
                    
                    # In C++: __m256 a_val = _mm256_broadcast_ss(&A[(i_start + r) * TILE_DIM + k_idx]);
                    a_val = self.A[row_idx, current_k]
                    
                    # FMA: c += a * b
                    # In C++: c00 = _mm256_fmadd_ps(a0, b0, c00);
                    c_regs[r][0] += a_val * b_vec0
                    c_regs[r][1] += a_val * b_vec1

                    if verbose and r < 2: # Only show first 2 rows
                        print(f"      {self.GREEN}Row {row_idx} Update:{self.RESET}")
                        print(f"        Broadcast A[{row_idx},{current_k}] = {a_val:.0f}")
                        print(f"        FMA: c_reg += {a_val:.0f} * b_vec")
                        print(f"        New Accumulator (first 8): {self.format_vec(c_regs[r][0])}")

        # 3. STORE RESULTS
        # In C++: _mm256_storeu_ps(...)
        for r in range(num_rows):
            row_idx = start_row + r
            self.C[row_idx, start_col : start_col+8] = c_regs[r][0]
            self.C[row_idx, start_col+8 : start_col+16] = c_regs[r][1]
        
        print(f"{self.BLUE}=== Micro-kernel Finished ==={self.RESET}")

    def run(self):
        print("Starting AVX2 Tile Matmul Simulation")
        print("Note: Showing detailed trace for the FIRST micro-kernel call only.")
        print("-" * 60)

        # Main Tiling Loop
        # 1. Process column block [0:16]
        # In C++: for (int i = 0; i < 30; i += 6) microkernel_avx2<6>(C, A, B, i, 0);
        #         microkernel_avx2<2>(C, A, B, 30, 0);
        
        col_start = 0
        # Rows 0-5
        self.microkernel_sim(0, col_start, 6)
        # Rows 6-11 (Silent)
        self.microkernel_sim(6, col_start, 6)
        # ... Skipping middle rows for simulation brevity ...
        
        # 2. Process column block [16:32]
        # Same loops for second half of columns
        pass

        print("\nSimulation visualization structure:")
        print("1. We loaded 6 rows of C into registers.")
        print("2. We iterated K from 0..31.")
        print("3. In each step, we loaded a vector form B, and broadcast a scalar from A.")
        print("4. This reused the B-vector 6 times (once for each active row).")

if __name__ == "__main__":
    sim = AVX2Simulator()
    sim.run()
