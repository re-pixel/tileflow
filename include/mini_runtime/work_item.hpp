/**
 * @file work_item.hpp
 * @brief Work item definitions for the pipelined (threaded) engine.
 *
 * These structures are passed between the DMA thread and the compute
 * thread via lock-free ring buffers.
 *
 * ComputeWorkItem: DMA thread -> Compute thread (execute a matmul tile)
 * StoreNotification: Compute thread -> DMA thread (accumulator ready to store)
 */

#pragma once

#include <cstdint>

namespace mini_runtime {

/**
 * @struct ComputeWorkItem
 * @brief Describes a single tile matmul-accumulate operation.
 *
 * Pushed by the DMA thread after loading operands into SRAM.
 * Consumed by the compute thread which executes the kernel.
 *
 * When is_last_k is true, the compute thread will push a
 * StoreNotification using the embedded store_* fields.
 */
struct ComputeWorkItem {
    uint32_t a_addr;           ///< SRAM address of A tile
    uint32_t b_addr;           ///< SRAM address of B tile
    uint32_t acc_addr;         ///< SRAM address of accumulator
    uint32_t m;                ///< Output tile row index
    uint32_t n;                ///< Output tile column index
    uint32_t k;                ///< Reduction dimension tile index
    bool     is_first_k;       ///< If true, clear accumulator before compute
    bool     is_last_k;        ///< If true, signal accumulator ready after compute

    // Store metadata (valid only when is_last_k == true)
    uint32_t store_tensor_id;  ///< Destination tensor ID for STORE
    uint32_t store_tile_row;   ///< Destination tile row
    uint32_t store_tile_col;   ///< Destination tile column
    bool     store_apply_relu; ///< Apply ReLU before storing
};

/**
 * @struct StoreNotification
 * @brief Signals that an accumulator tile is complete and ready to store.
 *
 * Pushed by the compute thread after executing the final partial product
 * for an (m, n) output tile. Consumed by the DMA thread which performs
 * the STORE back to tensor storage.
 */
struct StoreNotification {
    uint32_t m;           ///< Accumulator tile row index
    uint32_t n;           ///< Accumulator tile column index
    uint32_t acc_addr;    ///< SRAM address to read from
    uint32_t tensor_id;   ///< Destination tensor ID
    uint32_t tile_row;    ///< Destination tile row in tensor
    uint32_t tile_col;    ///< Destination tile column in tensor
    bool     apply_relu;  ///< Apply ReLU activation before storing
};

}  // namespace mini_runtime
