/**
 * @file schedule.hpp
 * @brief Schedule operation definitions for the runtime.
 *
 * These structures mirror the Python SchedOp types and define the
 * interface between the Python scheduler and C++ executor.
 */

#pragma once

#include <cstdint>
#include <variant>
#include <vector>

namespace mini_runtime {

/**
 * @struct SchedLoad
 * @brief Load a tile from DRAM (tensor storage) into SRAM.
 */
struct SchedLoad {
    uint32_t tensor_id;   ///< Index into tensor storage
    uint32_t tile_row;    ///< Tile row index
    uint32_t tile_col;    ///< Tile column index
    uint32_t dst_addr;    ///< Destination address in SRAM (bytes)
    uint32_t bytes;       ///< Number of bytes (always TILE_BYTES)
    int32_t  buffer;      ///< Buffer ID for double buffering (-1 if unused)

    SchedLoad(uint32_t tid, uint32_t tr, uint32_t tc,
              uint32_t addr, uint32_t b, int32_t buf = -1)
        : tensor_id(tid), tile_row(tr), tile_col(tc),
          dst_addr(addr), bytes(b), buffer(buf) {}
};

/**
 * @struct SchedExecute
 * @brief Execute a tile-level matrix multiplication accumulation.
 *
 * Performs: ACC[m,n] += A[m,k] @ B[k,n]
 * where all tiles are 32x32.
 */
struct SchedExecute {
    uint32_t m;           ///< Output tile row index
    uint32_t n;           ///< Output tile column index
    uint32_t k;           ///< Reduction dimension tile index
    uint32_t a_addr;      ///< SRAM address of A tile
    uint32_t b_addr;      ///< SRAM address of B tile
    uint32_t acc_addr;    ///< SRAM address of accumulator tile
    int32_t  buffer;      ///< Buffer ID for double buffering (-1 if unused)

    SchedExecute(uint32_t m_, uint32_t n_, uint32_t k_,
                 uint32_t a, uint32_t b, uint32_t acc, int32_t buf = -1)
        : m(m_), n(n_), k(k_),
          a_addr(a), b_addr(b), acc_addr(acc), buffer(buf) {}
};

/**
 * @struct SchedStore
 * @brief Store a tile from SRAM back to DRAM (tensor storage).
 */
struct SchedStore {
    uint32_t tensor_id;   ///< Index into tensor storage
    uint32_t tile_row;    ///< Tile row index
    uint32_t tile_col;    ///< Tile column index
    uint32_t src_addr;    ///< Source address in SRAM (bytes)
    uint32_t bytes;       ///< Number of bytes (always TILE_BYTES)
    bool     apply_relu;  ///< Apply ReLU activation before storing

    SchedStore(uint32_t tid, uint32_t tr, uint32_t tc,
               uint32_t addr, uint32_t b, bool relu = false)
        : tensor_id(tid), tile_row(tr), tile_col(tc),
          src_addr(addr), bytes(b), apply_relu(relu) {}
};

/// Variant type for schedule operations
using SchedOp = std::variant<SchedLoad, SchedExecute, SchedStore>;

}  // namespace mini_runtime
