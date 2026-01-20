/**
 * @file tensors.hpp
 * @brief DRAM tensor storage for backing memory.
 *
 * TensorStorage manages the "off-chip" memory where full tensors reside.
 * LOAD operations copy tiles from here into SRAM, and STORE operations
 * copy tiles back.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mini_runtime/constants.hpp"

namespace mini_runtime {

/**
 * @class TensorStorage
 * @brief Manages tensor data in simulated DRAM.
 *
 * Tensors are stored in row-major order with padding to tile boundaries.
 * This ensures tile accesses are always aligned and in-bounds.
 */
class TensorStorage {
public:
    TensorStorage() = default;

    // Non-copyable, movable
    TensorStorage(const TensorStorage&) = delete;
    TensorStorage& operator=(const TensorStorage&) = delete;
    TensorStorage(TensorStorage&&) noexcept = default;
    TensorStorage& operator=(TensorStorage&&) noexcept = default;

    /**
     * @brief Register a tensor with the given name and shape.
     * @param name Unique tensor identifier.
     * @param rows Number of rows (will be padded to tile boundary).
     * @param cols Number of columns (will be padded to tile boundary).
     * @return Tensor ID for use in schedule operations.
     * @throws std::invalid_argument if name already registered or shape is 0.
     */
    uint32_t register_tensor(const std::string& name, size_t rows, size_t cols);

    /**
     * @brief Get a pointer to a specific tile within a tensor.
     * @param tensor_id ID returned by register_tensor.
     * @param tile_row Tile row index (0-based).
     * @param tile_col Tile column index (0-based).
     * @return Pointer to the start of the tile (TILE_DIM x TILE_DIM floats).
     * @throws std::out_of_range if indices are invalid.
     */
    float* tile_ptr(uint32_t tensor_id, uint32_t tile_row, uint32_t tile_col);
    const float* tile_ptr(uint32_t tensor_id, uint32_t tile_row, uint32_t tile_col) const;

    /**
     * @brief Set tensor data from external buffer.
     * @param name Tensor name.
     * @param data Source data pointer (row-major, unpadded).
     * @param rows Number of rows in source data.
     * @param cols Number of columns in source data.
     * @throws std::invalid_argument if name not found or size mismatch.
     */
    void set_data(const std::string& name, const float* data, size_t rows, size_t cols);

    /**
     * @brief Get tensor data into external buffer.
     * @param name Tensor name.
     * @param data Destination data pointer (row-major, unpadded).
     * @param rows Number of rows to copy.
     * @param cols Number of columns to copy.
     * @throws std::invalid_argument if name not found or size mismatch.
     */
    void get_data(const std::string& name, float* data, size_t rows, size_t cols) const;

    /**
     * @brief Look up tensor ID by name.
     * @param name Tensor name.
     * @return Tensor ID.
     * @throws std::invalid_argument if name not found.
     */
    uint32_t name_to_id(const std::string& name) const;

    /**
     * @brief Check if a tensor with the given name exists.
     */
    bool has_tensor(const std::string& name) const;

    /**
     * @brief Get the original (unpadded) shape of a tensor.
     */
    std::pair<size_t, size_t> get_shape(const std::string& name) const;

    /**
     * @brief Get the row stride (padded_cols) for a tensor.
     * @param tensor_id Tensor ID.
     * @return Number of floats between consecutive rows.
     */
    size_t get_stride(uint32_t tensor_id) const;

private:
    struct TensorInfo {
        std::string name;
        size_t rows;          // Original rows
        size_t cols;          // Original cols
        size_t padded_rows;   // Padded to tile boundary
        size_t padded_cols;   // Padded to tile boundary
        size_t tile_rows;     // Number of tile rows
        size_t tile_cols;     // Number of tile columns
        std::unique_ptr<float[], decltype(&std::free)> data;

        TensorInfo(std::string n, size_t r, size_t c);
    };

    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, uint32_t> name_to_id_;

    static size_t pad_to_tile(size_t dim);
};

}  // namespace mini_runtime
