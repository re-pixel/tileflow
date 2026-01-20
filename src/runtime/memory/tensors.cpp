/**
 * @file tensors.cpp
 * @brief Tensor storage implementation.
 */

#include "mini_runtime/tensors.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace mini_runtime {

// Helper to compute padded dimension
size_t TensorStorage::pad_to_tile(size_t dim) {
    return ((dim + TILE_DIM - 1) / TILE_DIM) * TILE_DIM;
}

TensorStorage::TensorInfo::TensorInfo(std::string n, size_t r, size_t c)
    : name(std::move(n))
    , rows(r)
    , cols(c)
    , padded_rows(pad_to_tile(r))
    , padded_cols(pad_to_tile(c))
    , tile_rows(padded_rows / TILE_DIM)
    , tile_cols(padded_cols / TILE_DIM)
    , data(nullptr, &std::free)
{
    // Allocate aligned storage for padded tensor
    size_t total_bytes = padded_rows * padded_cols * sizeof(float);
    void* ptr = std::aligned_alloc(ALIGNMENT, total_bytes);
    if (!ptr) {
        throw std::bad_alloc();
    }
    data.reset(static_cast<float*>(ptr));

    // Zero-initialize (padding will be zeros)
    std::memset(data.get(), 0, total_bytes);
}

uint32_t TensorStorage::register_tensor(const std::string& name, size_t rows, size_t cols) {
    if (name.empty()) {
        throw std::invalid_argument("TensorStorage: tensor name cannot be empty");
    }
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("TensorStorage: tensor dimensions must be > 0");
    }
    if (name_to_id_.count(name)) {
        throw std::invalid_argument("TensorStorage: tensor '" + name + "' already registered");
    }

    uint32_t id = static_cast<uint32_t>(tensors_.size());
    tensors_.emplace_back(name, rows, cols);
    name_to_id_[name] = id;

    return id;
}

float* TensorStorage::tile_ptr(uint32_t tensor_id, uint32_t tile_row, uint32_t tile_col) {
    if (tensor_id >= tensors_.size()) {
        throw std::out_of_range("TensorStorage: invalid tensor_id");
    }

    const auto& info = tensors_[tensor_id];

    if (tile_row >= info.tile_rows || tile_col >= info.tile_cols) {
        std::ostringstream oss;
        oss << "TensorStorage: tile (" << tile_row << "," << tile_col
            << ") out of bounds for tensor '" << info.name
            << "' with tile dims (" << info.tile_rows << "," << info.tile_cols << ")";
        throw std::out_of_range(oss.str());
    }

    // Compute byte offset to tile start
    // Tiles are stored in row-major order within the padded tensor
    size_t row_start = tile_row * TILE_DIM;
    size_t col_start = tile_col * TILE_DIM;
    size_t offset = row_start * info.padded_cols + col_start;

    return info.data.get() + offset;
}

const float* TensorStorage::tile_ptr(uint32_t tensor_id, uint32_t tile_row, uint32_t tile_col) const {
    return const_cast<TensorStorage*>(this)->tile_ptr(tensor_id, tile_row, tile_col);
}

void TensorStorage::set_data(const std::string& name, const float* data, size_t rows, size_t cols) {
    auto it = name_to_id_.find(name);
    if (it == name_to_id_.end()) {
        throw std::invalid_argument("TensorStorage: tensor '" + name + "' not found");
    }

    auto& info = tensors_[it->second];

    if (rows != info.rows || cols != info.cols) {
        std::ostringstream oss;
        oss << "TensorStorage: size mismatch for '" << name
            << "'. Expected (" << info.rows << "," << info.cols
            << "), got (" << rows << "," << cols << ")";
        throw std::invalid_argument(oss.str());
    }

    // Copy row by row (source is unpadded, dest is padded)
    for (size_t r = 0; r < rows; ++r) {
        const float* src_row = data + r * cols;
        float* dst_row = info.data.get() + r * info.padded_cols;
        std::memcpy(dst_row, src_row, cols * sizeof(float));
        // Padding columns are already zero from initialization
    }
}

void TensorStorage::get_data(const std::string& name, float* data, size_t rows, size_t cols) const {
    auto it = name_to_id_.find(name);
    if (it == name_to_id_.end()) {
        throw std::invalid_argument("TensorStorage: tensor '" + name + "' not found");
    }

    const auto& info = tensors_[it->second];

    if (rows != info.rows || cols != info.cols) {
        std::ostringstream oss;
        oss << "TensorStorage: size mismatch for '" << name
            << "'. Expected (" << info.rows << "," << info.cols
            << "), got (" << rows << "," << cols << ")";
        throw std::invalid_argument(oss.str());
    }

    // Copy row by row (source is padded, dest is unpadded)
    for (size_t r = 0; r < rows; ++r) {
        const float* src_row = info.data.get() + r * info.padded_cols;
        float* dst_row = data + r * cols;
        std::memcpy(dst_row, src_row, cols * sizeof(float));
    }
}

uint32_t TensorStorage::name_to_id(const std::string& name) const {
    auto it = name_to_id_.find(name);
    if (it == name_to_id_.end()) {
        throw std::invalid_argument("TensorStorage: tensor '" + name + "' not found");
    }
    return it->second;
}

bool TensorStorage::has_tensor(const std::string& name) const {
    return name_to_id_.count(name) > 0;
}

std::pair<size_t, size_t> TensorStorage::get_shape(const std::string& name) const {
    auto it = name_to_id_.find(name);
    if (it == name_to_id_.end()) {
        throw std::invalid_argument("TensorStorage: tensor '" + name + "' not found");
    }
    const auto& info = tensors_[it->second];
    return {info.rows, info.cols};
}

size_t TensorStorage::get_stride(uint32_t tensor_id) const {
    if (tensor_id >= tensors_.size()) {
        throw std::out_of_range("TensorStorage: invalid tensor_id");
    }
    return tensors_[tensor_id].padded_cols;
}

}  // namespace mini_runtime
