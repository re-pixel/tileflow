/**
 * @file engine.cpp
 * @brief Schedule execution engine implementation.
 */

#include "mini_runtime/engine.hpp"
#include "mini_runtime/kernels.hpp"

#include <cstring>
#include <stdexcept>
#include <variant>

namespace mini_runtime {

Engine::Engine(const Config& config)
    : config_(config)
    , sram_(config.sram_bytes)
    , tensors_()
    , stats_()
{
}

uint32_t Engine::register_tensor(const std::string& name, size_t rows, size_t cols) {
    return tensors_.register_tensor(name, rows, cols);
}

void Engine::set_tensor(const std::string& name, const float* data, size_t rows, size_t cols) {
    tensors_.set_data(name, data, rows, cols);
}

void Engine::get_tensor(const std::string& name, float* data, size_t rows, size_t cols) const {
    tensors_.get_data(name, data, rows, cols);
}

void Engine::execute(const std::vector<SchedOp>& schedule) {
    // Clear SRAM before execution (accumulators start at zero)
    sram_.clear();

    // Dispatch each operation in sequence
    for (const auto& op : schedule) {
        std::visit([this](const auto& typed_op) {
            this->dispatch(typed_op);
        }, op);
    }
}

void Engine::dispatch(const SchedLoad& op) {
    // Validate SRAM range
    sram_.validate_range(op.dst_addr, op.bytes);

    // Get source pointer from tensor storage (points to first element of tile)
    const float* src = tensors_.tile_ptr(op.tensor_id, op.tile_row, op.tile_col);

    // Get destination pointer in SRAM
    float* dst = sram_.ptr(op.dst_addr);

    // Get the stride of the source tensor (padded_cols)
    // For a tensor, consecutive rows are separated by padded_cols elements
    // We need to copy row-by-row since tiles may not be contiguous in memory
    size_t src_stride = tensors_.get_stride(op.tensor_id);
    
    // Copy tile row by row from tensor (with stride) to SRAM (contiguous)
    for (uint32_t row = 0; row < TILE_DIM; ++row) {
        std::memcpy(dst + row * TILE_DIM, 
                    src + row * src_stride, 
                    TILE_DIM * sizeof(float));
    }

    stats_.loads++;
}

void Engine::dispatch(const SchedExecute& op) {
    // Validate SRAM ranges
    sram_.validate_range(op.a_addr, TILE_BYTES);
    sram_.validate_range(op.b_addr, TILE_BYTES);
    sram_.validate_range(op.acc_addr, TILE_BYTES);

    // Get tile pointers from SRAM
    const float* A = sram_.ptr(op.a_addr);
    const float* B = sram_.ptr(op.b_addr);
    float* C = sram_.ptr(op.acc_addr);

    // If k == 0, this is the first partial product for this output tile.
    // Clear the accumulator to zero before accumulating.
    // This handles the case where the accumulator slot was previously used
    // for a different output tile and still contains stale data.
    if (op.k == 0) {
        std::memset(C, 0, TILE_BYTES);
    }

    // Execute tile matmul-accumulate: C += A @ B
    matmul_tile_ref(C, A, B);

    stats_.executes++;
}

void Engine::dispatch(const SchedStore& op) {
    // Validate SRAM range
    sram_.validate_range(op.src_addr, op.bytes);

    // Get source pointer in SRAM
    float* src = sram_.ptr(op.src_addr);

    // Apply activation if requested (in-place in SRAM before copy)
    if (op.apply_relu) {
        relu_tile_inplace(src);
    }

    // Get destination pointer in tensor storage
    float* dst = tensors_.tile_ptr(op.tensor_id, op.tile_row, op.tile_col);

    // Get the stride of the destination tensor
    size_t dst_stride = tensors_.get_stride(op.tensor_id);
    
    // Copy tile row by row from SRAM (contiguous) to tensor (with stride)
    for (uint32_t row = 0; row < TILE_DIM; ++row) {
        std::memcpy(dst + row * dst_stride,
                    src + row * TILE_DIM,
                    TILE_DIM * sizeof(float));
    }

    stats_.stores++;
}

}  // namespace mini_runtime
