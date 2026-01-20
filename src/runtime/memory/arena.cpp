/**
 * @file arena.cpp
 * @brief SRAM arena implementation.
 */

#include "mini_runtime/arena.hpp"

#include <cstdlib>
#include <cstring>
#include <sstream>

namespace mini_runtime {

SRAMArena::SRAMArena(size_t total_bytes)
    : total_bytes_(total_bytes)
    , buffer_(nullptr, &std::free)
{
    if (total_bytes == 0) {
        throw std::invalid_argument("SRAMArena: total_bytes must be > 0");
    }

    // Allocate aligned memory
    void* ptr = std::aligned_alloc(ALIGNMENT, total_bytes);
    if (!ptr) {
        throw std::bad_alloc();
    }

    buffer_.reset(static_cast<char*>(ptr));

    // Zero-initialize
    clear();
}

float* SRAMArena::ptr(uint32_t addr) {
    validate_range(addr, sizeof(float));
    return reinterpret_cast<float*>(buffer_.get() + addr);
}

const float* SRAMArena::ptr(uint32_t addr) const {
    validate_range(addr, sizeof(float));
    return reinterpret_cast<const float*>(buffer_.get() + addr);
}

void SRAMArena::validate_range(uint32_t addr, size_t bytes) const {
    if (addr + bytes > total_bytes_) {
        std::ostringstream oss;
        oss << "SRAMArena: access out of bounds. "
            << "addr=0x" << std::hex << addr
            << ", bytes=" << std::dec << bytes
            << ", total=" << total_bytes_;
        throw std::out_of_range(oss.str());
    }
}

void SRAMArena::clear() {
    std::memset(buffer_.get(), 0, total_bytes_);
}

}  // namespace mini_runtime
