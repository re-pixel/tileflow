/**
 * @file arena.hpp
 * @brief SRAM arena for simulated on-chip memory.
 *
 * The arena provides a flat, aligned memory buffer that simulates
 * on-chip SRAM. Unlike a general allocator, the arena doesn't make
 * allocation decisions — the Python scheduler computes all addresses.
 * The C++ side just provides raw memory access with bounds checking.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "mini_runtime/constants.hpp"

namespace mini_runtime {

/**
 * @class SRAMArena
 * @brief Fixed-size aligned memory buffer simulating on-chip SRAM.
 */
class SRAMArena {
public:
    /**
     * @brief Construct an arena with the given size.
     * @param total_bytes Total size of the arena in bytes.
     * @throws std::invalid_argument if total_bytes is 0.
     * @throws std::bad_alloc if allocation fails.
     */
    explicit SRAMArena(size_t total_bytes);

    // Non-copyable, movable
    SRAMArena(const SRAMArena&) = delete;
    SRAMArena& operator=(const SRAMArena&) = delete;
    SRAMArena(SRAMArena&&) noexcept = default;
    SRAMArena& operator=(SRAMArena&&) noexcept = default;

    /**
     * @brief Get a pointer to the specified address in the arena.
     * @param addr Byte offset into the arena.
     * @return Pointer to the address as float*.
     * @throws std::out_of_range if addr is out of bounds.
     */
    float* ptr(uint32_t addr);
    const float* ptr(uint32_t addr) const;

    /**
     * @brief Validate that an address range is within bounds.
     * @param addr Starting byte offset.
     * @param bytes Number of bytes to access.
     * @throws std::out_of_range if the range exceeds arena bounds.
     */
    void validate_range(uint32_t addr, size_t bytes) const;

    /**
     * @brief Zero-fill the entire arena.
     *
     * Called before execution to ensure accumulators start at zero.
     */
    void clear();

    /**
     * @brief Get the total size of the arena.
     * @return Size in bytes.
     */
    size_t total_bytes() const noexcept { return total_bytes_; }

private:
    size_t total_bytes_;
    std::unique_ptr<char[], decltype(&std::free)> buffer_;
};

}  // namespace mini_runtime
