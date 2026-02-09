/**
 * @file ring_buffer.hpp
 * @brief Lock-free Single-Producer Single-Consumer (SPSC) ring buffer.
 *
 * This is the inter-thread communication primitive for the pipelined
 * engine. The DMA thread produces work items; the compute thread consumes
 * them. Only atomic increments are used — no mutexes.
 *
 * Design:
 * - Power-of-2 capacity for fast modulo (bitwise AND)
 * - Acquire/release memory ordering for cross-thread visibility
 * - No dynamic allocation — fixed-size array
 * - Cache-line padding to prevent false sharing between head and tail
 */

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <new>  // std::hardware_destructive_interference_size

namespace mini_runtime {

/// Cache line size for padding (fallback if not available)
#ifdef __cpp_lib_hardware_interference_size
    constexpr size_t CACHE_LINE_SIZE = std::hardware_destructive_interference_size;
#else
    constexpr size_t CACHE_LINE_SIZE = 64;
#endif

/**
 * @class RingBuffer
 * @brief Lock-free SPSC circular buffer with power-of-2 capacity.
 *
 * @tparam T Element type (must be trivially copyable for correctness).
 * @tparam Capacity Number of slots. Must be a power of 2.
 *
 * Thread safety:
 * - Exactly one thread may call try_push() (producer).
 * - Exactly one thread may call try_pop() (consumer).
 * - is_empty(), is_full(), and size() are safe from either thread
 *   but provide only a snapshot.
 */
template <typename T, size_t Capacity>
class RingBuffer {
public:
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "RingBuffer capacity must be a power of 2");
    static_assert(Capacity > 0, "RingBuffer capacity must be > 0");

    RingBuffer() = default;

    // Non-copyable, non-movable (contains atomics)
    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;
    RingBuffer(RingBuffer&&) = delete;
    RingBuffer& operator=(RingBuffer&&) = delete;

    /**
     * @brief Try to push an item into the buffer.
     *
     * @param item The item to push.
     * @return true if the item was pushed, false if the buffer is full.
     *
     * @note Only the producer thread may call this.
     */
    bool try_push(const T& item) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next_head = (head + 1) & mask_;

        // Check if full: next write position would collide with read position
        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false;  // Full
        }

        buffer_[head] = item;

        // Release: ensure the item write is visible before head advances
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    /**
     * @brief Spin-push: block until the item is pushed.
     *
     * @param item The item to push.
     *
     * @note Only the producer thread may call this.
     * @warning Will spin indefinitely if the consumer is not draining.
     */
    void push(const T& item) {
        while (!try_push(item)) {
            // Spin — in a real system we'd use pause/yield
#if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
#endif
        }
    }

    /**
     * @brief Try to pop an item from the buffer.
     *
     * @param item [out] The popped item (only valid if return is true).
     * @return true if an item was popped, false if the buffer is empty.
     *
     * @note Only the consumer thread may call this.
     */
    bool try_pop(T& item) {
        const size_t tail = tail_.load(std::memory_order_relaxed);

        // Check if empty: read position equals write position
        if (tail == head_.load(std::memory_order_acquire)) {
            return false;  // Empty
        }

        item = buffer_[tail];

        // Release: ensure the item read completes before tail advances
        tail_.store((tail + 1) & mask_, std::memory_order_release);
        return true;
    }

    /**
     * @brief Check if the buffer is empty.
     * @return true if empty (snapshot — may change immediately).
     */
    bool is_empty() const {
        return head_.load(std::memory_order_acquire)
            == tail_.load(std::memory_order_acquire);
    }

    /**
     * @brief Check if the buffer is full.
     * @return true if full (snapshot — may change immediately).
     */
    bool is_full() const {
        const size_t next_head = (head_.load(std::memory_order_acquire) + 1) & mask_;
        return next_head == tail_.load(std::memory_order_acquire);
    }

    /**
     * @brief Get the current number of items in the buffer.
     * @return Number of items (snapshot — may change immediately).
     */
    size_t size() const {
        const size_t head = head_.load(std::memory_order_acquire);
        const size_t tail = tail_.load(std::memory_order_acquire);
        return (head - tail) & mask_;
    }

    /**
     * @brief Get the capacity of the buffer.
     * @return Maximum number of items the buffer can hold (Capacity - 1 usable).
     */
    static constexpr size_t capacity() { return Capacity; }

private:
    static constexpr size_t mask_ = Capacity - 1;

    std::array<T, Capacity> buffer_{};

    // Pad to separate cache lines to prevent false sharing
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_{0};
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_{0};
};

}  // namespace mini_runtime
