/**
 * @file test_ring_buffer.cpp
 * @brief Unit tests for the lock-free SPSC ring buffer.
 *
 * Tests cover:
 * 1. Single-threaded push/pop correctness
 * 2. Capacity limits (full/empty detection)
 * 3. Wrap-around behavior
 * 4. Multi-threaded producer-consumer correctness
 */

#include "mini_runtime/ring_buffer.hpp"

#include <atomic>
#include <cstdint>
#include <numeric>
#include <thread>
#include <vector>

#ifdef GTEST_AVAILABLE
#include <gtest/gtest.h>
#else
// Minimal test harness if GTest is not available
#include <cassert>
#include <iostream>

#define TEST(suite, name) void suite##_##name()
#define EXPECT_TRUE(x) assert(x)
#define EXPECT_FALSE(x) assert(!(x))
#define EXPECT_EQ(a, b) assert((a) == (b))

#define RUN_TEST(suite, name) do { \
    std::cout << "  " #suite "." #name "..."; \
    suite##_##name(); \
    std::cout << " PASSED" << std::endl; \
} while(0)
#endif

using mini_runtime::RingBuffer;

// ============================================================================
// 1. Basic Push/Pop
// ============================================================================

TEST(RingBuffer, EmptyOnConstruction) {
    RingBuffer<int, 8> rb;
    EXPECT_TRUE(rb.is_empty());
    EXPECT_FALSE(rb.is_full());
    EXPECT_EQ(rb.size(), 0u);
}

TEST(RingBuffer, SinglePushPop) {
    RingBuffer<int, 8> rb;

    EXPECT_TRUE(rb.try_push(42));
    EXPECT_FALSE(rb.is_empty());
    EXPECT_EQ(rb.size(), 1u);

    int val = 0;
    EXPECT_TRUE(rb.try_pop(val));
    EXPECT_EQ(val, 42);
    EXPECT_TRUE(rb.is_empty());
}

TEST(RingBuffer, PopFromEmptyFails) {
    RingBuffer<int, 8> rb;
    int val = -1;
    EXPECT_FALSE(rb.try_pop(val));
    EXPECT_EQ(val, -1);  // Unchanged
}

TEST(RingBuffer, FIFOOrder) {
    RingBuffer<int, 8> rb;

    for (int i = 0; i < 7; ++i) {
        EXPECT_TRUE(rb.try_push(i * 10));
    }

    for (int i = 0; i < 7; ++i) {
        int val = -1;
        EXPECT_TRUE(rb.try_pop(val));
        EXPECT_EQ(val, i * 10);
    }
}

// ============================================================================
// 2. Capacity Limits
// ============================================================================

TEST(RingBuffer, FullDetection) {
    // Capacity 4 means 3 usable slots (one slot reserved for empty/full distinction)
    RingBuffer<int, 4> rb;

    EXPECT_TRUE(rb.try_push(1));
    EXPECT_TRUE(rb.try_push(2));
    EXPECT_TRUE(rb.try_push(3));

    // 4th push should fail (capacity is 4, but 1 slot reserved)
    EXPECT_TRUE(rb.is_full());
    EXPECT_FALSE(rb.try_push(4));

    // Pop one, then push should succeed
    int val = 0;
    EXPECT_TRUE(rb.try_pop(val));
    EXPECT_EQ(val, 1);
    EXPECT_FALSE(rb.is_full());
    EXPECT_TRUE(rb.try_push(4));
}

TEST(RingBuffer, SizeTracking) {
    RingBuffer<int, 8> rb;

    EXPECT_EQ(rb.size(), 0u);

    rb.try_push(1);
    EXPECT_EQ(rb.size(), 1u);

    rb.try_push(2);
    rb.try_push(3);
    EXPECT_EQ(rb.size(), 3u);

    int val;
    rb.try_pop(val);
    EXPECT_EQ(rb.size(), 2u);
}

// ============================================================================
// 3. Wrap-Around
// ============================================================================

TEST(RingBuffer, WrapAround) {
    RingBuffer<int, 4> rb;  // 3 usable slots

    // Fill and drain multiple times to force wrap-around
    for (int round = 0; round < 10; ++round) {
        for (int i = 0; i < 3; ++i) {
            EXPECT_TRUE(rb.try_push(round * 100 + i));
        }
        for (int i = 0; i < 3; ++i) {
            int val = -1;
            EXPECT_TRUE(rb.try_pop(val));
            EXPECT_EQ(val, round * 100 + i);
        }
        EXPECT_TRUE(rb.is_empty());
    }
}

// ============================================================================
// 4. Multi-Threaded Producer-Consumer
// ============================================================================

TEST(RingBuffer, ThreadedProducerConsumer) {
    constexpr size_t N = 100000;
    RingBuffer<uint64_t, 256> rb;

    std::atomic<bool> producer_done{false};
    std::vector<uint64_t> received;
    received.reserve(N);

    // Producer thread
    std::thread producer([&]() {
        for (uint64_t i = 0; i < N; ++i) {
            rb.push(i);  // Spin-push
        }
        producer_done.store(true, std::memory_order_release);
    });

    // Consumer thread (current thread)
    while (true) {
        uint64_t val;
        if (rb.try_pop(val)) {
            received.push_back(val);
        } else if (producer_done.load(std::memory_order_acquire) && rb.is_empty()) {
            break;
        }
    }

    producer.join();

    // Verify all items received in order
    EXPECT_EQ(received.size(), N);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_EQ(received[i], static_cast<uint64_t>(i));
    }
}

TEST(RingBuffer, ThreadedStressSmallBuffer) {
    // Stress test with a very small buffer (2 usable slots)
    constexpr size_t N = 50000;
    RingBuffer<uint32_t, 4> rb;

    std::atomic<bool> producer_done{false};
    std::vector<uint32_t> received;
    received.reserve(N);

    std::thread producer([&]() {
        for (uint32_t i = 0; i < N; ++i) {
            rb.push(i);
        }
        producer_done.store(true, std::memory_order_release);
    });

    while (true) {
        uint32_t val;
        if (rb.try_pop(val)) {
            received.push_back(val);
        } else if (producer_done.load(std::memory_order_acquire) && rb.is_empty()) {
            break;
        }
    }

    producer.join();

    EXPECT_EQ(received.size(), N);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_EQ(received[i], static_cast<uint32_t>(i));
    }
}

// ============================================================================
// Main (for non-GTest builds)
// ============================================================================

#ifndef GTEST_AVAILABLE
int main() {
    std::cout << "Ring Buffer Tests:" << std::endl;

    RUN_TEST(RingBuffer, EmptyOnConstruction);
    RUN_TEST(RingBuffer, SinglePushPop);
    RUN_TEST(RingBuffer, PopFromEmptyFails);
    RUN_TEST(RingBuffer, FIFOOrder);
    RUN_TEST(RingBuffer, FullDetection);
    RUN_TEST(RingBuffer, SizeTracking);
    RUN_TEST(RingBuffer, WrapAround);
    RUN_TEST(RingBuffer, ThreadedProducerConsumer);
    RUN_TEST(RingBuffer, ThreadedStressSmallBuffer);

    std::cout << "\nAll ring buffer tests passed!" << std::endl;
    return 0;
}
#endif
