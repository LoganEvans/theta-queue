// FILENAME: tests/queue_test_peer.hpp
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <theta/queue.hpp>

namespace theta::queue::detail {

struct QueueTestPeer {
  template <typename T, std::size_t kCap>
  [[nodiscard]] static std::uint8_t spsc_cons_blocking_mode(
      const SPSCQueue<T, kCap>& q) noexcept {
    return q.cons_park_.blocking_active_.load(std::memory_order_acquire);
  }

  template <typename T, std::size_t kCap>
  [[nodiscard]] static std::uint8_t mpmc_cons_blocking_mode(
      const MPMCQueue<T, kCap, QueueOrder::StrictFIFO>& q) noexcept {
    return q.cons_park_.blocking_active_.load(std::memory_order_acquire);
  }

  template <typename FastQueue>
  [[nodiscard]] static std::uint8_t rseq_prod_biased(
      const FastQueue& q) noexcept {
    return __atomic_load_n(&q.core_.prod_biased, __ATOMIC_ACQUIRE);
  }

  template <typename FastQueue>
  [[nodiscard]] static std::uint8_t rseq_num_prod(const FastQueue& q) noexcept {
    return q.core_.num_prod.load(std::memory_order_acquire);
  }

  template <typename FastQueue>
  [[nodiscard]] static std::uint32_t rseq_overflow_ops(
      const FastQueue& q) noexcept {
    return q.core_.overflow_ops.load(std::memory_order_acquire);
  }

  template <typename FastQueue>
  [[nodiscard]] static constexpr bool uses_rseq_lanes() noexcept {
    return FastQueue::kUseRseqLanes;
  }
};

}  // namespace theta::queue::detail
