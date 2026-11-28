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
    return AsymmetricParkLot::blocking_mode_slot(q.cons_park_);
  }

  template <typename T, std::size_t kCap>
  [[nodiscard]] static std::uint8_t spsc_cons_spin_shift(
      const SPSCQueue<T, kCap>& q) noexcept {
    const auto* s = q.cons_park_.load(std::memory_order_acquire);
    return s != nullptr ? static_cast<std::uint8_t>(
                              s->spin_shift.load(std::memory_order_acquire))
                        : kMaxParkSpinShift;
  }

  template <typename T, std::size_t kCap>
  [[nodiscard]] static bool spsc_cons_fence_armed(
      const SPSCQueue<T, kCap>& q) noexcept {
    const auto* s = q.cons_park_.load(std::memory_order_acquire);
    return s != nullptr && (s->sleep_state.load(std::memory_order_acquire) &
                            kParkFenceArmedBit) != 0U;
  }

  template <typename T, std::size_t kCap>
  [[nodiscard]] static std::uint8_t mpmc_cons_blocking_mode(
      const MPMCQueue<T, kCap, QueueOrder::StrictFIFO>& q) noexcept {
    return q.cons_park_.blocking_mode();
  }

  template <typename T, std::size_t kCap>
  [[nodiscard]] static bool mpmc_cons_fence_armed(
      const MPMCQueue<T, kCap, QueueOrder::StrictFIFO>& q) noexcept {
    const auto* s = q.cons_park_.state_.load(std::memory_order_acquire);
    return s != nullptr && (s->fence_state.load(std::memory_order_acquire) &
                            kParkFenceArmedBit) != 0U;
  }

  template <typename T, std::size_t kCap>
  [[nodiscard]] static bool mpmc_cons_work_conserving(
      const MPMCQueue<T, kCap, QueueOrder::StrictFIFO>& q) noexcept {
    return q.cons_park_.is_work_conserving();
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
