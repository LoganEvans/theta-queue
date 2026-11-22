// FILENAME: include/theta/queue/mpmc_queue.hpp
// Copyright (c) 2023-2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <climits>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <theta/arch.hpp>
#include <theta/queue/detail/ring_common.hpp>
#include <theta/queue/detail/rseq_lane_engine.hpp>
#include <theta/queue/queue_opts.hpp>
#include <type_traits>
#include <utility>

namespace theta {

// Bounded Multiple-Producer, Multiple-Consumer concurrent queue.
//
// Supports two ordering policies via `kOrder`:
// 1. `QueueOrder::StrictFIFO` (default):
//    - Blocking `push()` / `emplace()` / `pop()` use wait-free `fetch_add(1)`
//      (`LOCK XADD`) with contention-gated `PREFETCHW` and 16-way ticket-
//      striped `StripedParkLot`.
//    - Non-blocking `try_push()` / `try_emplace()` / `try_pop()` use strictly
//      lock-free Vyukov `CAS` with `stlf_micro_pause` contention backoff
//      (never advancing `head_` or `tail_` speculatively past ready turns,
//      preserving strict lock-freedom under arbitrary OS preemption), while
//      `try_push_bulk()` / `try_pop_bulk()` reserve up to 64 verified slots in
//      a single `CAS`.
//    - Hybrid `{turn, T}` layout (`TurnSlotArray<T>`) + 128B page-local slot
//      index bouncing (`remap_slot_index`).
// 2. `QueueOrder::PerProducer` (`FastMPMCQueue` alias):
//    - 2-Tier Linux `rseq` + 7-slot SIMD CPU CAM lanes with one-shot consumer
//      work-stealing promotion.
template <QueueElement T, std::size_t kStaticBufferSize = 0,
          QueueOrder kOrder = QueueOrder::StrictFIFO>
class MPMCQueue {
  friend struct queue::detail::QueueTestPeer;
  static_assert(kStaticBufferSize == 0 ||
                    (std::has_single_bit(kStaticBufferSize) &&
                     kStaticBufferSize >= 2),
                "kStaticBufferSize must be 0 (dynamic) or a power of 2 >= 2");

  static constexpr std::size_t resolve_buffer_size(
      const QueueOpts& opts) noexcept {
    if constexpr (kStaticBufferSize != 0) {
      return kStaticBufferSize;
    } else {
      constexpr std::size_t kMinSize = 2;
      return std::bit_ceil(std::max(opts.max_size(), kMinSize));
    }
  }

 public:
  using value_type = T;

  explicit MPMCQueue(const QueueOpts& opts = QueueOpts{})
      : cap_(resolve_buffer_size(opts)),
        mask_(cap_ - 1U),
        shift_(static_cast<std::size_t>(std::countr_zero(cap_))),
        slots_(cap_) {}

  ~MPMCQueue() {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      while (try_pop().has_value()) {
      }
    }
  }

  MPMCQueue(const MPMCQueue&) = delete;
  MPMCQueue& operator=(const MPMCQueue&) = delete;
  MPMCQueue(MPMCQueue&&) = delete;
  MPMCQueue& operator=(MPMCQueue&&) = delete;

  template <typename... Args>
    requires std::constructible_from<T, Args...>
  void emplace(Args&&... args) noexcept(
      std::is_nothrow_constructible_v<T, Args...>) {
    const std::size_t m = buf_mask();
    const std::size_t sh = buf_shift();
    const std::uint64_t pos = tail_.fetch_add(1U, std::memory_order_relaxed);
    const std::size_t ring_slot = static_cast<std::size_t>(pos) & m;
    const std::size_t phys = queue::detail::remap_slot_index(ring_slot, sh);
    if (pf_gate_.should_prefetch()) {
      slots_.prefetch_write(phys);
    }
    const std::uint64_t expected = (pos >> sh) << 1U;
    if (slots_.turn(phys).load(std::memory_order_acquire) != expected)
        [[unlikely]] {
      wait_turn(prod_park_, ring_slot, phys, expected);
    }
    slots_.storage(phys).construct(std::forward<Args>(args)...);
    slots_.turn(phys).store(expected + 1U, std::memory_order_release);
    cons_park_.notify_ticket(ring_slot);
  }

  void push(const T& val) noexcept(std::is_nothrow_copy_constructible_v<T>)
    requires std::copy_constructible<T>
  {
    emplace(val);
  }

  void push(T&& val) noexcept { emplace(std::move(val)); }

  template <typename... Args>
    requires std::constructible_from<T, Args...>
  bool try_emplace(Args&&... args) noexcept(
      std::is_nothrow_constructible_v<T, Args...>) {
    const std::size_t m = buf_mask();
    const std::size_t sh = buf_shift();
    std::uint64_t pos = tail_.load(std::memory_order_relaxed);
    for (;;) {
      const std::size_t ring_slot = static_cast<std::size_t>(pos) & m;
      const std::size_t phys = queue::detail::remap_slot_index(ring_slot, sh);
      const std::uint64_t seq =
          slots_.turn(phys).load(std::memory_order_acquire);
      const std::uint64_t expected = (pos >> sh) << 1U;
      const auto diff =
          static_cast<std::int64_t>(seq) - static_cast<std::int64_t>(expected);
      if (diff == 0) [[likely]] {
        if (tail_.compare_exchange_weak(pos, pos + 1U,
                                        std::memory_order_relaxed,
                                        std::memory_order_acquire)) [[likely]] {
          slots_.storage(phys).construct(std::forward<Args>(args)...);
          slots_.turn(phys).store(expected + 1U, std::memory_order_release);
          cons_park_.notify_ticket(ring_slot);
          return true;
        }
        queue::detail::ArchBackend::stlf_micro_pause();
      } else if (diff < 0) {
        return false;
      } else {
        queue::detail::ArchBackend::stlf_micro_pause();
        pos = tail_.load(std::memory_order_relaxed);
      }
    }
  }

  bool try_push(const T& val) noexcept(std::is_nothrow_copy_constructible_v<T>)
    requires std::copy_constructible<T>
  {
    return try_emplace(val);
  }

  bool try_push(T&& val) noexcept { return try_emplace(std::move(val)); }

  std::size_t try_push_bulk(std::span<const T> items) noexcept(
      std::is_nothrow_copy_constructible_v<T>)
    requires std::copy_constructible<T>
  {
    return try_push_bulk(items.data(), items.size());
  }

  template <typename InputIt>
  std::size_t try_push_bulk(InputIt first, std::size_t count) noexcept(
      std::is_nothrow_constructible_v<T, decltype(*first)>) {
    if (count == 0U) {
      return 0U;
    }
    const std::size_t m = buf_mask();
    const std::size_t sh = buf_shift();
    const std::size_t cap = m + 1U;
    std::size_t pushed = 0U;
    std::uint64_t pos = tail_.load(std::memory_order_relaxed);

    while (pushed < count) {
      const std::size_t max_batch =
          std::min({count - pushed, cap, std::size_t{64}});
      std::size_t ready = 0U;
      std::int64_t first_diff = 0;
      for (; ready < max_batch; ++ready) {
        const std::uint64_t p = pos + ready;
        const std::size_t ring_slot = static_cast<std::size_t>(p) & m;
        const std::size_t phys = queue::detail::remap_slot_index(ring_slot, sh);
        const std::uint64_t seq =
            slots_.turn(phys).load(std::memory_order_acquire);
        const std::uint64_t expected = (p >> sh) << 1U;
        const auto diff = static_cast<std::int64_t>(seq) -
                          static_cast<std::int64_t>(expected);
        if (diff != 0) {
          if (ready == 0U) {
            first_diff = diff;
          }
          break;
        }
      }
      if (ready > 0U) [[likely]] {
        if (tail_.compare_exchange_weak(pos, pos + ready,
                                        std::memory_order_relaxed,
                                        std::memory_order_acquire)) [[likely]] {
          for (std::size_t i = 0; i < ready; ++i, ++first) {
            const std::uint64_t p = pos + i;
            const std::size_t ring_slot = static_cast<std::size_t>(p) & m;
            const std::size_t phys =
                queue::detail::remap_slot_index(ring_slot, sh);
            const std::uint64_t expected = (p >> sh) << 1U;
            slots_.storage(phys).construct(*first);
            slots_.turn(phys).store(expected + 1U, std::memory_order_release);
          }
          cons_park_.notify_range(pos, ready, m);
          pushed += ready;
          pos += ready;
          continue;
        }
        queue::detail::ArchBackend::stlf_micro_pause();
      } else if (first_diff < 0) {
        break;
      } else {
        queue::detail::ArchBackend::stlf_micro_pause();
        pos = tail_.load(std::memory_order_relaxed);
      }
    }
    return pushed;
  }

  T pop() noexcept {
    const std::size_t m = buf_mask();
    const std::size_t sh = buf_shift();
    const std::uint64_t pos = head_.fetch_add(1U, std::memory_order_relaxed);
    const std::size_t ring_slot = static_cast<std::size_t>(pos) & m;
    const std::size_t phys = queue::detail::remap_slot_index(ring_slot, sh);
    const std::uint64_t expected = ((pos >> sh) << 1U) + 1U;
    if (slots_.turn(phys).load(std::memory_order_acquire) != expected)
        [[unlikely]] {
      wait_turn(cons_park_, ring_slot, phys, expected);
    }
    T out{slots_.storage(phys).move_and_destroy()};
    slots_.turn(phys).store(expected + 1U, std::memory_order_release);
    prod_park_.notify_ticket(ring_slot);
    return out;
  }

  void pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
    out = pop();
  }

  std::optional<T> try_pop() noexcept {
    const std::size_t m = buf_mask();
    const std::size_t sh = buf_shift();
    std::uint64_t pos = head_.load(std::memory_order_relaxed);
    for (;;) {
      const std::size_t ring_slot = static_cast<std::size_t>(pos) & m;
      const std::size_t phys = queue::detail::remap_slot_index(ring_slot, sh);
      const std::uint64_t seq =
          slots_.turn(phys).load(std::memory_order_acquire);
      const std::uint64_t expected = ((pos >> sh) << 1U) + 1U;
      const auto diff =
          static_cast<std::int64_t>(seq) - static_cast<std::int64_t>(expected);
      if (diff == 0) [[likely]] {
        if (head_.compare_exchange_weak(pos, pos + 1U,
                                        std::memory_order_relaxed,
                                        std::memory_order_acquire)) [[likely]] {
          std::optional<T> out{slots_.storage(phys).move_and_destroy()};
          slots_.turn(phys).store(expected + 1U, std::memory_order_release);
          prod_park_.notify_ticket(ring_slot);
          return out;
        }
        queue::detail::ArchBackend::stlf_micro_pause();
      } else if (diff < 0) {
        return std::nullopt;
      } else {
        queue::detail::ArchBackend::stlf_micro_pause();
        pos = head_.load(std::memory_order_relaxed);
      }
    }
  }

  bool try_pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
    const std::size_t m = buf_mask();
    const std::size_t sh = buf_shift();
    std::uint64_t pos = head_.load(std::memory_order_relaxed);
    for (;;) {
      const std::size_t ring_slot = static_cast<std::size_t>(pos) & m;
      const std::size_t phys = queue::detail::remap_slot_index(ring_slot, sh);
      const std::uint64_t seq =
          slots_.turn(phys).load(std::memory_order_acquire);
      const std::uint64_t expected = ((pos >> sh) << 1U) + 1U;
      const auto diff =
          static_cast<std::int64_t>(seq) - static_cast<std::int64_t>(expected);
      if (diff == 0) [[likely]] {
        if (head_.compare_exchange_weak(pos, pos + 1U,
                                        std::memory_order_relaxed,
                                        std::memory_order_acquire)) [[likely]] {
          slots_.storage(phys).move_into_and_destroy(out);
          slots_.turn(phys).store(expected + 1U, std::memory_order_release);
          prod_park_.notify_ticket(ring_slot);
          return true;
        }
        queue::detail::ArchBackend::stlf_micro_pause();
      } else if (diff < 0) {
        return false;
      } else {
        queue::detail::ArchBackend::stlf_micro_pause();
        pos = head_.load(std::memory_order_relaxed);
      }
    }
  }

  template <typename OutputIt>
  std::size_t
  try_pop_bulk(OutputIt out, std::size_t max_count = SIZE_MAX) noexcept(
      std::is_nothrow_move_assignable_v<T>) {
    return drain(
        [&](T&& item) noexcept(std::is_nothrow_move_assignable_v<T>) {
          *out = std::move(item);
          ++out;
        },
        max_count);
  }

  template <typename Func>
  std::size_t drain(Func&& fn, std::size_t max_count = SIZE_MAX) noexcept(
      std::is_nothrow_invocable_v<Func&, T&&>) {
    if (max_count == 0U) {
      return 0U;
    }
    if constexpr (std::is_invocable_r_v<bool, Func&, T&&>) {
      std::size_t count = 0U;
      while (count < max_count) {
        std::optional<T> item = try_pop();
        if (!item.has_value()) {
          break;
        }
        ++count;
        if (!queue::detail::invoke_drain_visitor(fn, std::move(*item))) {
          break;
        }
      }
      return count;
    } else {
      const std::size_t m = buf_mask();
      const std::size_t sh = buf_shift();
      const std::size_t cap = m + 1U;
      std::size_t popped = 0U;
      std::uint64_t pos = head_.load(std::memory_order_relaxed);

      while (popped < max_count) {
        const std::size_t max_batch =
            std::min({max_count - popped, cap, std::size_t{64}});
        std::size_t ready = 0U;
        std::int64_t first_diff = 0;
        for (; ready < max_batch; ++ready) {
          const std::uint64_t p = pos + ready;
          const std::size_t ring_slot = static_cast<std::size_t>(p) & m;
          const std::size_t phys =
              queue::detail::remap_slot_index(ring_slot, sh);
          const std::uint64_t seq =
              slots_.turn(phys).load(std::memory_order_acquire);
          const std::uint64_t expected = ((p >> sh) << 1U) + 1U;
          const auto diff = static_cast<std::int64_t>(seq) -
                            static_cast<std::int64_t>(expected);
          if (diff != 0) {
            if (ready == 0U) {
              first_diff = diff;
            }
            break;
          }
        }
        if (ready > 0U) [[likely]] {
          if (head_.compare_exchange_weak(
                  pos, pos + ready, std::memory_order_relaxed,
                  std::memory_order_acquire)) [[likely]] {
            for (std::size_t i = 0; i < ready; ++i) {
              const std::uint64_t p = pos + i;
              const std::size_t ring_slot = static_cast<std::size_t>(p) & m;
              const std::size_t phys =
                  queue::detail::remap_slot_index(ring_slot, sh);
              const std::uint64_t expected = ((p >> sh) << 1U) + 1U;
              T* ptr = slots_.storage(phys).ptr();
              fn(std::move(*ptr));
              std::destroy_at(ptr);
              slots_.turn(phys).store(expected + 1U, std::memory_order_release);
            }
            prod_park_.notify_range(pos, ready, m);
            popped += ready;
            pos += ready;
            continue;
          }
          queue::detail::ArchBackend::stlf_micro_pause();
        } else if (first_diff < 0) {
          break;
        } else {
          queue::detail::ArchBackend::stlf_micro_pause();
          pos = head_.load(std::memory_order_relaxed);
        }
      }
      return popped;
    }
  }

  [[nodiscard]] std::size_t size() const noexcept {
    const std::uint64_t h = head_.load(std::memory_order_acquire);
    const std::uint64_t t = tail_.load(std::memory_order_acquire);
    if (t <= h) {
      return 0U;
    }
    const auto diff = static_cast<std::size_t>(t - h);
    const std::size_t c = capacity();
    return diff < c ? diff : c;
  }

  [[nodiscard]] bool empty() const noexcept { return size() == 0U; }

  [[nodiscard]] std::size_t capacity() const noexcept {
    if constexpr (kStaticBufferSize != 0) {
      return kStaticBufferSize;
    } else {
      return cap_;
    }
  }

 private:
  [[nodiscard]] THETA_FORCE_INLINE constexpr std::size_t buf_mask()
      const noexcept {
    if constexpr (kStaticBufferSize != 0) {
      return kStaticBufferSize - 1U;
    } else {
      return mask_;
    }
  }

  [[nodiscard]] THETA_FORCE_INLINE constexpr std::size_t buf_shift()
      const noexcept {
    if constexpr (kStaticBufferSize != 0) {
      return static_cast<std::size_t>(std::countr_zero(kStaticBufferSize));
    } else {
      return shift_;
    }
  }

  THETA_ALIGNED_NOINLINE void wait_turn(queue::detail::StripedParkLot<16>& lot,
                                        std::size_t ring_slot, std::size_t phys,
                                        std::uint64_t expected) noexcept {
    lot.wait_ticket(ring_slot, [this, phys, expected]() noexcept {
      return slots_.turn(phys).load(std::memory_order_acquire) == expected;
    });
  }

  const std::size_t cap_;
  const std::size_t mask_;
  const std::size_t shift_;
  queue::detail::TurnSlotArray<T> slots_;
  queue::detail::StripedParkLot<16> prod_park_;
  queue::detail::StripedParkLot<16> cons_park_;

  queue::detail::ContentionPrefetchGate pf_gate_;

  alignas(
      kHardwareDestructiveInterferenceSize) std::atomic<std::uint64_t> head_{0};

  alignas(
      kHardwareDestructiveInterferenceSize) std::atomic<std::uint64_t> tail_{0};
};

// Specialization for `QueueOrder::PerProducer`: 2-Tier `rseq` + 7-slot SIMD CAM
template <QueueElement T, std::size_t kStaticBufferSize>
class MPMCQueue<T, kStaticBufferSize, QueueOrder::PerProducer>
    : public queue::detail::RseqLaneEngine<
          T, kStaticBufferSize, /*kMultiConsumer=*/true,
          MPMCQueue<T, kStaticBufferSize, QueueOrder::StrictFIFO>> {
  using Base = queue::detail::RseqLaneEngine<
      T, kStaticBufferSize, /*kMultiConsumer=*/true,
      MPMCQueue<T, kStaticBufferSize, QueueOrder::StrictFIFO>>;

 public:
  using Base::Base;
};

template <QueueElement T, std::size_t kStaticBufferSize = 0>
using FastMPMCQueue = MPMCQueue<T, kStaticBufferSize, QueueOrder::PerProducer>;

}  // namespace theta
