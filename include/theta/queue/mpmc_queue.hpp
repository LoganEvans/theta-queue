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
//    - `HybridFaaRing` ticket engine: uses wait-free `LOCK XADD` (`fetch_add`)
//      in the interior of the ring and transitions seamlessly to exact Vyukov
//      `CAS` near full/empty boundaries (`try_push_bulk` reserves N slots in
//      1 `XADD`).
//    - Hybrid `{turn, T}` layout (`TurnSlotArray<T>`), 128B page-local slot
//      index bouncing (`remap_slot_index`), contention-gated real `PREFETCHW`,
//      `stlf_micro_pause` SMT-friendly spin, and 16-way ticket-striped
//      `StripedParkLot`.
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

  static constexpr std::size_t resolve_faa_margin(std::size_t cap) noexcept {
    return cap >= 256U ? std::min<std::size_t>(cap / 4U, 128U) : cap;
  }

 public:
  using value_type = T;

  explicit MPMCQueue(const QueueOpts& opts = QueueOpts{})
      : cap_(resolve_buffer_size(opts)),
        mask_(cap_ - 1U),
        shift_(static_cast<std::size_t>(std::countr_zero(cap_))),
        faa_limit_(cap_ - resolve_faa_margin(cap_)),
        faa_pop_thresh_(resolve_faa_margin(cap_)),
        slots_(cap_) {}

  ~MPMCQueue() {
    while (try_pop().has_value()) {
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
    const std::uint64_t lim = faa_limit();

    // Interior wait-free `LOCK XADD` fast path (`cap >= 256` and `< lim` full)
    if (lim > 0U) [[likely]] {
      const std::uint64_t approx_t = tail_.load(std::memory_order_relaxed);
      std::uint64_t hc = prod_head_cache_.load(std::memory_order_relaxed);
      if (approx_t > hc && (approx_t - hc) >= lim) [[unlikely]] {
        const std::uint64_t fresh_hc = head_.load(std::memory_order_acquire);
        if (fresh_hc != hc) {
          prod_head_cache_.store(fresh_hc, std::memory_order_relaxed);
          hc = fresh_hc;
        }
      }
      if (approx_t <= hc || (approx_t - hc) < lim) [[likely]] {
        const std::uint64_t pos =
            tail_.fetch_add(1U, std::memory_order_relaxed);
        const std::size_t ring_slot = static_cast<std::size_t>(pos) & m;
        const std::size_t phys = queue::detail::remap_slot_index(ring_slot, sh);
        if (pf_gate_.should_prefetch()) {
          slots_.prefetch_write(phys);
        }
        const std::uint64_t expected = (pos >> sh) << 1U;
        std::uint32_t spin = 0;
        while (slots_.turn(phys).load(std::memory_order_acquire) != expected)
            [[unlikely]] {
          queue::detail::spin_pause(spin);
        }
        slots_.storage(phys).construct(std::forward<Args>(args)...);
        slots_.turn(phys).store(expected + 1U, std::memory_order_release);
        cons_park_.notify_ticket(ring_slot);
        return true;
      }
    }

    // Near-full boundary path (or small `cap < 256` rings): exact Vyukov `CAS`
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
      } else if (diff < 0) {
        return false;
      } else {
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
    const std::uint64_t lim = faa_limit();
    std::size_t pushed = 0U;

    if (lim > 0U) {
      const std::uint64_t approx_t = tail_.load(std::memory_order_relaxed);
      std::uint64_t hc = prod_head_cache_.load(std::memory_order_relaxed);
      std::uint64_t used = approx_t > hc ? (approx_t - hc) : 0U;
      if (used + count > lim) {
        hc = head_.load(std::memory_order_acquire);
        prod_head_cache_.store(hc, std::memory_order_relaxed);
        used = approx_t > hc ? (approx_t - hc) : 0U;
      }
      if (used < lim) {
        const std::size_t batch =
            std::min<std::size_t>(count, static_cast<std::size_t>(lim - used));
        const std::uint64_t base =
            tail_.fetch_add(batch, std::memory_order_relaxed);
        const bool do_pf = pf_gate_.should_prefetch();
        for (std::size_t i = 0; i < batch; ++i, ++first) {
          const std::uint64_t pos = base + i;
          const std::size_t ring_slot = static_cast<std::size_t>(pos) & m;
          const std::size_t phys =
              queue::detail::remap_slot_index(ring_slot, sh);
          if (do_pf) {
            slots_.prefetch_write(phys);
          }
          const std::uint64_t expected = (pos >> sh) << 1U;
          std::uint32_t spin = 0;
          while (slots_.turn(phys).load(std::memory_order_acquire) != expected)
              [[unlikely]] {
            queue::detail::spin_pause(spin);
          }
          slots_.storage(phys).construct(*first);
          slots_.turn(phys).store(expected + 1U, std::memory_order_release);
        }
        cons_park_.notify_range(base & m, batch);
        pushed += batch;
      }
    }

    for (; pushed < count; ++pushed, ++first) {
      if (!try_emplace(*first)) {
        break;
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
    const std::uint64_t thresh = faa_pop_thresh();

    if (fa_pop_eligible(thresh)) [[unlikely]] {
      const std::uint64_t pos = head_.fetch_add(1U, std::memory_order_relaxed);
      const std::size_t ring_slot = static_cast<std::size_t>(pos) & m;
      const std::size_t phys = queue::detail::remap_slot_index(ring_slot, sh);
      const std::uint64_t expected = ((pos >> sh) << 1U) + 1U;
      std::uint32_t spin = 0;
      while (slots_.turn(phys).load(std::memory_order_acquire) != expected)
          [[unlikely]] {
        queue::detail::spin_pause(spin);
      }
      std::optional<T> out{slots_.storage(phys).move_and_destroy()};
      slots_.turn(phys).store(expected + 1U, std::memory_order_release);
      prod_park_.notify_ticket(ring_slot);
      return out;
    }

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
      } else if (diff < 0) {
        return std::nullopt;
      } else {
        pos = head_.load(std::memory_order_relaxed);
      }
    }
  }

  bool try_pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
    const std::size_t m = buf_mask();
    const std::size_t sh = buf_shift();
    const std::uint64_t thresh = faa_pop_thresh();

    if (fa_pop_eligible(thresh)) [[unlikely]] {
      const std::uint64_t pos = head_.fetch_add(1U, std::memory_order_relaxed);
      const std::size_t ring_slot = static_cast<std::size_t>(pos) & m;
      const std::size_t phys = queue::detail::remap_slot_index(ring_slot, sh);
      const std::uint64_t expected = ((pos >> sh) << 1U) + 1U;
      std::uint32_t spin = 0;
      while (slots_.turn(phys).load(std::memory_order_acquire) != expected)
          [[unlikely]] {
        queue::detail::spin_pause(spin);
      }
      slots_.storage(phys).move_into_and_destroy(out);
      slots_.turn(phys).store(expected + 1U, std::memory_order_release);
      prod_park_.notify_ticket(ring_slot);
      return true;
    }

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
      } else if (diff < 0) {
        return false;
      } else {
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

  [[nodiscard]] THETA_FORCE_INLINE constexpr std::uint64_t faa_limit()
      const noexcept {
    if constexpr (kStaticBufferSize != 0) {
      return kStaticBufferSize - resolve_faa_margin(kStaticBufferSize);
    } else {
      return faa_limit_;
    }
  }

  [[nodiscard]] THETA_FORCE_INLINE constexpr std::uint64_t faa_pop_thresh()
      const noexcept {
    if constexpr (kStaticBufferSize != 0) {
      return resolve_faa_margin(kStaticBufferSize);
    } else {
      return faa_pop_thresh_;
    }
  }

  [[nodiscard]] THETA_FORCE_INLINE bool fa_pop_eligible(
      std::uint64_t thresh) noexcept {
    if (faa_limit() == 0U) {
      return false;
    }
    const std::uint64_t approx_h = head_.load(std::memory_order_relaxed);
    std::uint64_t tc = cons_tail_cache_.load(std::memory_order_relaxed);
    if (tc <= approx_h || (tc - approx_h) <= thresh) [[likely]] {
      const std::uint64_t fresh_tc = tail_.load(std::memory_order_acquire);
      if (fresh_tc != tc) {
        cons_tail_cache_.store(fresh_tc, std::memory_order_relaxed);
        tc = fresh_tc;
      }
    }
    return tc > approx_h && (tc - approx_h) > thresh;
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
  const std::uint64_t faa_limit_;
  const std::uint64_t faa_pop_thresh_;
  queue::detail::TurnSlotArray<T> slots_;

  queue::detail::ContentionPrefetchGate pf_gate_;

  alignas(
      kHardwareDestructiveInterferenceSize) std::atomic<std::uint64_t> head_{0};
  std::atomic<std::uint64_t> cons_tail_cache_{0};

  alignas(
      kHardwareDestructiveInterferenceSize) std::atomic<std::uint64_t> tail_{0};
  std::atomic<std::uint64_t> prod_head_cache_{0};

  queue::detail::StripedParkLot<16> prod_park_;
  queue::detail::StripedParkLot<16> cons_park_;
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
