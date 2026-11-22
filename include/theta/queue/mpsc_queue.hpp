// FILENAME: include/theta/queue/mpsc_queue.hpp
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

// Bounded Multiple-Producer, Single-Consumer concurrent queue.
//
// Supports two ordering policies via `kOrder`:
// 1. `QueueOrder::StrictFIFO` (default):
//    - `HybridFaaRing` producer fast path (`LOCK XADD` in interior, exact `CAS`
//      near full boundary without speculative `turn` RFO pollution) + zero-RMW
//      single-consumer `head_` line + bulk `try_push_bulk` / `drain`.
// 2. `QueueOrder::PerProducer` (`FastMPSCQueue` alias):
//    - 2-Tier Linux `rseq` + 7-slot SIMD CPU CAM lanes with zero-RMW deficit
//      round-robin single-consumer draining.
template <QueueElement T, std::size_t kStaticBufferSize = 0,
          QueueOrder kOrder = QueueOrder::StrictFIFO>
class MPSCQueue {
  friend struct queue::detail::QueueTestPeer;
  static_assert(kStaticBufferSize == 0 ||
                    (std::has_single_bit(kStaticBufferSize) &&
                     kStaticBufferSize >= 2),
                "kStaticBufferSize must be 0 (dynamic) or a power of 2 >= 2");

  static constexpr std::size_t resolve_cap(const QueueOpts& opts) noexcept {
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

  static constexpr std::size_t next_pow_2(std::size_t val) noexcept {
    constexpr std::size_t kMinSize = 2;
    return std::bit_ceil(std::max(val, kMinSize));
  }

  explicit MPSCQueue(QueueOpts opts = QueueOpts{})
      : cap_(resolve_cap(opts)),
        mask_(cap_ - 1U),
        shift_(static_cast<std::size_t>(std::countr_zero(cap_))),
        faa_limit_(cap_ - resolve_faa_margin(cap_)),
        slots_(cap_) {}

  ~MPSCQueue() {
    while (try_pop().has_value()) {
    }
  }

  MPSCQueue(const MPSCQueue&) = delete;
  MPSCQueue& operator=(const MPSCQueue&) = delete;
  MPSCQueue(MPSCQueue&&) = delete;
  MPSCQueue& operator=(MPSCQueue&&) = delete;

  template <typename... Args>
    requires std::constructible_from<T, Args...>
  bool try_emplace(std::size_t* num_items, Args&&... args) noexcept(
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
        cons_park_.notify_if_waiters();
        if (num_items != nullptr) {
          const std::uint64_t h = head_.load(std::memory_order_acquire);
          *num_items =
              (pos + 1U > h) ? static_cast<std::size_t>((pos + 1U) - h) : 1U;
        }
        return true;
      }
    }

    // Near-full boundary `CAS` path
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
          cons_park_.notify_if_waiters();
          if (num_items != nullptr) {
            const std::uint64_t h = head_.load(std::memory_order_acquire);
            *num_items =
                (pos + 1U > h) ? static_cast<std::size_t>((pos + 1U) - h) : 1U;
          }
          return true;
        }
      } else if (diff < 0) {
        if (num_items != nullptr) {
          *num_items = capacity();
        }
        return false;
      } else {
        pos = tail_.load(std::memory_order_relaxed);
      }
    }
  }

  template <typename... Args>
    requires std::constructible_from<T, Args...>
  bool try_emplace(Args&&... args) noexcept(
      std::is_nothrow_constructible_v<T, Args...>) {
    return try_emplace(nullptr, std::forward<Args>(args)...);
  }

  bool try_push(const T& val) noexcept(std::is_nothrow_copy_constructible_v<T>)
    requires std::copy_constructible<T>
  {
    return try_emplace(nullptr, val);
  }

  bool try_push(T&& val) noexcept {
    return try_emplace(nullptr, std::move(val));
  }

  bool try_push(const T& val, std::size_t* num_items) noexcept(
      std::is_nothrow_copy_constructible_v<T>)
    requires std::copy_constructible<T>
  {
    return try_emplace(num_items, val);
  }

  bool try_push(T&& val, std::size_t* num_items) noexcept {
    return try_emplace(num_items, std::move(val));
  }

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
        cons_park_.notify_if_waiters();
        pushed += batch;
      }
    }

    for (; pushed < count; ++pushed, ++first) {
      if (!try_emplace(nullptr, *first)) {
        break;
      }
    }
    return pushed;
  }

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
      prod_park_.wait_ticket(ring_slot, [this, phys, expected]() noexcept {
        return slots_.turn(phys).load(std::memory_order_acquire) == expected;
      });
    }
    slots_.storage(phys).construct(std::forward<Args>(args)...);
    slots_.turn(phys).store(expected + 1U, std::memory_order_release);
    cons_park_.notify_if_waiters();
  }

  void push(const T& val) noexcept(std::is_nothrow_copy_constructible_v<T>)
    requires std::copy_constructible<T>
  {
    emplace(val);
  }

  void push(T&& val) noexcept { emplace(std::move(val)); }

  std::optional<T> try_pop() noexcept {
    const std::size_t m = buf_mask();
    const std::size_t sh = buf_shift();
    const std::uint64_t pos = head_.load(std::memory_order_relaxed);
    const std::size_t ring_slot = static_cast<std::size_t>(pos) & m;
    const std::size_t phys = queue::detail::remap_slot_index(ring_slot, sh);
    const std::uint64_t expected = ((pos >> sh) << 1U) + 1U;
    if (slots_.turn(phys).load(std::memory_order_acquire) != expected) {
      return std::nullopt;
    }
    head_.store(pos + 1U, std::memory_order_release);
    std::optional<T> out{slots_.storage(phys).move_and_destroy()};
    slots_.turn(phys).store(expected + 1U, std::memory_order_release);
    prod_park_.notify_ticket(ring_slot);
    return out;
  }

  bool try_pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
    const std::size_t m = buf_mask();
    const std::size_t sh = buf_shift();
    const std::uint64_t pos = head_.load(std::memory_order_relaxed);
    const std::size_t ring_slot = static_cast<std::size_t>(pos) & m;
    const std::size_t phys = queue::detail::remap_slot_index(ring_slot, sh);
    const std::uint64_t expected = ((pos >> sh) << 1U) + 1U;
    if (slots_.turn(phys).load(std::memory_order_acquire) != expected) {
      return false;
    }
    head_.store(pos + 1U, std::memory_order_release);
    slots_.storage(phys).move_into_and_destroy(out);
    slots_.turn(phys).store(expected + 1U, std::memory_order_release);
    prod_park_.notify_ticket(ring_slot);
    return true;
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
    const std::size_t m = buf_mask();
    const std::size_t sh = buf_shift();
    const std::uint64_t start_pos = head_.load(std::memory_order_relaxed);
    std::size_t consumed = 0U;

    for (; consumed < max_count; ++consumed) {
      const std::uint64_t pos = start_pos + consumed;
      const std::size_t ring_slot = static_cast<std::size_t>(pos) & m;
      const std::size_t phys = queue::detail::remap_slot_index(ring_slot, sh);
      const std::uint64_t expected = ((pos >> sh) << 1U) + 1U;
      if (slots_.turn(phys).load(std::memory_order_acquire) != expected) {
        break;
      }
      T* p = slots_.storage(phys).ptr();
      const bool cont = queue::detail::invoke_drain_visitor(fn, std::move(*p));
      std::destroy_at(p);
      slots_.turn(phys).store(expected + 1U, std::memory_order_release);
      if (!cont) {
        ++consumed;
        break;
      }
    }
    if (consumed > 0U) {
      head_.store(start_pos + consumed, std::memory_order_release);
      prod_park_.notify_range(start_pos & m, consumed);
    }
    return consumed;
  }

  T pop() noexcept {
    const std::size_t m = buf_mask();
    const std::size_t sh = buf_shift();
    const std::uint64_t pos = head_.load(std::memory_order_relaxed);
    const std::size_t ring_slot = static_cast<std::size_t>(pos) & m;
    const std::size_t phys = queue::detail::remap_slot_index(ring_slot, sh);
    const std::uint64_t expected = ((pos >> sh) << 1U) + 1U;
    if (slots_.turn(phys).load(std::memory_order_acquire) != expected)
        [[unlikely]] {
      cons_park_.wait_until([this, phys, expected]() noexcept {
        return slots_.turn(phys).load(std::memory_order_acquire) == expected;
      });
    }
    head_.store(pos + 1U, std::memory_order_release);
    T out{slots_.storage(phys).move_and_destroy()};
    slots_.turn(phys).store(expected + 1U, std::memory_order_release);
    prod_park_.notify_ticket(ring_slot);
    return out;
  }

  void pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
    out = pop();
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

  const std::size_t cap_;
  const std::size_t mask_;
  const std::size_t shift_;
  const std::uint64_t faa_limit_;
  queue::detail::TurnSlotArray<T> slots_;

  queue::detail::ContentionPrefetchGate pf_gate_;

  alignas(
      kHardwareDestructiveInterferenceSize) std::atomic<std::uint64_t> tail_{0};
  std::atomic<std::uint64_t> prod_head_cache_{0};

  alignas(
      kHardwareDestructiveInterferenceSize) std::atomic<std::uint64_t> head_{0};

  queue::detail::StripedParkLot<16> prod_park_;
  queue::detail::AsymmetricParkLot cons_park_;
};

// Specialization for `QueueOrder::PerProducer`: 2-Tier `rseq` + 7-slot SIMD CAM
template <QueueElement T, std::size_t kStaticBufferSize>
class MPSCQueue<T, kStaticBufferSize, QueueOrder::PerProducer>
    : public queue::detail::RseqLaneEngine<
          T, kStaticBufferSize, /*kMultiConsumer=*/false,
          MPSCQueue<T, kStaticBufferSize, QueueOrder::StrictFIFO>> {
  using Base = queue::detail::RseqLaneEngine<
      T, kStaticBufferSize, /*kMultiConsumer=*/false,
      MPSCQueue<T, kStaticBufferSize, QueueOrder::StrictFIFO>>;

 public:
  using Base::Base;
};

template <QueueElement T, std::size_t kStaticBufferSize = 0>
using FastMPSCQueue = MPSCQueue<T, kStaticBufferSize, QueueOrder::PerProducer>;

}  // namespace theta
