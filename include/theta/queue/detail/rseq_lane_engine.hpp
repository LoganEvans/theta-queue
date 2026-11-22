// FILENAME: include/theta/queue/detail/rseq_lane_engine.hpp
// Copyright (c) 2023-2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <bit>
#include <climits>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <theta/arch.hpp>
#include <theta/queue/detail/ring_common.hpp>
#include <theta/queue/queue_opts.hpp>
#include <type_traits>
#include <utility>

namespace theta::queue::detail {

// ============================================================================
// 2-Tier Linux `rseq` + 7-Slot SIMD CPU CAM Multi-Lane Queue Engine
// (`QueueOrder::PerProducer` backend for `FastMPSCQueue` & `FastMPMCQueue`).
//
// - Tier 1 (`prod_biased == 1` / `cons_biased == 1`): 2-instruction `movzwl`
//   bijection check + offset-`0` `meta[0]` addressing (`0` SSE2, `0` shift,
//   `0` LOCK instructions).
// - Tier 2 (`prod_biased == 0`): 16-byte SIMD inverse CAM mapping up to 7
//   active CPUs (`cpu_id + 1` in `[1..8192]`) onto 7 cache-isolated SPSC lanes
//   (`sizeof(LaneMeta) == 256` -> 1-cycle `shlq $8` indexing).
// - Overflow & Migration (`> 7` CPUs): Extra CPUs push to `fallback_` (`Lane 0`
//   `HybridFaaRing`); cold CAM slot assignment, `MEMBARRIER_RSEQ` recycling,
//   and work-stealing state transitions live in `src/queue.cpp`.
// ============================================================================
template <QueueElement T, std::size_t kStaticBufferSize, bool kMultiConsumer,
          typename FallbackQueue>
class RseqLaneEngine {
  friend struct QueueTestPeer;
  static constexpr bool kWord8Trivial = std::is_trivially_copyable_v<T> &&
                                        std::is_trivially_destructible_v<T> &&
                                        (sizeof(T) <= 8U) && (alignof(T) <= 8U);
  static constexpr bool kUseRseqLanes =
      ArchBackend::kHasRseqLanes && kWord8Trivial;

  static constexpr std::size_t resolve_lane_cap(
      const QueueOpts& opts) noexcept {
    if constexpr (kStaticBufferSize != 0) {
      return kStaticBufferSize;
    } else {
      constexpr std::size_t kMinSize = 2;
      return std::bit_ceil(std::max(opts.max_size(), kMinSize));
    }
  }

  [[nodiscard]] static THETA_FORCE_INLINE std::uint64_t pack_word(
      const T& val) noexcept {
    std::uint64_t w = 0;
    std::memcpy(&w, static_cast<const void*>(std::addressof(val)), sizeof(T));
    return w;
  }

  [[nodiscard]] static THETA_FORCE_INLINE T
  unpack_word(std::uint64_t w) noexcept {
    T out;
    std::memcpy(static_cast<void*>(std::addressof(out)), &w, sizeof(T));
    return out;
  }

 public:
  using value_type = T;

  explicit RseqLaneEngine(const QueueOpts& opts = QueueOpts{})
      : fallback_(opts) {
    core_.lane_cap = resolve_lane_cap(opts);
    if constexpr (kUseRseqLanes) {
      core_.init(core_.lane_cap, kMultiConsumer);
    }
  }

  ~RseqLaneEngine() {
    if constexpr (kUseRseqLanes) {
      core_.destroy();
    }
  }

  RseqLaneEngine(const RseqLaneEngine&) = delete;
  RseqLaneEngine& operator=(const RseqLaneEngine&) = delete;
  RseqLaneEngine(RseqLaneEngine&&) = delete;
  RseqLaneEngine& operator=(RseqLaneEngine&&) = delete;

  template <typename... Args>
    requires std::constructible_from<T, Args...>
  bool try_emplace(std::size_t* num_items, Args&&... args) noexcept(
      std::is_nothrow_constructible_v<T, Args...>) {
    if constexpr (!kUseRseqLanes) {
      if (num_items != nullptr) {
        const bool ok = fallback_.try_emplace(std::forward<Args>(args)...);
        *num_items = ok ? size() : capacity();
        return ok;
      }
      return fallback_.try_emplace(std::forward<Args>(args)...);
    } else {
      const T val(std::forward<Args>(args)...);
      const bool ok = try_push_word_fast(pack_word(val));
      if (num_items != nullptr) {
        *num_items = ok ? std::max<std::size_t>(1U, size()) : capacity();
      }
      return ok;
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
    if constexpr (kUseRseqLanes) {
      return try_push_word_fast(pack_word(val));
    } else {
      return fallback_.try_push(val);
    }
  }

  bool try_push(T&& val) noexcept {
    if constexpr (kUseRseqLanes) {
      return try_push_word_fast(pack_word(val));
    } else {
      return fallback_.try_push(std::move(val));
    }
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
    if constexpr (!kUseRseqLanes) {
      return fallback_.try_push_bulk(first, count);
    } else {
      std::size_t pushed = 0U;
      for (; pushed < count; ++pushed, ++first) {
        const T item(*first);
        if (!try_push_word_fast(pack_word(item))) {
          break;
        }
      }
      return pushed;
    }
  }

  template <typename... Args>
    requires std::constructible_from<T, Args...>
  void emplace(Args&&... args) noexcept(
      std::is_nothrow_constructible_v<T, Args...>) {
    if constexpr (!kUseRseqLanes) {
      fallback_.emplace(std::forward<Args>(args)...);
    } else {
      const std::uint64_t w = pack_word(T(std::forward<Args>(args)...));
      if (try_push_word_fast(w)) [[likely]] {
        return;
      }
      push_word_blocking_slow(w);
    }
  }

  void push(const T& val) noexcept(std::is_nothrow_copy_constructible_v<T>)
    requires std::copy_constructible<T>
  {
    emplace(val);
  }

  void push(T&& val) noexcept { emplace(std::move(val)); }

  std::optional<T> try_pop() noexcept {
    if constexpr (!kUseRseqLanes) {
      return fallback_.try_pop();
    } else {
      std::uint64_t w = 0;
      if (!try_pop_word_fast(w)) {
        return std::nullopt;
      }
      return unpack_word(w);
    }
  }

  bool try_pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
    if constexpr (!kUseRseqLanes) {
      return fallback_.try_pop(out);
    } else {
      std::uint64_t w = 0;
      if (!try_pop_word_fast(w)) {
        return false;
      }
      out = unpack_word(w);
      return true;
    }
  }

  template <typename OutputIt>
  std::size_t
  try_pop_bulk(OutputIt out, std::size_t max_count = SIZE_MAX) noexcept(
      std::is_nothrow_move_assignable_v<T>) {
    if constexpr (!kUseRseqLanes) {
      return fallback_.try_pop_bulk(out, max_count);
    } else {
      return drain(
          [&](T&& item) noexcept(std::is_nothrow_move_assignable_v<T>) {
            *out = std::move(item);
            ++out;
          },
          max_count);
    }
  }

  template <typename Func>
  std::size_t drain(Func&& fn, std::size_t max_count = SIZE_MAX) noexcept(
      std::is_nothrow_invocable_v<Func&, T&&>) {
    if constexpr (!kUseRseqLanes) {
      return fallback_.drain(std::forward<Func>(fn), max_count);
    } else if constexpr (!kMultiConsumer) {
      return drain_sc(fn, max_count);
    } else {
      std::size_t count = 0U;
      std::uint64_t w = 0;
      while (count < max_count && try_pop_word_fast(w)) {
        ++count;
        if (!invoke_drain_visitor(fn, unpack_word(w))) {
          break;
        }
      }
      return count;
    }
  }

  T pop() noexcept {
    if constexpr (!kUseRseqLanes) {
      return fallback_.pop();
    } else {
      std::uint64_t w = 0;
      if (try_pop_word_fast(w)) [[likely]] {
        return unpack_word(w);
      }
      return unpack_word(pop_word_blocking_slow());
    }
  }

  void pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
    out = pop();
  }

  [[nodiscard]] std::size_t size() const noexcept {
    std::size_t total = fallback_.size();
    if constexpr (kUseRseqLanes) {
      total += core_.approx_size();
    }
    return total;
  }

  [[nodiscard]] bool empty() const noexcept { return size() == 0U; }

  [[nodiscard]] std::size_t capacity() const noexcept {
    if constexpr (kStaticBufferSize != 0) {
      return kStaticBufferSize;
    } else {
      return core_.lane_cap;
    }
  }

 private:
  THETA_FORCE_INLINE bool try_push_word_fast(std::uint64_t val) noexcept {
    const RseqPushResult res =
        ArchBackend::try_push_word_rseq(core_, cons_park_, val);
    if (res == RseqPushResult::Pushed) [[likely]] {
      return true;
    }
    if (res == RseqPushResult::Full) {
      return false;
    }
    return push_overflow_fallback_slow(val);
  }

  THETA_FORCE_INLINE bool try_pop_word_fast(std::uint64_t& out) noexcept {
    if constexpr (!kMultiConsumer) {
      return try_pop_word_sc(out);
    } else {
      return ArchBackend::try_pop_word_mc_rseq(
          core_, prod_park_, out, [this](std::uint64_t& dst) noexcept {
            return steal_or_retry_own_mc_slow(dst);
          });
    }
  }

  // Single-consumer MPSC fast path: Deficit Round-Robin (`kRseqScQuantum = 64`)
  // across active lanes — amortizes local `tail_cache` buffering by 64x while
  // guaranteeing strict O(7 * 64) starvation freedom under infinite flood.
  THETA_FORCE_INLINE bool try_pop_word_sc(std::uint64_t& out) noexcept {
    const std::uint8_t cur_idx = core_.sc_cursor;
    LaneMeta& m = core_.meta[cur_idx];
    const std::uint64_t h = m.head;
    if (h != m.tail_cache && core_.sc_quantum != 0U) [[likely]] {
      --core_.sc_quantum;
      out = m.data_cons[h & m.mask_cons];
      __atomic_store_n(&m.head, h + 1U, __ATOMIC_RELEASE);
      prod_park_.notify_if_waiters();
      return true;
    }
    return try_pop_word_sc_slow(out);
  }

  THETA_ALIGNED_NOINLINE bool try_pop_word_sc_slow(
      std::uint64_t& out) noexcept {
    const bool quantum_expired = (core_.sc_quantum == 0U);
    if (quantum_expired &&
        core_.overflow_ops.load(std::memory_order_relaxed) != 0U) [[unlikely]] {
      T fb_item;
      if (fallback_.try_pop(fb_item)) {
        core_.sc_quantum = kRseqScQuantum - 1U;
        out = pack_word(fb_item);
        prod_park_.notify_if_waiters();
        return true;
      }
    }

    const std::uint8_t n_prod = core_.num_prod.load(std::memory_order_acquire);
    const std::uint8_t limit = n_prod < kMaxRseqPreferred
                                   ? (n_prod == 0U ? 1U : n_prod)
                                   : kMaxRseqPreferred;
    const std::uint8_t start =
        quantum_expired
            ? static_cast<std::uint8_t>((core_.sc_cursor + 1U) % limit)
            : core_.sc_cursor;
    for (std::uint8_t step = 0; step < limit; ++step) {
      const auto idx = static_cast<std::uint8_t>((start + step) % limit);
      LaneMeta& m = core_.meta[idx];
      const std::uint64_t h = m.head;
      const std::uint64_t t = __atomic_load_n(&m.tail, __ATOMIC_ACQUIRE);
      if (h != t) {
        m.tail_cache = t;
        core_.sc_cursor = idx;
        core_.sc_quantum = kRseqScQuantum - 1U;
        out = m.data_cons[h & m.mask_cons];
        __atomic_store_n(&m.head, h + 1U, __ATOMIC_RELEASE);
        prod_park_.notify_if_waiters();
        return true;
      }
    }
    for (std::uint8_t idx = limit; idx < kMaxRseqPreferred; ++idx) {
      LaneMeta& m = core_.meta[idx];
      const std::uint64_t h = m.head;
      const std::uint64_t t = __atomic_load_n(&m.tail, __ATOMIC_ACQUIRE);
      if (h != t) {
        m.tail_cache = t;
        core_.sc_cursor = idx;
        core_.sc_quantum = kRseqScQuantum - 1U;
        out = m.data_cons[h & m.mask_cons];
        __atomic_store_n(&m.head, h + 1U, __ATOMIC_RELEASE);
        prod_park_.notify_if_waiters();
        return true;
      }
    }
    if (core_.overflow_ops.load(std::memory_order_relaxed) != 0U) {
      T fb_item;
      if (fallback_.try_pop(fb_item)) {
        core_.sc_quantum = kRseqScQuantum - 1U;
        out = pack_word(fb_item);
        prod_park_.notify_if_waiters();
        return true;
      }
    }
    return false;
  }

  template <typename Func>
  std::size_t drain_sc(Func& fn, std::size_t max_count) noexcept(
      std::is_nothrow_invocable_v<Func&, T&&>) {
    if (max_count == 0U) {
      return 0U;
    }
    std::size_t total = 0U;
    bool keep_going = true;
    for (std::uint8_t step = 0;
         step < kMaxRseqPreferred && keep_going && total < max_count; ++step) {
      const auto idx = static_cast<std::uint8_t>((core_.sc_cursor + step) %
                                                 kMaxRseqPreferred);
      LaneMeta& m = core_.meta[idx];
      const std::uint64_t h = m.head;
      std::uint64_t t = m.tail_cache;
      if (h == t) {
        t = __atomic_load_n(&m.tail, __ATOMIC_ACQUIRE);
        m.tail_cache = t;
        if (h == t) {
          continue;
        }
      }
      core_.sc_cursor = idx;
      const auto avail = static_cast<std::size_t>(t - h);
      const std::size_t limit = std::min(max_count - total, avail);
      std::size_t consumed = 0U;
      for (; consumed < limit; ++consumed) {
        const std::uint64_t w = m.data_cons[(h + consumed) & m.mask_cons];
        if (!invoke_drain_visitor(fn, unpack_word(w))) {
          ++consumed;
          keep_going = false;
          break;
        }
      }
      __atomic_store_n(&m.head, h + consumed, __ATOMIC_RELEASE);
      prod_park_.notify_if_waiters();
      total += consumed;
    }
    if (keep_going && total < max_count) {
      const std::size_t fb_n = fallback_.drain(
          [&](T&& item) noexcept(std::is_nothrow_invocable_v<Func&, T&&>) {
            return invoke_drain_visitor(fn, std::move(item));
          },
          max_count - total);
      if (fb_n > 0U) {
        prod_park_.notify_if_waiters();
      }
      total += fb_n;
    }
    return total;
  }

  THETA_ALIGNED_NOINLINE bool push_overflow_fallback_slow(
      std::uint64_t val) noexcept {
    const std::uint32_t ov =
        core_.overflow_ops.fetch_add(1U, std::memory_order_relaxed) + 1U;
    if ((ov & 4095U) == 0U) [[unlikely]] {
      core_.maybe_recycle_prod_cam();
    }
    const bool ok = fallback_.try_push(unpack_word(val));
    if (ok) {
      cons_park_.notify_if_waiters();
    }
    return ok;
  }

  THETA_ALIGNED_NOINLINE int steal_or_retry_own_mc_slow(
      std::uint64_t& out) noexcept {
    if (core_.overflow_ops.load(std::memory_order_relaxed) != 0U) [[unlikely]] {
      T fb_item;
      if (fallback_.try_pop(fb_item)) {
        out = pack_word(fb_item);
        prod_park_.notify_if_waiters();
        return 1;
      }
    }
    const int st = core_.try_steal_lane_word_slow(out);
    if (st > 0) {
      prod_park_.notify_if_waiters();
    }
    return st;
  }

  THETA_ALIGNED_NOINLINE void push_word_blocking_slow(
      std::uint64_t val) noexcept {
    bool pushed = false;
    prod_park_.wait_until([this, val, &pushed]() noexcept {
      if (!pushed) {
        pushed = try_push_word_fast(val);
      }
      return pushed;
    });
  }

  [[nodiscard]] THETA_ALIGNED_NOINLINE std::uint64_t
  pop_word_blocking_slow() noexcept {
    std::uint64_t out = 0;
    bool got = false;
    cons_park_.wait_until([this, &out, &got]() noexcept {
      if (!got) {
        got = try_pop_word_fast(out);
      }
      return got;
    });
    return out;
  }

  RseqLaneCore core_{};
  FallbackQueue fallback_;
  AsymmetricParkLot prod_park_;
  AsymmetricParkLot cons_park_;
};

}  // namespace theta::queue::detail
