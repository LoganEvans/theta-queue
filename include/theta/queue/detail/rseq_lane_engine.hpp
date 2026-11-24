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

struct EmptyRichLanePool {};

// Per-lane storage for `> 8B` or non-trivial/move-only payloads
// (`!kWord8Trivial`):
// - Fuses `{uint64_t turn, RawSlotStorage<T>}` into a single 64B cache line
//   when `sizeof(T) <= 56 && alignof(T) <= 32` (e.g. `MoveOnly32`), so turn
//   acquisition pulls `T` into L1D in a single coherence transfer.
// - Separates 64B-strided `RichTurnLine[]` and `RawSlotStorage<T>[]` for larger
//   or `>= 64B`-aligned payloads (`Payload64`, `Payload256`, `AlignedSimd64`).
template <typename T, bool kFuse = (sizeof(T) <= 56U && alignof(T) <= 32U)>
class RichLanePool;

template <typename T>
class RichLanePool<T, true> {
  struct alignas(64) FusedRichSlot {
    std::uint64_t turn;
    RawSlotStorage<T> val;
  };
  static_assert(sizeof(FusedRichSlot) == 64);
  static_assert(offsetof(FusedRichSlot, turn) == 0);

 public:
  void init(std::size_t total_slots) {
    slots_ = aligned_alloc_uninit<FusedRichSlot>(total_slots);
  }

  void destroy() noexcept {
    aligned_free_uninit(slots_);
    slots_ = nullptr;
  }

  [[nodiscard]] THETA_FORCE_INLINE std::uint64_t* base_turn_ptr() noexcept {
    return &slots_[0].turn;
  }

  [[nodiscard]] THETA_FORCE_INLINE std::uint64_t* turn_ptr(
      std::size_t flat) noexcept {
    return &slots_[flat].turn;
  }

  [[nodiscard]] THETA_FORCE_INLINE RawSlotStorage<T>& storage(
      std::size_t flat) noexcept {
    return slots_[flat].val;
  }

 private:
  FusedRichSlot* slots_{nullptr};
};

template <typename T>
class RichLanePool<T, false> {
 public:
  void init(std::size_t total_slots) {
    turns_ = aligned_alloc_uninit<RichTurnLine>(total_slots);
    vals_ = aligned_alloc_uninit<RawSlotStorage<T>>(total_slots);
  }

  void destroy() noexcept {
    aligned_free_uninit(turns_);
    aligned_free_uninit(vals_);
    turns_ = nullptr;
    vals_ = nullptr;
  }

  [[nodiscard]] THETA_FORCE_INLINE std::uint64_t* base_turn_ptr() noexcept {
    return &turns_[0].turn;
  }

  [[nodiscard]] THETA_FORCE_INLINE std::uint64_t* turn_ptr(
      std::size_t flat) noexcept {
    return &turns_[flat].turn;
  }

  [[nodiscard]] THETA_FORCE_INLINE RawSlotStorage<T>& storage(
      std::size_t flat) noexcept {
    return vals_[flat];
  }

 private:
  RichTurnLine* turns_{nullptr};
  RawSlotStorage<T>* vals_{nullptr};
};

// ============================================================================
// 2-Tier Linux `rseq` + 7-Slot SIMD/SWAR CPU CAM Multi-Lane Queue Engine
// (`QueueOrder::PerProducer` backend for `FastMPSCQueue` & `FastMPMCQueue`).
//
// - Word Mode (`<= 8B` trivially copyable/destructible): single-instruction
//   inline register store/load inside `rseq`.
// - Rich Mode (`> 8B`, move-only, or non-trivial payloads): `rseq` reserves
//   only `(lane, ticket)` against a 64B-strided turn word; `T` construction,
//   move-extraction, destruction, and turn release execute 100% outside `rseq`.
// - Overflow & Migration (`> 7` CPUs): Extra CPUs push to `fallback_` (`Lane 0`
//   `StrictFIFO` ring); cold CAM slot assignment, `MEMBARRIER_RSEQ` recycling,
//   and work-stealing state transitions live in `src/queue.cpp`.
// ============================================================================
template <QueueElement T, std::size_t kStaticBufferSize, bool kMultiConsumer,
          typename FallbackQueue>
class RseqLaneEngine {
  friend struct QueueTestPeer;
  static constexpr bool kWord8Trivial = std::is_trivially_copyable_v<T> &&
                                        std::is_trivially_destructible_v<T> &&
                                        (sizeof(T) <= 8U) && (alignof(T) <= 8U);
  static constexpr bool kUseWordLanes =
      ArchBackend::kHasRseqLanes && kWord8Trivial;
  static constexpr bool kUseRichLanes =
      ArchBackend::kHasRseqLanes && !kWord8Trivial;
  static constexpr bool kUseAnyRseqLanes = ArchBackend::kHasRseqLanes;
  static constexpr bool kUseRseqLanes = kUseAnyRseqLanes;

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
    const std::size_t cap = resolve_lane_cap(opts);
    core_.lane_cap = cap;
    if constexpr (kUseWordLanes) {
      core_.init(cap, kMultiConsumer);
    } else if constexpr (kUseRichLanes) {
      rich_.init(kMaxRseqPreferred * cap);
      core_.init_rich_turns(cap, kMultiConsumer, rich_.base_turn_ptr());
    }
  }

  ~RseqLaneEngine() {
    if constexpr (kUseWordLanes) {
      core_.destroy();
    } else if constexpr (kUseRichLanes) {
      if constexpr (!std::is_trivially_destructible_v<T>) {
        for (std::uint32_t i = 0; i < kMaxRseqPreferred; ++i) {
          const std::uint64_t h = core_.meta[i].head;
          const std::uint64_t t = core_.meta[i].tail;
          for (std::uint64_t p = h; p < t; ++p) {
            const std::size_t flat = rich_flat_index(i, p);
            if (*rich_.turn_ptr(flat) == (p * 2ULL) + 1ULL) {
              rich_.storage(flat).destroy();
            }
          }
        }
      }
      rich_.destroy();
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
    if constexpr (!kUseAnyRseqLanes) {
      if (num_items != nullptr) {
        const bool ok = fallback_.try_emplace(std::forward<Args>(args)...);
        *num_items = ok ? size() : capacity();
        return ok;
      }
      return fallback_.try_emplace(std::forward<Args>(args)...);
    } else if constexpr (kUseWordLanes) {
      const T val(std::forward<Args>(args)...);
      const bool ok = try_push_word_fast(pack_word(val));
      if (num_items != nullptr) {
        *num_items = ok ? std::max<std::size_t>(1U, size()) : capacity();
      }
      return ok;
    } else {
      const bool ok = try_emplace_rich(std::forward<Args>(args)...);
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
    if constexpr (kUseWordLanes) {
      return try_push_word_fast(pack_word(val));
    } else if constexpr (kUseRichLanes) {
      return try_emplace_rich(val);
    } else {
      return fallback_.try_push(val);
    }
  }

  bool try_push(T&& val) noexcept {
    if constexpr (kUseWordLanes) {
      return try_push_word_fast(pack_word(val));
    } else if constexpr (kUseRichLanes) {
      return try_emplace_rich(std::move(val));
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
    if constexpr (!kUseAnyRseqLanes) {
      return fallback_.try_push_bulk(first, count);
    } else if constexpr (kUseWordLanes) {
      std::size_t pushed = 0U;
      for (; pushed < count; ++pushed, ++first) {
        const T item(*first);
        if (!try_push_word_fast(pack_word(item))) {
          break;
        }
      }
      return pushed;
    } else {
      std::size_t pushed = 0U;
      for (; pushed < count; ++pushed, ++first) {
        if (!try_emplace_rich(*first)) {
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
    if constexpr (!kUseAnyRseqLanes) {
      fallback_.emplace(std::forward<Args>(args)...);
    } else if constexpr (kUseWordLanes) {
      const std::uint64_t w = pack_word(T(std::forward<Args>(args)...));
      if (try_push_word_fast(w)) [[likely]] {
        return;
      }
      push_word_blocking_slow(w);
    } else {
      if (try_emplace_rich(std::forward<Args>(args)...)) [[likely]] {
        return;
      }
      T pending(std::forward<Args>(args)...);
      push_rich_blocking_slow(pending);
    }
  }

  void push(const T& val) noexcept(std::is_nothrow_copy_constructible_v<T>)
    requires std::copy_constructible<T>
  {
    emplace(val);
  }

  void push(T&& val) noexcept { emplace(std::move(val)); }

  std::optional<T> try_pop() noexcept {
    if constexpr (!kUseAnyRseqLanes) {
      return fallback_.try_pop();
    } else if constexpr (kUseWordLanes) {
      std::uint64_t w = 0;
      if (!try_pop_word_fast(w)) {
        return std::nullopt;
      }
      return unpack_word(w);
    } else {
      return try_pop_rich_opt();
    }
  }

  bool try_pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
    if constexpr (!kUseAnyRseqLanes) {
      return fallback_.try_pop(out);
    } else if constexpr (kUseWordLanes) {
      std::uint64_t w = 0;
      if (!try_pop_word_fast(w)) {
        return false;
      }
      out = unpack_word(w);
      return true;
    } else {
      return try_pop_rich_into(out);
    }
  }

  template <typename OutputIt>
  std::size_t
  try_pop_bulk(OutputIt out, std::size_t max_count = SIZE_MAX) noexcept(
      std::is_nothrow_move_assignable_v<T>) {
    if constexpr (!kUseAnyRseqLanes) {
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
    if constexpr (!kUseAnyRseqLanes) {
      return fallback_.drain(std::forward<Func>(fn), max_count);
    } else if constexpr (kUseWordLanes) {
      if constexpr (!kMultiConsumer) {
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
    } else {
      if constexpr (!kMultiConsumer) {
        return drain_rich_sc(fn, max_count);
      } else {
        return drain_rich_mc(fn, max_count);
      }
    }
  }

  T pop() noexcept {
    if constexpr (!kUseAnyRseqLanes) {
      return fallback_.pop();
    } else if constexpr (kUseWordLanes) {
      std::uint64_t w = 0;
      if (try_pop_word_fast(w)) [[likely]] {
        return unpack_word(w);
      }
      return unpack_word(pop_word_blocking_slow());
    } else {
      if (auto item = try_pop_rich_opt(); item.has_value()) [[likely]] {
        return std::move(*item);
      }
      return pop_rich_blocking_slow();
    }
  }

  void pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
    if constexpr (kUseRichLanes) {
      if (try_pop_rich_into(out)) [[likely]] {
        return;
      }
      out = pop_rich_blocking_slow();
    } else {
      out = pop();
    }
  }

  [[nodiscard]] std::size_t size() const noexcept {
    std::size_t total = fallback_.size();
    if constexpr (kUseAnyRseqLanes) {
      total += core_.approx_size();
    }
    return total;
  }

  [[nodiscard]] bool empty() const noexcept { return size() == 0U; }

  [[nodiscard]] std::size_t capacity() const noexcept {
    return lane_cap_fast();
  }

 private:
  [[nodiscard]] THETA_FORCE_INLINE std::size_t lane_cap_fast() const noexcept {
    if constexpr (kStaticBufferSize != 0) {
      return kStaticBufferSize;
    } else {
      return core_.lane_cap;
    }
  }

  [[nodiscard]] THETA_FORCE_INLINE std::size_t rich_flat_index(
      std::uint32_t lane, std::uint64_t ticket) const noexcept {
    const std::size_t cap = lane_cap_fast();
    return (static_cast<std::size_t>(lane) * cap) +
           (static_cast<std::size_t>(ticket) & (cap - 1U));
  }

  template <typename... Args>
  THETA_FORCE_INLINE bool try_emplace_rich(Args&&... args) noexcept(
      std::is_nothrow_constructible_v<T, Args...>) {
    if constexpr (!std::is_nothrow_constructible_v<T, Args...>) {
      T staged(std::forward<Args>(args)...);
      return try_emplace_rich_nothrow(std::move(staged));
    } else {
      return try_emplace_rich_nothrow(std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  THETA_FORCE_INLINE bool try_emplace_rich_nothrow(Args&&... args) noexcept {
    std::uint32_t lane = 0;
    std::uint64_t ticket = 0;
    const RseqPushResult res =
        ArchBackend::try_reserve_rich_slot_rseq(core_, lane, ticket);
    if (res == RseqPushResult::Pushed) [[likely]] {
      const std::size_t flat = rich_flat_index(lane, ticket);
      rich_.storage(flat).construct(std::forward<Args>(args)...);
      __atomic_store_n(rich_.turn_ptr(flat), (ticket * 2ULL) + 1ULL,
                       __ATOMIC_RELEASE);
      cons_park_.notify_if_waiters();
      return true;
    }
    if (res == RseqPushResult::Full) {
      return false;
    }
    return push_rich_overflow_fallback_slow(std::forward<Args>(args)...);
  }

  THETA_FORCE_INLINE T finish_rich_pop(std::uint32_t lane,
                                       std::uint64_t ticket) noexcept {
    const std::size_t cap = lane_cap_fast();
    const std::size_t flat = rich_flat_index(lane, ticket);
    T out = rich_.storage(flat).move_and_destroy();
    __atomic_store_n(rich_.turn_ptr(flat), (ticket + cap) * 2ULL,
                     __ATOMIC_RELEASE);
    prod_park_.notify_if_waiters();
    return out;
  }

  THETA_FORCE_INLINE void finish_rich_pop_into(
      std::uint32_t lane, std::uint64_t ticket,
      T& dst) noexcept(std::is_nothrow_move_assignable_v<T>) {
    const std::size_t cap = lane_cap_fast();
    const std::size_t flat = rich_flat_index(lane, ticket);
    rich_.storage(flat).move_into_and_destroy(dst);
    __atomic_store_n(rich_.turn_ptr(flat), (ticket + cap) * 2ULL,
                     __ATOMIC_RELEASE);
    prod_park_.notify_if_waiters();
  }

  THETA_FORCE_INLINE std::optional<T> try_pop_rich_opt() noexcept {
    if constexpr (!kMultiConsumer) {
      const std::uint8_t cur_idx = core_.sc_cursor;
      LaneMeta& m = core_.meta[cur_idx];
      const std::uint64_t h = m.head;
      if (core_.sc_quantum != 0U) [[likely]] {
        const std::size_t flat = rich_flat_index(cur_idx, h);
        if (__atomic_load_n(rich_.turn_ptr(flat), __ATOMIC_ACQUIRE) ==
            (h * 2ULL) + 1ULL) [[likely]] {
          --core_.sc_quantum;
          __atomic_store_n(&m.head, h + 1U, __ATOMIC_RELEASE);
          return std::optional<T>{finish_rich_pop(cur_idx, h)};
        }
      }
      return try_pop_rich_sc_slow();
    } else {
      std::uint32_t lane = 0;
      std::uint64_t ticket = 0;
      std::optional<T> fb_out;
      if (!ArchBackend::try_claim_rich_pop_mc_rseq(
              core_, lane, ticket,
              [this, &fb_out](std::uint32_t& l_out,
                              std::uint64_t& t_out) noexcept {
                return steal_or_retry_own_rich_opt_slow(l_out, t_out, fb_out);
              })) {
        return std::nullopt;
      }
      if (lane == UINT32_MAX) [[unlikely]] {
        return fb_out;
      }
      return std::optional<T>{finish_rich_pop(lane, ticket)};
    }
  }

  THETA_FORCE_INLINE bool try_pop_rich_into(T& dst) noexcept(
      std::is_nothrow_move_assignable_v<T>) {
    if constexpr (!kMultiConsumer) {
      const std::uint8_t cur_idx = core_.sc_cursor;
      LaneMeta& m = core_.meta[cur_idx];
      const std::uint64_t h = m.head;
      if (core_.sc_quantum != 0U) [[likely]] {
        const std::size_t flat = rich_flat_index(cur_idx, h);
        if (__atomic_load_n(rich_.turn_ptr(flat), __ATOMIC_ACQUIRE) ==
            (h * 2ULL) + 1ULL) [[likely]] {
          --core_.sc_quantum;
          __atomic_store_n(&m.head, h + 1U, __ATOMIC_RELEASE);
          finish_rich_pop_into(cur_idx, h, dst);
          return true;
        }
      }
      auto slow = try_pop_rich_sc_slow();
      if (!slow.has_value()) {
        return false;
      }
      dst = std::move(*slow);
      return true;
    } else {
      std::uint32_t lane = 0;
      std::uint64_t ticket = 0;
      if (!ArchBackend::try_claim_rich_pop_mc_rseq(
              core_, lane, ticket,
              [this, &dst](std::uint32_t& l_out, std::uint64_t& t_out) noexcept(
                  std::is_nothrow_move_assignable_v<T>) {
                return steal_or_retry_own_rich_into_slow(l_out, t_out, dst);
              })) {
        return false;
      }
      if (lane != UINT32_MAX) [[likely]] {
        finish_rich_pop_into(lane, ticket, dst);
      }
      return true;
    }
  }

  THETA_ALIGNED_NOINLINE std::optional<T> try_pop_rich_sc_slow() noexcept {
    const bool quantum_expired = (core_.sc_quantum == 0U);
    if (quantum_expired &&
        core_.overflow_ops.load(std::memory_order_relaxed) != 0U) [[unlikely]] {
      if (auto fb = fallback_.try_pop(); fb.has_value()) {
        core_.sc_quantum = kRseqScQuantum - 1U;
        prod_park_.notify_if_waiters();
        return fb;
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
      const std::size_t flat = rich_flat_index(idx, h);
      if (__atomic_load_n(rich_.turn_ptr(flat), __ATOMIC_ACQUIRE) ==
          (h * 2ULL) + 1ULL) {
        core_.sc_cursor = idx;
        core_.sc_quantum = kRseqScQuantum - 1U;
        __atomic_store_n(&m.head, h + 1U, __ATOMIC_RELEASE);
        return std::optional<T>{finish_rich_pop(idx, h)};
      }
    }
    for (std::uint8_t idx = limit; idx < kMaxRseqPreferred; ++idx) {
      LaneMeta& m = core_.meta[idx];
      const std::uint64_t h = m.head;
      const std::size_t flat = rich_flat_index(idx, h);
      if (__atomic_load_n(rich_.turn_ptr(flat), __ATOMIC_ACQUIRE) ==
          (h * 2ULL) + 1ULL) {
        core_.sc_cursor = idx;
        core_.sc_quantum = kRseqScQuantum - 1U;
        __atomic_store_n(&m.head, h + 1U, __ATOMIC_RELEASE);
        return std::optional<T>{finish_rich_pop(idx, h)};
      }
    }
    if (core_.overflow_ops.load(std::memory_order_relaxed) != 0U) {
      if (auto fb = fallback_.try_pop(); fb.has_value()) {
        core_.sc_quantum = kRseqScQuantum - 1U;
        prod_park_.notify_if_waiters();
        return fb;
      }
    }
    return std::nullopt;
  }

  template <typename... Args>
  THETA_ALIGNED_NOINLINE bool push_rich_overflow_fallback_slow(
      Args&&... args) noexcept {
    const std::uint32_t ov =
        core_.overflow_ops.fetch_add(1U, std::memory_order_relaxed) + 1U;
    if ((ov & 4095U) == 0U) [[unlikely]] {
      core_.maybe_recycle_prod_cam();
    }
    const bool ok = fallback_.try_emplace(std::forward<Args>(args)...);
    if (ok) {
      cons_park_.notify_if_waiters();
    }
    return ok;
  }

  THETA_ALIGNED_NOINLINE int steal_or_retry_own_rich_opt_slow(
      std::uint32_t& out_lane, std::uint64_t& out_ticket,
      std::optional<T>& fb_out) noexcept {
    if (core_.overflow_ops.load(std::memory_order_relaxed) != 0U) [[unlikely]] {
      fb_out = fallback_.try_pop();
      if (fb_out.has_value()) {
        out_lane = UINT32_MAX;
        prod_park_.notify_if_waiters();
        return 1;
      }
    }
    return core_.try_steal_rich_slot_slow(out_lane, out_ticket);
  }

  THETA_ALIGNED_NOINLINE int steal_or_retry_own_rich_into_slow(
      std::uint32_t& out_lane, std::uint64_t& out_ticket,
      T& dst) noexcept(std::is_nothrow_move_assignable_v<T>) {
    if (core_.overflow_ops.load(std::memory_order_relaxed) != 0U) [[unlikely]] {
      if (fallback_.try_pop(dst)) {
        out_lane = UINT32_MAX;
        prod_park_.notify_if_waiters();
        return 1;
      }
    }
    return core_.try_steal_rich_slot_slow(out_lane, out_ticket);
  }

  THETA_ALIGNED_NOINLINE void push_rich_blocking_slow(T& pending) noexcept {
    bool pushed = false;
    prod_park_.wait_until([this, &pending, &pushed]() noexcept {
      if (!pushed) {
        pushed = try_emplace_rich_nothrow(std::move(pending));
      }
      return pushed;
    });
  }

  [[nodiscard]] THETA_ALIGNED_NOINLINE T pop_rich_blocking_slow() noexcept {
    std::optional<T> out;
    cons_park_.wait_until([this, &out]() noexcept {
      if (!out.has_value()) {
        out = try_pop_rich_opt();
      }
      return out.has_value();
    });
    return std::move(*out);
  }

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
    if (core_.sc_quantum != 0U) [[likely]] {
      if (h == m.tail_cache) {
        const std::uint64_t tc = __atomic_load_n(&m.tail, __ATOMIC_ACQUIRE);
        if (h == tc) {
          return try_pop_word_sc_slow(out);
        }
        m.tail_cache = tc;
      }
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

  template <typename Func>
  std::size_t drain_rich_sc(Func& fn, std::size_t max_count) noexcept(
      std::is_nothrow_invocable_v<Func&, T&&>) {
    if (max_count == 0U) {
      return 0U;
    }
    const std::size_t cap = lane_cap_fast();
    std::size_t total = 0U;
    bool keep_going = true;
    for (std::uint8_t step = 0;
         step < kMaxRseqPreferred && keep_going && total < max_count; ++step) {
      const auto idx = static_cast<std::uint8_t>((core_.sc_cursor + step) %
                                                 kMaxRseqPreferred);
      LaneMeta& m = core_.meta[idx];
      const std::uint64_t h = m.head;
      const std::size_t max_lane = max_count - total;
      std::size_t consumed = 0U;
      const struct LaneCommitGuard {
        RseqLaneEngine* self;
        LaneMeta* meta;
        std::uint64_t start_h;
        std::size_t* count;
        ~LaneCommitGuard() noexcept {
          if (*count > 0U) {
            __atomic_store_n(&meta->head, start_h + *count, __ATOMIC_RELEASE);
            self->prod_park_.notify_if_waiters();
          }
        }
      } lane_guard{this, &m, h, &consumed};

      while (consumed < max_lane) {
        const std::uint64_t p_idx = h + consumed;
        const std::size_t flat = rich_flat_index(idx, p_idx);
        if (__atomic_load_n(rich_.turn_ptr(flat), __ATOMIC_ACQUIRE) !=
            (p_idx * 2ULL) + 1ULL) {
          break;
        }
        ++consumed;
        const struct RichSlotCleanup {
          T* ptr;
          std::uint64_t* turn_p;
          std::uint64_t next_turn;
          ~RichSlotCleanup() noexcept {
            std::destroy_at(ptr);
            __atomic_store_n(turn_p, next_turn, __ATOMIC_RELEASE);
          }
        } slot_cleanup{rich_.storage(flat).ptr(), rich_.turn_ptr(flat),
                       (p_idx + cap) * 2ULL};
        if (!invoke_drain_visitor(fn, std::move(*slot_cleanup.ptr))) {
          keep_going = false;
          break;
        }
      }
      if (consumed > 0U) {
        core_.sc_cursor = idx;
        total += consumed;
      }
    }
    if (keep_going && total < max_count &&
        core_.overflow_ops.load(std::memory_order_relaxed) != 0U) {
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

  template <typename Func>
  std::size_t drain_rich_mc(Func& fn, std::size_t max_count) noexcept(
      std::is_nothrow_invocable_v<Func&, T&&>) {
    std::size_t count = 0U;
    const std::size_t cap = lane_cap_fast();
    while (count < max_count) {
      std::uint32_t lane = 0;
      std::uint64_t ticket = 0;
      std::optional<T> fb_out;
      if (!ArchBackend::try_claim_rich_pop_mc_rseq(
              core_, lane, ticket,
              [this, &fb_out](std::uint32_t& l_out,
                              std::uint64_t& t_out) noexcept {
                return steal_or_retry_own_rich_opt_slow(l_out, t_out, fb_out);
              })) {
        break;
      }
      ++count;
      if (lane == UINT32_MAX) [[unlikely]] {
        if (!invoke_drain_visitor(fn, std::move(*fb_out))) {
          break;
        }
        continue;
      }
      const std::size_t flat = rich_flat_index(lane, ticket);
      const struct RichSlotCleanup {
        RseqLaneEngine* self;
        T* ptr;
        std::uint64_t* turn_p;
        std::uint64_t next_turn;
        ~RichSlotCleanup() noexcept {
          std::destroy_at(ptr);
          __atomic_store_n(turn_p, next_turn, __ATOMIC_RELEASE);
          self->prod_park_.notify_if_waiters();
        }
      } slot_cleanup{this, rich_.storage(flat).ptr(), rich_.turn_ptr(flat),
                     (ticket + cap) * 2ULL};
      if (!invoke_drain_visitor(fn, std::move(*slot_cleanup.ptr))) {
        break;
      }
    }
    return count;
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
  [[no_unique_address]]
  std::conditional_t<kUseRichLanes, RichLanePool<T>, EmptyRichLanePool> rich_{};
  FallbackQueue fallback_;
  alignas(kHardwareDestructiveInterferenceSize) AsymmetricParkLot prod_park_;
  AsymmetricParkLot cons_park_;
};

}  // namespace theta::queue::detail
