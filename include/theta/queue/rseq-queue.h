// FILENAME: include/theta/queue/rseq-queue.h
// Copyright (c) 2023-2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

#include "theta/queue/defs.h"
#include "theta/queue/queue-opts.h"

namespace theta::queue_detail {

// ============================================================================
// 2-Tier Linux `rseq` + 7-Slot SIMD CPU CAM Multi-Lane Queue Engine
// (`QueueOrder::PerProducer` backend for `FastMPSCQueue` & `FastMPMCQueue`).
//
// Design highlights (cross-pollinated from `theta::Mutex` & `theta-notes`):
// - Tier 1 (`prod_biased_ == 1` / `cons_biased_ == 1`): 2-instruction `movzwl`
//   bijection check + offset-`0` `meta_[0]` addressing (`0` SSE2, `0` shift,
//   `0` LOCK instructions -> ~3.5 ns `1T` Push+Pop).
// - Tier 2 (`prod_biased_ == 0`): 16-byte SSE2 `pcmpeqw` inverse CAM mapping up
//   to 7 active CPUs (`cpu_id + 1` in `[1..8192]`) onto 7 cache-isolated SPSC
//   lanes (`sizeof(LaneMeta) == 256` -> 1-cycle `shlq $8` indexing).
// - Overflow & Migration (`> 7` CPUs): Extra CPUs push to `fallback_` (`Lane 0`
//   `HybridFaaRing`). If preferred lanes go idle after thread migration, a
//   one-shot `MEMBARRIER_CMD_PRIVATE_EXPEDITED_RSEQ` recycles the CAM table.
// - Consumer paths:
//   * MPSC (`kMultiConsumer == false`): Single consumer owns all 7 lane heads
//     exclusively with sticky round-robin draining (`0` `rseq`, `0` `CAS`).
//   * MPMC (`kMultiConsumer == true`): Dual 2-Tier `rseq` Consumer CAM with
//     inline `tail_cache` refresh and one-shot `cons_exclusive` promotion to
//     monotonic 64-bit `CAS` work-stealing (`0` wrap-around ABA, at most 1
//     `MEMBARRIER_RSEQ` per stolen lane over the queue's lifetime).
// ============================================================================
template <QueueElement T, std::size_t kStaticBufferSize, bool kMultiConsumer,
          typename FallbackQueue>
class RseqLaneEngine {
  static constexpr std::uint8_t kMaxPreferred = 7U;
  static constexpr std::uint8_t kScQuantum = 64U;
  static constexpr std::int32_t kMaxRseqCpus = 8192;

  static constexpr bool kWord8Trivial = std::is_trivially_copyable_v<T> &&
                                        std::is_trivially_destructible_v<T> &&
                                        (sizeof(T) <= 8U) && (alignof(T) <= 8U);

#if THETA_QUEUE_HAS_LINUX_RSEQ && defined(__x86_64__) && !THETA_QUEUE_SANITIZED
  static constexpr bool kUseRseqLanes = kWord8Trivial;
#else
  static constexpr bool kUseRseqLanes = false;
#endif

  // Exact 256-byte (`1 << 8`) lane metadata:
  //   [0..127]   Producer cache-line pair (`tail`, `head_cache`, `data_prod`,
  //              `cap_prod`, `mask_prod`)
  //   [128..255] Consumer cache-line pair (`head`, `tail_cache`, `data_cons`,
  //              `mask_cons`, `cons_exclusive`)
  struct alignas(256) LaneMeta {
    std::uint64_t tail{0};
    std::uint64_t head_cache{0};
    std::uint64_t* data_prod{nullptr};
    std::uint64_t cap_prod{0};
    std::uint64_t mask_prod{0};
    std::array<char, 128 - 40> pad_prod{};

    std::uint64_t head{0};
    std::uint64_t tail_cache{0};
    std::uint64_t* data_cons{nullptr};
    std::uint64_t mask_cons{0};
    std::uint8_t cons_exclusive{1};
    std::array<char, 128 - 33> pad_cons{};
  };
  static_assert(sizeof(LaneMeta) == 256);
  static_assert(offsetof(LaneMeta, cap_prod) == 24);
  static_assert(offsetof(LaneMeta, mask_prod) == 32);
  static_assert(offsetof(LaneMeta, head) == 128);
  static_assert(offsetof(LaneMeta, tail_cache) == 136);
  static_assert(offsetof(LaneMeta, data_cons) == 144);
  static_assert(offsetof(LaneMeta, mask_cons) == 152);
  static_assert(offsetof(LaneMeta, cons_exclusive) == 160);

  static constexpr std::size_t resolve_lane_cap(
      const QueueOpts& opts) noexcept {
    if constexpr (kStaticBufferSize != 0) {
      return kStaticBufferSize;
    } else {
      constexpr std::size_t kMinSize = 2;
      return std::bit_ceil(std::max(opts.max_size(), kMinSize));
    }
  }

  [[nodiscard]] static THETA_QUEUE_FORCE_INLINE std::uint64_t pack_word(
      const T& val) noexcept {
    std::uint64_t w = 0;
    std::memcpy(&w, static_cast<const void*>(std::addressof(val)), sizeof(T));
    return w;
  }

  [[nodiscard]] static THETA_QUEUE_FORCE_INLINE T
  unpack_word(std::uint64_t w) noexcept {
    T out;
    std::memcpy(static_cast<void*>(std::addressof(out)), &w, sizeof(T));
    return out;
  }

 public:
  using value_type = T;

  explicit RseqLaneEngine(const QueueOpts& opts = QueueOpts{})
      : lane_cap_(resolve_lane_cap(opts)), fallback_(opts) {
    if constexpr (kUseRseqLanes) {
      lane_buf_ =
          aligned_alloc_uninit<std::uint64_t>(kMaxPreferred * lane_cap_);
      const std::uint64_t m = lane_cap_ - 1U;
      for (std::size_t i = 0; i < kMaxPreferred; ++i) {
        meta_[i].data_prod = lane_buf_ + (i * lane_cap_);
        meta_[i].cap_prod = lane_cap_;
        meta_[i].mask_prod = m;
        meta_[i].data_cons = lane_buf_ + (i * lane_cap_);
        meta_[i].mask_cons = m;
        meta_[i].cons_exclusive = kMultiConsumer ? 1U : 0U;
      }
      // Prime expedited RSEQ membarrier registration once per process.
      (void)asymmetric_membarrier_rseq_expedited();
    }
  }

  ~RseqLaneEngine() {
    if constexpr (kUseRseqLanes) {
      aligned_free_uninit(lane_buf_);
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
      for (std::size_t i = 0; i < kMaxPreferred; ++i) {
        const std::uint64_t h =
            __atomic_load_n(&meta_[i].head, __ATOMIC_ACQUIRE);
        const std::uint64_t t =
            __atomic_load_n(&meta_[i].tail, __ATOMIC_ACQUIRE);
        if (t > h) {
          const auto d = static_cast<std::size_t>(t - h);
          total += (d < lane_cap_ ? d : lane_cap_);
        }
      }
    }
    return total;
  }

  [[nodiscard]] bool empty() const noexcept { return size() == 0U; }

  [[nodiscard]] std::size_t capacity() const noexcept {
    if constexpr (kStaticBufferSize != 0) {
      return kStaticBufferSize;
    } else {
      return lane_cap_;
    }
  }

 private:
#if THETA_QUEUE_HAS_LINUX_RSEQ && defined(__x86_64__) && !THETA_QUEUE_SANITIZED
  THETA_QUEUE_FORCE_INLINE bool try_push_word_fast(std::uint64_t val) noexcept {
    const std::ptrdiff_t rseq_off = __rseq_offset;

  retry_push:
    // Tier 1: 1-CPU Biased (`movzwl` bijection + `meta_[0]`, 0 SSE2, 0 shift!)
    {
      std::uint64_t tmp = 0;
      std::uint64_t slot = 0;
      std::uint64_t cur_t = 0;
      std::uint64_t next_t = 0;
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "leaq 1b(%%rip), %[tmp]\n\t"
          "movq %[tmp], %%fs:8(%[rseq_off])\n\t"
          "2:\n\t"
          "cmpb $0, (%[biased])\n\t"
          "je %l[tier2_7slot]\n\t"
          "movl %%fs:4(%[rseq_off]), %k[tmp]\n\t"
          "movzwl 2(%[cam]), %k[slot]\n\t"
          "incl %k[tmp]\n\t"
          "cmpl %k[slot], %k[tmp]\n\t"
          "jne %l[slow_push]\n\t"
          "movq (%[meta]), %[cur_t]\n\t"
          "movq %[cur_t], %[next_t]\n\t"
          "subq 8(%[meta]), %[next_t]\n\t"
          "cmpq 24(%[meta]), %[next_t]\n\t"
          "jae %l[refresh_push]\n\t"
          "leaq 1(%[cur_t]), %[next_t]\n\t"
          "andq 32(%[meta]), %[cur_t]\n\t"
          "movq 16(%[meta]), %[tmp]\n\t"
          "movq %[val], (%[tmp], %[cur_t], 8)\n\t"
          "movq %[next_t], (%[meta])\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax\"\n\t"
          ".long 0x53053053\n\t"
          "4:\n\t"
          "jmp 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [slot] "=&r"(slot), [cur_t] "=&r"(cur_t),
            [next_t] "=&r"(next_t)
          : [rseq_off] "r"(rseq_off), [biased] "r"(&prod_biased_),
            [cam] "r"(prod_cam_.data()), [meta] "r"(meta_.data()),
            [val] "r"(val)
          : "memory", "cc"
          : tier2_7slot, refresh_push, slow_push);
      cons_park_.notify_if_waiters();
      return true;
    }

  tier2_7slot:
    {
      __m128i v_cpu{};  // NOLINT(misc-const-correctness)
      std::uint64_t tmp = 0;
      std::uint64_t slot = 0;
      std::uint64_t cur_t = 0;
      std::uint64_t next_t = 0;
      std::uintptr_t lp = 0;
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "leaq 1b(%%rip), %[tmp]\n\t"
          "movq %[tmp], %%fs:8(%[rseq_off])\n\t"
          "2:\n\t"
          "movl %%fs:4(%[rseq_off]), %k[tmp]\n\t"
          "cmpl $8192, %k[tmp]\n\t"
          "jae %l[slow_push]\n\t"
          "incl %k[tmp]\n\t"
          "movd %k[tmp], %[v]\n\t"
          "pshuflw $0, %[v], %[v]\n\t"
          "punpcklqdq %[v], %[v]\n\t"
          "pcmpeqw (%[cam]), %[v]\n\t"
          "pmovmskb %[v], %k[slot]\n\t"
          "bsfl %k[slot], %k[slot]\n\t"
          "je %l[slow_push]\n\t"
          "shrl $1, %k[slot]\n\t"
          "leaq -1(%[slot]), %[tmp]\n\t"
          "shlq $8, %[tmp]\n\t"
          "leaq (%[meta], %[tmp], 1), %[lp]\n\t"
          "movq (%[lp]), %[cur_t]\n\t"
          "movq %[cur_t], %[next_t]\n\t"
          "subq 8(%[lp]), %[next_t]\n\t"
          "cmpq 24(%[lp]), %[next_t]\n\t"
          "jae %l[refresh_push]\n\t"
          "leaq 1(%[cur_t]), %[next_t]\n\t"
          "andq 32(%[lp]), %[cur_t]\n\t"
          "movq 16(%[lp]), %[tmp]\n\t"
          "movq %[val], (%[tmp], %[cur_t], 8)\n\t"
          "movq %[next_t], (%[lp])\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax\"\n\t"
          ".long 0x53053053\n\t"
          "4:\n\t"
          "jmp 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [slot] "=&r"(slot), [cur_t] "=&r"(cur_t),
            [next_t] "=&r"(next_t), [lp] "=&r"(lp), [v] "=&x"(v_cpu)
          : [rseq_off] "r"(rseq_off), [cam] "r"(prod_cam_.data()),
            [meta] "r"(meta_.data()), [val] "r"(val)
          : "memory", "cc"
          : refresh_push, slow_push);
      cons_park_.notify_if_waiters();
      return true;
    }

  refresh_push:
    {
      const int status = refresh_push_state_slow();
      if (status > 0) {
        goto retry_push;
      }
      if (status == 0) {
        return false;
      }
    }
  slow_push:
    if (try_assign_prod_slot_slow()) {
      goto retry_push;
    }
    return push_overflow_fallback_slow(val);
  }

  THETA_QUEUE_FORCE_INLINE bool try_pop_word_fast(std::uint64_t& out) noexcept {
    if constexpr (!kMultiConsumer) {
      return try_pop_word_sc(out);
    } else {
      return try_pop_word_mc(out);
    }
  }

  // Single-consumer MPSC fast path: Deficit Round-Robin (`kScQuantum = 64`)
  // across active lanes — amortizes local `tail_cache` buffering by 64x while
  // guaranteeing strict O(7 * 64) starvation freedom under infinite flood!
  THETA_QUEUE_FORCE_INLINE bool try_pop_word_sc(std::uint64_t& out) noexcept {
    const std::uint8_t cur_idx = sc_cursor_;
    LaneMeta& m = meta_[cur_idx];
    const std::uint64_t h = m.head;
    if (h != m.tail_cache && sc_quantum_ != 0U) [[likely]] {
      --sc_quantum_;
      out = m.data_cons[h & m.mask_cons];
      __atomic_store_n(&m.head, h + 1U, __ATOMIC_RELEASE);
      prod_park_.notify_if_waiters();
      return true;
    }
    return try_pop_word_sc_slow(out);
  }

  THETA_QUEUE_ALIGNED_COLD bool try_pop_word_sc_slow(
      std::uint64_t& out) noexcept {
    const bool quantum_expired = (sc_quantum_ == 0U);
    if (quantum_expired && overflow_ops_.load(std::memory_order_relaxed) != 0U)
        [[unlikely]] {
      T fb_item;
      if (fallback_.try_pop(fb_item)) {
        sc_quantum_ = kScQuantum - 1U;
        out = pack_word(fb_item);
        prod_park_.notify_if_waiters();
        return true;
      }
    }

    const std::uint8_t n_prod = num_prod_.load(std::memory_order_acquire);
    const std::uint8_t limit =
        n_prod < kMaxPreferred ? (n_prod == 0U ? 1U : n_prod) : kMaxPreferred;
    const std::uint8_t start =
        quantum_expired ? static_cast<std::uint8_t>((sc_cursor_ + 1U) % limit)
                        : sc_cursor_;
    for (std::uint8_t step = 0; step < limit; ++step) {
      const auto idx = static_cast<std::uint8_t>((start + step) % limit);
      LaneMeta& m = meta_[idx];
      const std::uint64_t h = m.head;
      const std::uint64_t t = __atomic_load_n(&m.tail, __ATOMIC_ACQUIRE);
      if (h != t) {
        m.tail_cache = t;
        sc_cursor_ = idx;
        sc_quantum_ = kScQuantum - 1U;
        out = m.data_cons[h & m.mask_cons];
        __atomic_store_n(&m.head, h + 1U, __ATOMIC_RELEASE);
        prod_park_.notify_if_waiters();
        return true;
      }
    }
    for (std::uint8_t idx = limit; idx < kMaxPreferred; ++idx) {
      LaneMeta& m = meta_[idx];
      const std::uint64_t h = m.head;
      const std::uint64_t t = __atomic_load_n(&m.tail, __ATOMIC_ACQUIRE);
      if (h != t) {
        m.tail_cache = t;
        sc_cursor_ = idx;
        sc_quantum_ = kScQuantum - 1U;
        out = m.data_cons[h & m.mask_cons];
        __atomic_store_n(&m.head, h + 1U, __ATOMIC_RELEASE);
        prod_park_.notify_if_waiters();
        return true;
      }
    }
    if (overflow_ops_.load(std::memory_order_relaxed) != 0U) {
      T fb_item;
      if (fallback_.try_pop(fb_item)) {
        sc_quantum_ = kScQuantum - 1U;
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
         step < kMaxPreferred && keep_going && total < max_count; ++step) {
      const auto idx =
          static_cast<std::uint8_t>((sc_cursor_ + step) % kMaxPreferred);
      LaneMeta& m = meta_[idx];
      const std::uint64_t h = m.head;
      std::uint64_t t = m.tail_cache;
      if (h == t) {
        t = __atomic_load_n(&m.tail, __ATOMIC_ACQUIRE);
        m.tail_cache = t;
        if (h == t) {
          continue;
        }
      }
      sc_cursor_ = idx;
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

  // Multi-consumer MPMC fast path: Dual 2-Tier `rseq` Consumer CAM with inline
  // `tail_cache` reload + one-shot `cons_exclusive` promotion to `CAS` steal.
  THETA_QUEUE_FORCE_INLINE bool try_pop_word_mc(std::uint64_t& out) noexcept {
    const std::ptrdiff_t rseq_off = __rseq_offset;

  retry_pop:
    // Tier 1 Consumer (`cons_biased_ == 1`):
    {
      std::uint64_t tmp = 0;
      std::uint64_t slot = 0;
      std::uint64_t cur_h = 0;
      std::uint64_t next_h = 0;
      std::uint64_t val_out = 0;
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "leaq 1b(%%rip), %[tmp]\n\t"
          "movq %[tmp], %%fs:8(%[rseq_off])\n\t"
          "2:\n\t"
          "cmpb $0, (%[biased])\n\t"
          "je %l[tier2_pop]\n\t"
          "movl %%fs:4(%[rseq_off]), %k[tmp]\n\t"
          "movzwl 2(%[cam]), %k[slot]\n\t"
          "incl %k[tmp]\n\t"
          "cmpl %k[slot], %k[tmp]\n\t"
          "jne %l[slow_pop]\n\t"
          "movq 128(%[meta]), %[cur_h]\n\t"
          "cmpq 136(%[meta]), %[cur_h]\n\t"
          "jne 7f\n\t"
          "movq (%[meta]), %[tmp]\n\t"
          "cmpq %[tmp], %[cur_h]\n\t"
          "je %l[steal_pop]\n\t"
          "movq %[tmp], 136(%[meta])\n\t"
          "7:\n\t"
          "leaq 1(%[cur_h]), %[next_h]\n\t"
          "andq 152(%[meta]), %[cur_h]\n\t"
          "movq 144(%[meta]), %[tmp]\n\t"
          "movq (%[tmp], %[cur_h], 8), %[val_out]\n\t"
          "movq %[next_h], 128(%[meta])\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax\"\n\t"
          ".long 0x53053053\n\t"
          "4:\n\t"
          "jmp 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [slot] "=&r"(slot), [cur_h] "=&r"(cur_h),
            [next_h] "=&r"(next_h), [val_out] "=&r"(val_out)
          : [rseq_off] "r"(rseq_off), [biased] "r"(&cons_biased_),
            [cam] "r"(cons_cam_.data()), [meta] "r"(meta_.data())
          : "memory", "cc"
          : tier2_pop, steal_pop, slow_pop);
      out = val_out;
      prod_park_.notify_if_waiters();
      return true;
    }

  tier2_pop:
    {
      __m128i v_cpu{};  // NOLINT(misc-const-correctness)
      std::uint64_t tmp = 0;
      std::uint64_t slot = 0;
      std::uint64_t cur_h = 0;
      std::uint64_t next_h = 0;
      std::uint64_t val_out = 0;
      std::uintptr_t lp = 0;
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "leaq 1b(%%rip), %[tmp]\n\t"
          "movq %[tmp], %%fs:8(%[rseq_off])\n\t"
          "2:\n\t"
          "movl %%fs:4(%[rseq_off]), %k[tmp]\n\t"
          "cmpl $8192, %k[tmp]\n\t"
          "jae %l[slow_pop]\n\t"
          "incl %k[tmp]\n\t"
          "movd %k[tmp], %[v]\n\t"
          "pshuflw $0, %[v], %[v]\n\t"
          "punpcklqdq %[v], %[v]\n\t"
          "pcmpeqw (%[cam]), %[v]\n\t"
          "pmovmskb %[v], %k[slot]\n\t"
          "bsfl %k[slot], %k[slot]\n\t"
          "je %l[slow_pop]\n\t"
          "shrl $1, %k[slot]\n\t"
          "leaq -1(%[slot]), %[tmp]\n\t"
          "shlq $8, %[tmp]\n\t"
          "leaq (%[meta], %[tmp], 1), %[lp]\n\t"
          "cmpb $0, 160(%[lp])\n\t"
          "je %l[steal_pop]\n\t"
          "movq 128(%[lp]), %[cur_h]\n\t"
          "cmpq 136(%[lp]), %[cur_h]\n\t"
          "jne 7f\n\t"
          "movq (%[lp]), %[tmp]\n\t"
          "cmpq %[tmp], %[cur_h]\n\t"
          "je %l[steal_pop]\n\t"
          "movq %[tmp], 136(%[lp])\n\t"
          "7:\n\t"
          "leaq 1(%[cur_h]), %[next_h]\n\t"
          "andq 152(%[lp]), %[cur_h]\n\t"
          "movq 144(%[lp]), %[tmp]\n\t"
          "movq (%[tmp], %[cur_h], 8), %[val_out]\n\t"
          "movq %[next_h], 128(%[lp])\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax\"\n\t"
          ".long 0x53053053\n\t"
          "4:\n\t"
          "jmp 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [slot] "=&r"(slot), [cur_h] "=&r"(cur_h),
            [next_h] "=&r"(next_h), [lp] "=&r"(lp), [val_out] "=&r"(val_out),
            [v] "=&x"(v_cpu)
          : [rseq_off] "r"(rseq_off), [cam] "r"(cons_cam_.data()),
            [meta] "r"(meta_.data())
          : "memory", "cc"
          : steal_pop, slow_pop);
      out = val_out;
      prod_park_.notify_if_waiters();
      return true;
    }

  steal_pop:
    {
      const int st = steal_or_retry_own_mc_slow(out);
      if (st < 0) {
        goto retry_pop;
      }
      return st > 0;
    }
  slow_pop:
    if (try_assign_cons_slot_slow()) {
      goto retry_pop;
    }
    return steal_or_retry_own_mc_slow(out) > 0;
  }

  // Returns: `1` if `head_cache` refreshed with free space, `0` if lane is
  // genuinely full, `-1` if current CPU does not own a preferred slot.
  THETA_QUEUE_ALIGNED_COLD int refresh_push_state_slow() noexcept {
    const std::int32_t cpu = read_rseq_cpu_id();
    if (cpu >= 0 && cpu < kMaxRseqCpus) {
      const auto target = static_cast<std::uint16_t>(cpu + 1);
      for (std::uint8_t s = 1; s <= kMaxPreferred; ++s) {
        if (__atomic_load_n(&prod_cam_[s], __ATOMIC_RELAXED) == target) {
          LaneMeta& m = meta_[s - 1U];
          const std::uint64_t h = __atomic_load_n(&m.head, __ATOMIC_ACQUIRE);
          __atomic_store_n(&m.head_cache, h, __ATOMIC_RELAXED);
          if ((__atomic_load_n(&m.tail, __ATOMIC_RELAXED) - h) >= lane_cap_) {
            return 0;
          }
          return 1;
        }
      }
    }
    return -1;
  }

  THETA_QUEUE_ALIGNED_COLD bool try_assign_prod_slot_slow() noexcept {
    const std::int32_t cpu = read_rseq_cpu_id();
    if (cpu >= 0 && cpu < kMaxRseqCpus &&
        num_prod_.load(std::memory_order_relaxed) < kMaxPreferred) {
      bool assigned = false;
      const ColdSpinGuard guard(cam_lock_);
      const auto target = static_cast<std::uint16_t>(cpu + 1);
      for (std::uint8_t s = 1; s <= kMaxPreferred; ++s) {
        if (prod_cam_[s] == target) {
          assigned = true;
          break;
        }
      }
      if (!assigned) {
        const std::uint8_t n = num_prod_.load(std::memory_order_relaxed);
        if (n < kMaxPreferred) {
          const std::uint8_t s = n + 1U;
          __atomic_store_n(&prod_cam_[s], target, __ATOMIC_RELEASE);
          if (s == 1U) {
            __atomic_store_n(&prod_biased_, std::uint8_t{1}, __ATOMIC_RELEASE);
          } else if (__atomic_load_n(&prod_biased_, __ATOMIC_RELAXED) != 0U) {
            __atomic_store_n(&prod_biased_, std::uint8_t{0}, __ATOMIC_RELAXED);
            (void)asymmetric_membarrier_rseq_expedited();
          }
          num_prod_.store(s, std::memory_order_release);
          assigned = true;
        }
      }
      return assigned;
    }
    return false;
  }

  THETA_QUEUE_ALIGNED_COLD bool push_overflow_fallback_slow(
      std::uint64_t val) noexcept {
    const std::uint32_t ov =
        overflow_ops_.fetch_add(1U, std::memory_order_relaxed) + 1U;
    if ((ov & 4095U) == 0U) [[unlikely]] {
      maybe_recycle_prod_cam();
    }
    const bool ok = fallback_.try_push(unpack_word(val));
    if (ok) {
      cons_park_.notify_if_waiters();
    }
    return ok;
  }

  THETA_QUEUE_ALIGNED_COLD void maybe_recycle_prod_cam() noexcept {
    if (!cam_lock_.try_lock()) {
      return;
    }
    std::uint64_t tail_sum = 0;
    for (std::size_t i = 0; i < kMaxPreferred; ++i) {
      tail_sum += __atomic_load_n(&meta_[i].tail, __ATOMIC_RELAXED);
    }
    if (tail_sum == last_recycle_tail_sum_ &&
        num_prod_.load(std::memory_order_relaxed) == kMaxPreferred) {
      for (std::uint8_t s = 1; s <= kMaxPreferred; ++s) {
        __atomic_store_n(&prod_cam_[s], std::uint16_t{0}, __ATOMIC_RELAXED);
      }
      __atomic_store_n(&prod_biased_, std::uint8_t{0}, __ATOMIC_RELAXED);
      (void)asymmetric_membarrier_rseq_expedited();
      num_prod_.store(0U, std::memory_order_release);
    }
    last_recycle_tail_sum_ = tail_sum;
    cam_lock_.unlock();
  }

  THETA_QUEUE_ALIGNED_COLD bool try_assign_cons_slot_slow() noexcept {
    const std::int32_t cpu = read_rseq_cpu_id();
    if (cpu >= 0 && cpu < kMaxRseqCpus &&
        num_cons_.load(std::memory_order_relaxed) < kMaxPreferred) {
      bool assigned = false;
      const ColdSpinGuard guard(cam_lock_);
      const auto target = static_cast<std::uint16_t>(cpu + 1);
      for (std::uint8_t s = 1; s <= kMaxPreferred; ++s) {
        if (cons_cam_[s] == target) {
          assigned = true;
          break;
        }
      }
      if (!assigned) {
        const std::uint8_t n = num_cons_.load(std::memory_order_relaxed);
        if (n < kMaxPreferred) {
          const std::uint8_t s = n + 1U;
          __atomic_store_n(&cons_cam_[s], target, __ATOMIC_RELEASE);
          if (s == 1U && __atomic_load_n(&meta_[0].cons_exclusive,
                                         __ATOMIC_RELAXED) != 0U) {
            __atomic_store_n(&cons_biased_, std::uint8_t{1}, __ATOMIC_RELEASE);
          } else if (__atomic_load_n(&cons_biased_, __ATOMIC_RELAXED) != 0U) {
            __atomic_store_n(&cons_biased_, std::uint8_t{0}, __ATOMIC_RELAXED);
            (void)asymmetric_membarrier_rseq_expedited();
          }
          num_cons_.store(s, std::memory_order_release);
          assigned = true;
        }
      }
      return assigned;
    }
    return false;
  }

  // Returns: `1` if item stolen into `out`, `-1` if caller's own exclusive
  // `rseq` lane became non-empty (caller retries `rseq` without self-demoting
  // `cons_exclusive`!), `0` if all lanes are empty.
  THETA_QUEUE_ALIGNED_COLD int steal_or_retry_own_mc_slow(
      std::uint64_t& out) noexcept {
    if (overflow_ops_.load(std::memory_order_relaxed) != 0U) [[unlikely]] {
      T fb_item;
      if (fallback_.try_pop(fb_item)) {
        out = pack_word(fb_item);
        prod_park_.notify_if_waiters();
        return 1;
      }
    }
    const std::int32_t cpu = read_rseq_cpu_id();
    const std::uint16_t self_target = (cpu >= 0 && cpu < kMaxRseqCpus)
                                          ? static_cast<std::uint16_t>(cpu + 1)
                                          : std::uint16_t{0};

    thread_local std::uint8_t tls_cursor = 0;
    for (std::uint8_t step = 0; step < kMaxPreferred; ++step) {
      const auto idx =
          static_cast<std::uint8_t>((tls_cursor + step) % kMaxPreferred);
      LaneMeta& m = meta_[idx];
      std::uint64_t h = __atomic_load_n(&m.head, __ATOMIC_ACQUIRE);
      std::uint64_t t = __atomic_load_n(&m.tail, __ATOMIC_ACQUIRE);
      if (h == t) {
        continue;
      }

      if (__atomic_load_n(&m.cons_exclusive, __ATOMIC_ACQUIRE) != 0U) {
        // Never self-demote our own exclusive home lane!
        if (self_target != 0U &&
            __atomic_load_n(&cons_cam_[idx + 1U], __ATOMIC_RELAXED) ==
                self_target) {
          return -1;
        }
        const ColdSpinGuard guard(cam_lock_);
        if (__atomic_load_n(&m.cons_exclusive, __ATOMIC_RELAXED) != 0U) {
          __atomic_store_n(&cons_biased_, std::uint8_t{0}, __ATOMIC_RELAXED);
          __atomic_store_n(&m.cons_exclusive, std::uint8_t{0},
                           __ATOMIC_RELAXED);
          (void)asymmetric_membarrier_rseq_expedited();
        }
      }

      h = __atomic_load_n(&m.head, __ATOMIC_RELAXED);
      for (;;) {
        t = __atomic_load_n(&m.tail, __ATOMIC_ACQUIRE);
        if (h == t) {
          break;
        }
        const std::uint64_t val =
            __atomic_load_n(&m.data_cons[h & m.mask_cons], __ATOMIC_RELAXED);
        const std::uint64_t next_h = h + 1U;
        if (__atomic_compare_exchange_n(&m.head, &h, next_h, true,
                                        __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
          out = val;
          tls_cursor = idx;
          prod_park_.notify_if_waiters();
          return 1;
        }
      }
    }
    return 0;
  }

  THETA_QUEUE_ALIGNED_COLD void push_word_blocking_slow(
      std::uint64_t val) noexcept {
    bool pushed = false;
    prod_park_.wait_until([this, val, &pushed]() noexcept {
      if (!pushed) {
        pushed = try_push_word_fast(val);
      }
      return pushed;
    });
  }

  [[nodiscard]] THETA_QUEUE_ALIGNED_COLD std::uint64_t
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
#else
  // Unreachable two-phase template-lookup stubs when `kUseRseqLanes == false`
  // (e.g., under ASan/UBSan/TSan or non-x86 architectures).
  static bool try_push_word_fast(std::uint64_t /*val*/) noexcept {
    return false;
  }
  static bool try_pop_word_fast(std::uint64_t& /*out*/) noexcept {
    return false;
  }
  template <typename Func>
  static std::size_t drain_sc(Func& /*fn*/,
                              std::size_t /*max_count*/) noexcept {
    return 0U;
  }
  static void push_word_blocking_slow(std::uint64_t /*val*/) noexcept {}
  static std::uint64_t pop_word_blocking_slow() noexcept { return 0U; }
#endif

  const std::size_t lane_cap_;
  std::uint64_t* lane_buf_{nullptr};

  alignas(kHardwareDestructiveInterferenceSize)
      std::array<std::uint16_t, 8> prod_cam_{};
  std::uint8_t prod_biased_{0};

  alignas(kHardwareDestructiveInterferenceSize)
      std::array<std::uint16_t, 8> cons_cam_{};
  std::uint8_t cons_biased_{0};
  std::uint8_t sc_cursor_{0};
  std::uint8_t sc_quantum_{kScQuantum};

  alignas(kHardwareDestructiveInterferenceSize) ColdSpinLock cam_lock_;
  std::atomic<std::uint8_t> num_prod_{0};
  std::atomic<std::uint8_t> num_cons_{0};
  std::atomic<std::uint32_t> overflow_ops_{0};
  std::uint64_t last_recycle_tail_sum_{0};

  std::array<LaneMeta, kMaxPreferred> meta_{};
  FallbackQueue fallback_;
  AsymmetricParkLot prod_park_;
  AsymmetricParkLot cons_park_;
};

}  // namespace theta::queue_detail
