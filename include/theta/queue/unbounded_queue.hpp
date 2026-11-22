// FILENAME: include/theta/queue/unbounded_queue.hpp
// Copyright (c) 2023-2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
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

namespace theta {

// Elastic flat-slab unbounded concurrent queue (`UnboundedQueue`) with
// topology-specialized aliases:
// - `USPSCQueue<T, kLgSlab>` (`SingleProducer=true,  SingleConsumer=true`):
//   Wraps circularly in-place inside a single 128B-isolated flat slab while
//   `depth < 2^kLgSlab`, and chains/recycles flat slabs lock-free on bursts.
// - `UMPSCQueue<T, kLgSlab>` (`SingleProducer=false, SingleConsumer=true`):
//   `fetch_add` producer ticket + bounced hybrid `{turn, T}` slabs + zero-RMW
//   single-consumer slab reader + type-stable recycled slab pool.
// - `UMPMCQueue<T, kLgSlab>` (`SingleProducer=false, SingleConsumer=false`):
//   Type-stable pooled elastic slabs with zero per-op Hazard Pointers,
//   128B page-local slot bouncing, and contention-gated `PREFETCHW` hints.
template <QueueElement T, bool kSingleProducer = false,
          bool kSingleConsumer = false, std::size_t kLgSlab = 9>
class UnboundedQueue;

// ============================================================================
// Specialization 1: Unbounded SPSC (`USPSCQueue`) — Circular-Then-Chained Slabs
// ============================================================================
template <QueueElement T, std::size_t kLgSlab>
class UnboundedQueue<T, true, true, kLgSlab> {
  static_assert(kLgSlab >= 2 && kLgSlab <= 20,
                "kLgSlab must be in [2, 20] (4 .. 1,048,576 slots per slab)");

  static constexpr std::size_t kSlabCap = std::size_t{1} << kLgSlab;
  static constexpr std::size_t kMask = kSlabCap - 1U;
  using Slot = queue::detail::RawSlotStorage<T>;

  // 128B separation between `write_pos`/`next` header and `data[0]` prevents
  // Intel L2 adjacent-line spatial prefetcher false sharing.
  struct alignas(kHardwareDestructiveInterferenceSize) Slab {
    alignas(kHardwareDestructiveInterferenceSize)
        std::atomic<std::uint64_t> write_pos{0};
    std::atomic<Slab*> next{nullptr};
    Slab* free_next{nullptr};
    alignas(
        kHardwareDestructiveInterferenceSize) std::array<Slot, kSlabCap> data;
  };

  static Slab* alloc_slab() {
    Slab* raw = queue::detail::aligned_alloc_uninit<Slab>(1U);
    std::construct_at(&raw->write_pos, std::uint64_t{0});
    std::construct_at(&raw->next, nullptr);
    raw->free_next = nullptr;
    return raw;
  }

  static void free_slab(Slab* s) noexcept {
    queue::detail::aligned_free_uninit(s);
  }

 public:
  using value_type = T;

  UnboundedQueue() {
    Slab* initial = alloc_slab();
    prod_slab_ = initial;
    cons_slab_ = initial;
    latest_prod_slab_.store(initial, std::memory_order_relaxed);
    active_cons_slab_.store(initial, std::memory_order_relaxed);
  }

  ~UnboundedQueue() {
    while (try_pop().has_value()) {
    }
    Slab* cur = cons_slab_;
    while (cur != nullptr) {
      Slab* nxt = cur->next.load(std::memory_order_relaxed);
      free_slab(cur);
      cur = nxt;
    }
    Slab* f = free_head_.load(std::memory_order_relaxed);
    while (f != nullptr) {
      Slab* nxt = f->free_next;
      free_slab(f);
      f = nxt;
    }
  }

  UnboundedQueue(const UnboundedQueue&) = delete;
  UnboundedQueue& operator=(const UnboundedQueue&) = delete;
  UnboundedQueue(UnboundedQueue&&) = delete;
  UnboundedQueue& operator=(UnboundedQueue&&) = delete;

  template <typename... Args>
    requires std::constructible_from<T, Args...>
  void emplace(Args&&... args) noexcept(
      std::is_nothrow_constructible_v<T, Args...>) {
    Slab* s = prod_slab_;
    const std::uint64_t w = prod_w_;
    if (w - prod_r_cache_ < kSlabCap) [[likely]] {
      s->data[static_cast<std::size_t>(w) & kMask].construct(
          std::forward<Args>(args)...);
      prod_w_ = w + 1U;
      s->write_pos.store(w + 1U, std::memory_order_release);
      cons_park_.notify_if_waiters();
      return;
    }
    emplace_slow(s, w, std::forward<Args>(args)...);
  }

  void push(const T& val) noexcept(std::is_nothrow_copy_constructible_v<T>)
    requires std::copy_constructible<T>
  {
    emplace(val);
  }

  void push(T&& val) noexcept { emplace(std::move(val)); }

  bool try_push(const T& val) noexcept(std::is_nothrow_copy_constructible_v<T>)
    requires std::copy_constructible<T>
  {
    emplace(val);
    return true;
  }

  bool try_push(T&& val) noexcept {
    emplace(std::move(val));
    return true;
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
    Slab* s = prod_slab_;
    const std::uint64_t w = prod_w_;
    if (w + count - prod_r_cache_ <= kSlabCap) [[likely]] {
      for (std::size_t i = 0; i < count; ++i, ++first) {
        s->data[static_cast<std::size_t>(w + i) & kMask].construct(*first);
      }
      prod_w_ = w + count;
      s->write_pos.store(w + count, std::memory_order_release);
      cons_park_.notify_if_waiters();
      return count;
    }
    for (std::size_t i = 0; i < count; ++i, ++first) {
      emplace(*first);
    }
    return count;
  }

  std::optional<T> try_pop() noexcept {
    const std::uint64_t r = cons_r_;
    if (r != cons_w_cache_) [[likely]] {
      std::optional<T> out{cons_slab_->data[static_cast<std::size_t>(r) & kMask]
                               .move_and_destroy()};
      cons_r_ = r + 1U;
      if (cons_is_same_slab_) [[likely]] {
        active_read_pos_.store(r + 1U, std::memory_order_release);
      } else {
        total_popped_.store(r + 1U, std::memory_order_relaxed);
      }
      return out;
    }
    return pop_slow_opt(r);
  }

  bool try_pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
    const std::uint64_t r = cons_r_;
    if (r != cons_w_cache_) [[likely]] {
      cons_slab_->data[static_cast<std::size_t>(r) & kMask]
          .move_into_and_destroy(out);
      cons_r_ = r + 1U;
      if (cons_is_same_slab_) [[likely]] {
        active_read_pos_.store(r + 1U, std::memory_order_release);
      } else {
        total_popped_.store(r + 1U, std::memory_order_relaxed);
      }
      return true;
    }
    std::optional<T> slow = pop_slow_opt(r);
    if (!slow.has_value()) {
      return false;
    }
    out = std::move(*slow);
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
    std::size_t total = 0U;
    while (total < max_count) {
      const std::uint64_t r = cons_r_;
      if (r != cons_w_cache_) [[likely]] {
        const auto avail = static_cast<std::size_t>(cons_w_cache_ - r);
        const std::size_t limit = std::min(max_count - total, avail);
        std::size_t consumed = 0U;
        bool stop = false;
        for (; consumed < limit; ++consumed) {
          Slot& slot =
              cons_slab_->data[static_cast<std::size_t>(r + consumed) & kMask];
          T* p = slot.ptr();
          const bool cont =
              queue::detail::invoke_drain_visitor(fn, std::move(*p));
          std::destroy_at(p);
          if (!cont) {
            ++consumed;
            stop = true;
            break;
          }
        }
        cons_r_ = r + consumed;
        if (cons_is_same_slab_) [[likely]] {
          active_read_pos_.store(r + consumed, std::memory_order_release);
        } else {
          total_popped_.store(r + consumed, std::memory_order_relaxed);
        }
        total += consumed;
        if (stop) {
          break;
        }
        continue;
      }
      std::optional<T> slow = pop_slow_opt(r);
      if (!slow.has_value()) {
        break;
      }
      ++total;
      if (!queue::detail::invoke_drain_visitor(fn, std::move(*slow))) {
        break;
      }
    }
    return total;
  }

  T pop() noexcept {
    if (std::optional<T> fast = try_pop(); fast.has_value()) [[likely]] {
      return std::move(*fast);
    }
    std::optional<T> res;
    cons_park_.wait_until([this, &res]() noexcept {
      if (!res.has_value()) {
        res = try_pop();
      }
      return res.has_value();
    });
    if (!res.has_value()) {
      std::unreachable();
    }
    return std::move(*res);
  }

  void pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
    out = pop();
  }

  [[nodiscard]] std::size_t size() const noexcept {
    const std::uint64_t r_same =
        active_read_pos_.load(std::memory_order_acquire);
    const std::uint64_t r_spill = total_popped_.load(std::memory_order_acquire);
    const std::uint64_t r = r_same > r_spill ? r_same : r_spill;
    const Slab* ps = latest_prod_slab_.load(std::memory_order_acquire);
    const std::uint64_t w = ps->write_pos.load(std::memory_order_acquire);
    return w > r ? static_cast<std::size_t>(w - r) : 0U;
  }

  [[nodiscard]] bool empty() const noexcept { return size() == 0U; }

 private:
  template <typename... Args>
  THETA_ALIGNED_NOINLINE void emplace_slow(
      Slab* s, std::uint64_t w,
      Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>) {
    if (s == active_cons_slab_.load(std::memory_order_acquire)) {
      prod_r_cache_ = active_read_pos_.load(std::memory_order_acquire);
      if (w - prod_r_cache_ < kSlabCap) [[likely]] {
        s->data[static_cast<std::size_t>(w) & kMask].construct(
            std::forward<Args>(args)...);
        prod_w_ = w + 1U;
        s->write_pos.store(w + 1U, std::memory_order_release);
        cons_park_.notify_if_waiters();
        return;
      }
    }
    Slab* ns = pop_free_slab();
    ns->next.store(nullptr, std::memory_order_relaxed);
    ns->data[static_cast<std::size_t>(w) & kMask].construct(
        std::forward<Args>(args)...);
    ns->write_pos.store(w + 1U, std::memory_order_release);
    prod_slab_ = ns;
    prod_w_ = w + 1U;
    prod_r_cache_ = w;
    latest_prod_slab_.store(ns, std::memory_order_release);
    s->next.store(ns, std::memory_order_release);
    cons_park_.notify_if_waiters();
  }

  THETA_ALIGNED_NOINLINE std::optional<T> pop_slow_opt(
      std::uint64_t r) noexcept {
    Slab* s = cons_slab_;
    std::uint64_t w = s->write_pos.load(std::memory_order_acquire);
    if (r != w) [[likely]] {
      cons_w_cache_ = w;
      cons_is_same_slab_ = (s->next.load(std::memory_order_relaxed) == nullptr);
      std::optional<T> out{
          s->data[static_cast<std::size_t>(r) & kMask].move_and_destroy()};
      cons_r_ = r + 1U;
      if (cons_is_same_slab_) {
        active_read_pos_.store(r + 1U, std::memory_order_release);
      } else {
        total_popped_.store(r + 1U, std::memory_order_relaxed);
      }
      return out;
    }
    Slab* nxt = s->next.load(std::memory_order_acquire);
    if (nxt == nullptr) {
      return std::nullopt;
    }
    w = s->write_pos.load(std::memory_order_acquire);
    if (r != w) {
      cons_w_cache_ = w;
      cons_is_same_slab_ = false;
      std::optional<T> out{
          s->data[static_cast<std::size_t>(r) & kMask].move_and_destroy()};
      cons_r_ = r + 1U;
      total_popped_.store(r + 1U, std::memory_order_relaxed);
      return out;
    }
    cons_slab_ = nxt;
    active_read_pos_.store(r, std::memory_order_release);
    total_popped_.store(r, std::memory_order_relaxed);
    active_cons_slab_.store(nxt, std::memory_order_release);
    push_free_slab(s);

    cons_w_cache_ = nxt->write_pos.load(std::memory_order_acquire);
    cons_is_same_slab_ = (nxt->next.load(std::memory_order_relaxed) == nullptr);
    std::optional<T> out{
        nxt->data[static_cast<std::size_t>(r) & kMask].move_and_destroy()};
    cons_r_ = r + 1U;
    if (cons_is_same_slab_) {
      active_read_pos_.store(r + 1U, std::memory_order_release);
    } else {
      total_popped_.store(r + 1U, std::memory_order_relaxed);
    }
    return out;
  }

  Slab* pop_free_slab() {
    Slab* head = free_head_.load(std::memory_order_acquire);
    while (head != nullptr) {
      if (free_head_.compare_exchange_weak(head, head->free_next,
                                           std::memory_order_acquire,
                                           std::memory_order_acquire)) {
        return head;
      }
    }
    return alloc_slab();
  }

  void push_free_slab(Slab* s) noexcept {
    Slab* head = free_head_.load(std::memory_order_relaxed);
    do {
      s->free_next = head;
    } while (!free_head_.compare_exchange_weak(
        head, s, std::memory_order_release, std::memory_order_relaxed));
  }

  alignas(kHardwareDestructiveInterferenceSize) Slab* prod_slab_{nullptr};
  std::uint64_t prod_w_{0};
  std::uint64_t prod_r_cache_{0};
  std::atomic<Slab*> latest_prod_slab_{nullptr};

  alignas(kHardwareDestructiveInterferenceSize) Slab* cons_slab_{nullptr};
  std::uint64_t cons_r_{0};
  std::uint64_t cons_w_cache_{0};
  bool cons_is_same_slab_{true};
  std::atomic<std::uint64_t> total_popped_{0};

  alignas(kHardwareDestructiveInterferenceSize)
      std::atomic<std::uint64_t> active_read_pos_{0};
  std::atomic<Slab*> active_cons_slab_{nullptr};
  alignas(kHardwareDestructiveInterferenceSize) std::atomic<Slab*> free_head_{
      nullptr};
  queue::detail::AsymmetricParkLot cons_park_;
};

// ============================================================================
// Specialization 2: Multi-Producer Unbounded (`UMPSCQueue` & `UMPMCQueue`)
//   - Type-stable pooled elastic slabs (zero Hazard Pointers on push/pop)
//   - Hybrid fused/separated `{turn, T}` storage + 128B page-local bounce
//   - Single-consumer specialization (`kSingleConsumer == true`) elides
//     consumer `CAS` and `consumed` RMWs on `try_pop`.
// ============================================================================
template <QueueElement T, bool kSingleProducer, bool kSingleConsumer,
          std::size_t kLgSlab>
class UnboundedQueue {
  static_assert(
      !kSingleProducer || !kSingleConsumer,
      "SPSC should match the <T, true, true, kLgSlab> specialization");
  static_assert(kLgSlab >= 2 && kLgSlab <= 20,
                "kLgSlab must be in [2, 20] (4 .. 1,048,576 slots per slab)");

  static constexpr std::size_t kSlabCap = std::size_t{1} << kLgSlab;
  static constexpr std::size_t kMask = kSlabCap - 1U;
  static constexpr std::uint32_t kFlagAllConsumed = 1U;
  static constexpr std::uint32_t kFlagIsHead = 2U;
  static constexpr std::uint32_t kFlagReadyToRecycle =
      kFlagAllConsumed | kFlagIsHead;

  struct Slab {
    std::atomic<std::uint64_t> slab_id{0};
    std::atomic<Slab*> next{nullptr};
    std::atomic<std::size_t> consumed{0};
    std::atomic<std::uint32_t> retire_state{0};
    Slab* pool_next{nullptr};
    Slab* all_next{nullptr};
    queue::detail::TurnSlotArray<T> slots;

    explicit Slab(std::uint64_t id) : slab_id(id), slots(kSlabCap) {}
  };

 public:
  using value_type = T;

  UnboundedQueue() {
    Slab* s = alloc_fresh_slab(0U);
    s->retire_state.store(kFlagIsHead, std::memory_order_relaxed);
    head_slab_.store(s, std::memory_order_relaxed);
    tail_slab_.store(s, std::memory_order_relaxed);
  }

  ~UnboundedQueue() {
    while (try_pop().has_value()) {
    }
    const Slab* cur = all_slabs_.load(std::memory_order_relaxed);
    while (cur != nullptr) {
      const Slab* nxt = cur->all_next;
      delete cur;
      cur = nxt;
    }
  }

  UnboundedQueue(const UnboundedQueue&) = delete;
  UnboundedQueue& operator=(const UnboundedQueue&) = delete;
  UnboundedQueue(UnboundedQueue&&) = delete;
  UnboundedQueue& operator=(UnboundedQueue&&) = delete;

  template <typename... Args>
    requires std::constructible_from<T, Args...>
  void emplace(Args&&... args) noexcept(
      std::is_nothrow_constructible_v<T, Args...>) {
    Slab* s = tail_slab_.load(std::memory_order_acquire);
    const std::uint64_t t =
        kSingleProducer ? fetch_inc_sp(tail_)
                        : tail_.fetch_add(1U, std::memory_order_relaxed);
    const std::uint64_t want_id = t >> kLgSlab;
    if (s->slab_id.load(std::memory_order_acquire) != want_id) [[unlikely]] {
      s = find_or_create_slab(s, want_id);
    }
    const std::size_t phys = queue::detail::remap_slot_index(
        static_cast<std::size_t>(t) & kMask, kLgSlab);
    if (pf_gate_.should_prefetch()) {
      s->slots.prefetch_write(phys);
    }
    s->slots.storage(phys).construct(std::forward<Args>(args)...);
    s->slots.turn(phys).store(t + 1U, std::memory_order_release);
    cons_park_.notify_if_waiters();
    if (((t + 1U) & kMask) == 0U) [[unlikely]] {
      advance_tail_slab(s, want_id + 1U);
    }
  }

  void push(const T& val) noexcept(std::is_nothrow_copy_constructible_v<T>)
    requires std::copy_constructible<T>
  {
    emplace(val);
  }

  void push(T&& val) noexcept { emplace(std::move(val)); }

  bool try_push(const T& val) noexcept(std::is_nothrow_copy_constructible_v<T>)
    requires std::copy_constructible<T>
  {
    emplace(val);
    return true;
  }

  bool try_push(T&& val) noexcept {
    emplace(std::move(val));
    return true;
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
    Slab* s = tail_slab_.load(std::memory_order_acquire);
    const std::uint64_t base =
        kSingleProducer ? fetch_add_sp(tail_, count)
                        : tail_.fetch_add(count, std::memory_order_relaxed);
    const bool do_pf = pf_gate_.should_prefetch();

    for (std::size_t i = 0; i < count; ++i, ++first) {
      const std::uint64_t t = base + i;
      const std::uint64_t want_id = t >> kLgSlab;
      if (s->slab_id.load(std::memory_order_acquire) != want_id) [[unlikely]] {
        s = find_or_create_slab(s, want_id);
      }
      const std::size_t phys = queue::detail::remap_slot_index(
          static_cast<std::size_t>(t) & kMask, kLgSlab);
      if (do_pf) {
        s->slots.prefetch_write(phys);
      }
      s->slots.storage(phys).construct(*first);
      s->slots.turn(phys).store(t + 1U, std::memory_order_release);
      if (((t + 1U) & kMask) == 0U) [[unlikely]] {
        advance_tail_slab(s, want_id + 1U);
      }
    }
    cons_park_.notify_if_waiters();
    return count;
  }

  std::optional<T> try_pop() noexcept {
    if constexpr (kSingleConsumer) {
      return try_pop_sc();
    } else {
      return try_pop_mc();
    }
  }

  bool try_pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
    std::optional<T> res = try_pop();
    if (!res.has_value()) {
      return false;
    }
    out = std::move(*res);
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

  T pop() noexcept {
    if (std::optional<T> fast = try_pop(); fast.has_value()) [[likely]] {
      return std::move(*fast);
    }
    std::optional<T> res;
    cons_park_.wait_until([this, &res]() noexcept {
      if (!res.has_value()) {
        res = try_pop();
      }
      return res.has_value();
    });
    if (!res.has_value()) {
      std::unreachable();
    }
    return std::move(*res);
  }

  void pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
    out = pop();
  }

  [[nodiscard]] std::size_t size() const noexcept {
    const std::uint64_t h = head_.load(std::memory_order_acquire);
    const std::uint64_t t = tail_.load(std::memory_order_acquire);
    return t > h ? static_cast<std::size_t>(t - h) : 0U;
  }

  [[nodiscard]] bool empty() const noexcept { return size() == 0U; }

 private:
  static THETA_FORCE_INLINE std::uint64_t fetch_inc_sp(
      std::atomic<std::uint64_t>& v) noexcept {
    const std::uint64_t cur = v.load(std::memory_order_relaxed);
    v.store(cur + 1U, std::memory_order_release);
    return cur;
  }

  static THETA_FORCE_INLINE std::uint64_t fetch_add_sp(
      std::atomic<std::uint64_t>& v, std::size_t n) noexcept {
    const std::uint64_t cur = v.load(std::memory_order_relaxed);
    v.store(cur + n, std::memory_order_release);
    return cur;
  }

  std::optional<T> try_pop_sc() noexcept {
    const std::uint64_t c = head_.load(std::memory_order_relaxed);
    const std::uint64_t want_id = c >> kLgSlab;
    Slab* s = head_slab_.load(std::memory_order_acquire);
    if (s->slab_id.load(std::memory_order_acquire) != want_id) [[unlikely]] {
      s = find_existing_slab(want_id);
      if (s == nullptr) {
        return std::nullopt;
      }
    }
    const std::size_t phys = queue::detail::remap_slot_index(
        static_cast<std::size_t>(c) & kMask, kLgSlab);
    if (s->slots.turn(phys).load(std::memory_order_acquire) != c + 1U) {
      return std::nullopt;
    }
    head_.store(c + 1U, std::memory_order_release);
    std::optional<T> out{s->slots.storage(phys).move_and_destroy()};
    if (((c + 1U) & kMask) == 0U) [[unlikely]] {
      retire_full_slab(s, want_id);
    }
    return out;
  }

  std::optional<T> try_pop_mc() noexcept {
    std::uint64_t c = head_.load(std::memory_order_relaxed);
    Slab* s = head_slab_.load(std::memory_order_acquire);
    for (;;) {
      const std::uint64_t want_id = c >> kLgSlab;
      if (s->slab_id.load(std::memory_order_acquire) != want_id) [[unlikely]] {
        s = find_existing_slab(want_id);
        if (s == nullptr) {
          c = head_.load(std::memory_order_relaxed);
          s = head_slab_.load(std::memory_order_acquire);
          if ((c >> kLgSlab) > s->slab_id.load(std::memory_order_acquire)) {
            continue;
          }
          return std::nullopt;
        }
      }
      const std::size_t phys = queue::detail::remap_slot_index(
          static_cast<std::size_t>(c) & kMask, kLgSlab);
      const std::uint64_t ready_tag =
          s->slots.turn(phys).load(std::memory_order_acquire);
      if (ready_tag == c + 1U) [[likely]] {
        if (head_.compare_exchange_weak(c, c + 1U, std::memory_order_relaxed,
                                        std::memory_order_acquire)) [[likely]] {
          std::optional<T> out{s->slots.storage(phys).move_and_destroy()};
          if (s->consumed.fetch_add(1U, std::memory_order_acq_rel) + 1U ==
              kSlabCap) [[unlikely]] {
            retire_full_slab(s, want_id);
          }
          return out;
        }
      } else {
        const std::uint64_t cur_tail = tail_.load(std::memory_order_acquire);
        if (c >= cur_tail) {
          return std::nullopt;
        }
        const std::uint64_t fresh_c = head_.load(std::memory_order_relaxed);
        if (fresh_c == c) {
          return std::nullopt;
        }
        c = fresh_c;
      }
    }
  }

  THETA_ALIGNED_NOINLINE Slab* find_or_create_slab(Slab* s,
                                                   std::uint64_t want_id) {
    for (;;) {
      const std::uint64_t cur_id = s->slab_id.load(std::memory_order_acquire);
      if (cur_id == want_id) {
        return s;
      }
      if (cur_id > want_id) {
        s = head_slab_.load(std::memory_order_acquire);
        continue;
      }
      Slab* nxt = s->next.load(std::memory_order_acquire);
      if (nxt == nullptr) {
        Slab* cand = pop_or_alloc_slab(cur_id + 1U);
        if (s->slab_id.load(std::memory_order_acquire) != cur_id) {
          cand->slab_id.store(~std::uint64_t{0}, std::memory_order_relaxed);
          push_pool(cand);
          s = head_slab_.load(std::memory_order_acquire);
          continue;
        }
        if (s->next.compare_exchange_strong(nxt, cand,
                                            std::memory_order_release,
                                            std::memory_order_acquire)) {
          nxt = cand;
        } else {
          cand->slab_id.store(~std::uint64_t{0}, std::memory_order_relaxed);
          push_pool(cand);
        }
      }
      s = nxt;
    }
  }

  THETA_ALIGNED_NOINLINE void advance_tail_slab(Slab* s,
                                                std::uint64_t next_id) {
    if (next_id <= latest_tail_id_.load(std::memory_order_acquire)) {
      return;
    }
    for (;;) {
      const std::uint64_t cur_id = s->slab_id.load(std::memory_order_acquire);
      if (cur_id == next_id) {
        promote_tail_monotonic(s, next_id);
        return;
      }
      if (cur_id > next_id) {
        s = head_slab_.load(std::memory_order_acquire);
        if (s->slab_id.load(std::memory_order_acquire) > next_id) {
          return;
        }
        continue;
      }
      Slab* nxt = s->next.load(std::memory_order_acquire);
      if (nxt == nullptr) {
        Slab* cand = pop_or_alloc_slab(cur_id + 1U);
        if (s->slab_id.load(std::memory_order_acquire) != cur_id) {
          cand->slab_id.store(~std::uint64_t{0}, std::memory_order_relaxed);
          push_pool(cand);
          s = head_slab_.load(std::memory_order_acquire);
          if (s->slab_id.load(std::memory_order_acquire) > next_id) {
            return;
          }
          continue;
        }
        if (s->next.compare_exchange_strong(nxt, cand,
                                            std::memory_order_release,
                                            std::memory_order_acquire)) {
          nxt = cand;
        } else {
          cand->slab_id.store(~std::uint64_t{0}, std::memory_order_relaxed);
          push_pool(cand);
        }
      }
      s = nxt;
    }
  }

  void promote_tail_monotonic(Slab* nxt, std::uint64_t next_id) noexcept {
    if (next_id <= latest_tail_id_.load(std::memory_order_acquire)) {
      return;
    }
    const queue::detail::ColdSpinGuard g(pool_lock_);
    if (next_id > latest_tail_id_.load(std::memory_order_relaxed)) {
      tail_slab_.store(nxt, std::memory_order_release);
      latest_tail_id_.store(next_id, std::memory_order_release);
    }
  }

  [[nodiscard]] THETA_ALIGNED_NOINLINE Slab* find_existing_slab(
      std::uint64_t want_id) const noexcept {
    Slab* s = head_slab_.load(std::memory_order_acquire);
    while (s != nullptr) {
      const std::uint64_t id = s->slab_id.load(std::memory_order_acquire);
      if (id == want_id) {
        return s;
      }
      if (id > want_id) {
        s = head_slab_.load(std::memory_order_acquire);
        if (s->slab_id.load(std::memory_order_acquire) > want_id) {
          return nullptr;
        }
        continue;
      }
      s = s->next.load(std::memory_order_acquire);
    }
    return nullptr;
  }

  THETA_ALIGNED_NOINLINE void retire_full_slab(Slab* s, std::uint64_t slab_id) {
    const std::uint32_t prev =
        s->retire_state.fetch_or(kFlagAllConsumed, std::memory_order_acq_rel);
    if ((prev | kFlagAllConsumed) != kFlagReadyToRecycle) {
      return;
    }
    for (;;) {
      Slab* nxt = find_or_create_slab(s, slab_id + 1U);
      promote_tail_monotonic(nxt, slab_id + 1U);
      head_slab_.store(nxt, std::memory_order_release);
      s->slab_id.store(~std::uint64_t{0}, std::memory_order_release);
      push_pool(s);

      const std::uint32_t np =
          nxt->retire_state.fetch_or(kFlagIsHead, std::memory_order_acq_rel);
      if ((np | kFlagIsHead) != kFlagReadyToRecycle) {
        break;
      }
      s = nxt;
      ++slab_id;
    }
  }

  Slab* alloc_fresh_slab(std::uint64_t id) {
    auto* s = new Slab(id);
    Slab* head = all_slabs_.load(std::memory_order_relaxed);
    do {
      s->all_next = head;
    } while (!all_slabs_.compare_exchange_weak(
        head, s, std::memory_order_release, std::memory_order_relaxed));
    return s;
  }

  Slab* pop_or_alloc_slab(std::uint64_t new_id) {
    {
      const queue::detail::ColdSpinGuard g(pool_lock_);
      if (pool_count_ > 2U && pool_head_ != nullptr) {
        Slab* s = pool_head_;
        pool_head_ = s->pool_next;
        if (pool_head_ == nullptr) {
          pool_tail_ = nullptr;
        }
        --pool_count_;
        s->consumed.store(0U, std::memory_order_relaxed);
        s->retire_state.store(0U, std::memory_order_relaxed);
        s->next.store(nullptr, std::memory_order_relaxed);
        s->slab_id.store(new_id, std::memory_order_release);
        return s;
      }
    }
    return alloc_fresh_slab(new_id);
  }

  void push_pool(Slab* s) noexcept {
    const queue::detail::ColdSpinGuard g(pool_lock_);
    s->pool_next = nullptr;
    if (pool_tail_ != nullptr) {
      pool_tail_->pool_next = s;
    } else {
      pool_head_ = s;
    }
    pool_tail_ = s;
    ++pool_count_;
  }

  queue::detail::ContentionPrefetchGate pf_gate_;

  alignas(kHardwareDestructiveInterferenceSize) std::atomic<Slab*> head_slab_{
      nullptr};
  std::atomic<std::uint64_t> head_{0};

  alignas(kHardwareDestructiveInterferenceSize) std::atomic<Slab*> tail_slab_{
      nullptr};
  std::atomic<std::uint64_t> tail_{0};
  std::atomic<std::uint64_t> latest_tail_id_{0};

  alignas(kHardwareDestructiveInterferenceSize) queue::detail::ColdSpinLock
      pool_lock_;
  Slab* pool_head_ THETA_QUEUE_GUARDED_BY(pool_lock_){nullptr};
  Slab* pool_tail_ THETA_QUEUE_GUARDED_BY(pool_lock_){nullptr};
  std::size_t pool_count_ THETA_QUEUE_GUARDED_BY(pool_lock_){0};
  std::atomic<Slab*> all_slabs_{nullptr};

  queue::detail::AsymmetricParkLot cons_park_;
};

template <QueueElement T, std::size_t kLgSlab = 9>
using USPSCQueue = UnboundedQueue<T, true, true, kLgSlab>;

template <QueueElement T, std::size_t kLgSlab = 9>
using UMPSCQueue = UnboundedQueue<T, false, true, kLgSlab>;

template <QueueElement T, std::size_t kLgSlab = 9>
using UMPMCQueue = UnboundedQueue<T, false, false, kLgSlab>;

}  // namespace theta
