// FILENAME: include/theta/queue/spsc_queue.hpp
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

// Bounded Single-Producer, Single-Consumer lock-free ring queue.
// Uses pre-wrapped [0 .. N-1] indices (capacity N - 1) so array accesses avoid
// masking instructions, linear slot layout for L1D write-combining,
// 128B-isolated producer/consumer cache lines with local head/tail shadows,
// frameless 2-tier asymmetric futex wakeup, and zero-copy bulk
// `try_push_bulk` / `try_pop_bulk` / `drain` (`Flusher`) operations.
template <QueueElement T, std::size_t kStaticBufferSize = 0>
class SPSCQueue {
  static_assert(kStaticBufferSize == 0 ||
                    (std::has_single_bit(kStaticBufferSize) &&
                     kStaticBufferSize >= 2),
                "kStaticBufferSize must be 0 (dynamic) or a power of 2 >= 2");

  friend struct queue::detail::QueueTestPeer;
  using Slot = queue::detail::RawSlotStorage<T>;
  static constexpr std::size_t kSlotAlign = std::max<std::size_t>(
      kHardwareDestructiveInterferenceSize, alignof(Slot));

  static constexpr std::size_t resolve_ring_len(
      const QueueOpts& opts) noexcept {
    if constexpr (kStaticBufferSize != 0) {
      return kStaticBufferSize;
    } else {
      constexpr std::size_t kMinSlots = 2;
      const std::size_t want = opts.max_size() < (SIZE_MAX / 2)
                                   ? (opts.max_size() + 1U)
                                   : opts.max_size();
      return std::bit_ceil(std::max(want, kMinSlots));
    }
  }

 public:
  using value_type = T;

  explicit SPSCQueue(const QueueOpts& opts = QueueOpts{})
      : mask_(resolve_ring_len(opts) - 1U),
        slots_(
            queue::detail::aligned_alloc_uninit<Slot>(resolve_ring_len(opts))) {
  }

  ~SPSCQueue() {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      std::size_t h = head_.load(std::memory_order_relaxed);
      const std::size_t t = tail_.load(std::memory_order_relaxed);
      const std::size_t m = ring_mask();
      while (h != t) {
        slots_[h].destroy();
        h = (h + 1U) & m;
      }
    }
    queue::detail::destroy_spsc_core(
        slots_, kSlotAlign, prod_park_.load(std::memory_order_relaxed),
        cons_park_.load(std::memory_order_relaxed));
  }

  SPSCQueue(const SPSCQueue&) = delete;
  SPSCQueue& operator=(const SPSCQueue&) = delete;
  SPSCQueue(SPSCQueue&&) = delete;
  SPSCQueue& operator=(SPSCQueue&&) = delete;

  template <typename... Args>
    requires std::constructible_from<T, Args...>
  bool try_emplace(Args&&... args) noexcept(
      std::is_nothrow_constructible_v<T, Args...>) {
    const std::size_t m = ring_mask();
    const std::size_t t = tail_.load(std::memory_order_relaxed);
    const std::size_t next = (t + 1U) & m;
    if (next == head_cache_) [[unlikely]] {
      const std::size_t hc = head_.load(std::memory_order_acquire);
      if (next == hc) {
        return false;
      }
      head_cache_ = hc;
    }
    slots_[t].construct(std::forward<Args>(args)...);
    tail_.store(next, std::memory_order_release);
    queue::detail::AsymmetricParkLot::notify_slot(cons_park_);
    return true;
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
    const std::size_t m = ring_mask();
    const std::size_t ring_len = m + 1U;
    const std::size_t t = tail_.load(std::memory_order_relaxed);
    std::size_t avail = (head_cache_ - t - 1U) & m;
    if (avail < count) {
      head_cache_ = head_.load(std::memory_order_acquire);
      avail = (head_cache_ - t - 1U) & m;
      if (avail == 0U) {
        return 0U;
      }
    }
    const std::size_t n = std::min(count, avail);
    const std::size_t first_span = std::min(n, ring_len - t);
    const std::size_t second_span = n - first_span;

    if constexpr (std::is_trivially_copyable_v<T> &&
                  (std::same_as<InputIt, const T*> ||
                   std::same_as<InputIt, T*>)) {
      std::memcpy(slots_ + t, first, first_span * sizeof(T));
      if (second_span > 0U) {
        std::memcpy(slots_, first + first_span, second_span * sizeof(T));
      }
    } else {
      for (std::size_t i = 0; i < first_span; ++i, ++first) {
        slots_[t + i].construct(*first);
      }
      for (std::size_t i = 0; i < second_span; ++i, ++first) {
        slots_[i].construct(*first);
      }
    }

    tail_.store((t + n) & m, std::memory_order_release);
    queue::detail::AsymmetricParkLot::notify_slot(cons_park_);
    return n;
  }

  template <typename... Args>
    requires std::constructible_from<T, Args...>
  void emplace(Args&&... args) noexcept(
      std::is_nothrow_constructible_v<T, Args...>) {
    const std::size_t m = ring_mask();
    const std::size_t t = tail_.load(std::memory_order_relaxed);
    const std::size_t next = (t + 1U) & m;
    if (next == head_cache_) [[unlikely]] {
      head_cache_ = head_.load(std::memory_order_acquire);
      if (next == head_cache_) {
        const std::atomic<std::size_t>* const head_ptr = &head_;
        queue::detail::AsymmetricParkLot::wait_slot_until(
            prod_park_, [head_ptr, next]() noexcept {
              return head_ptr->load(std::memory_order_acquire) != next;
            });
        head_cache_ = head_.load(std::memory_order_acquire);
      }
    }
    slots_[t].construct(std::forward<Args>(args)...);
    tail_.store(next, std::memory_order_release);
    queue::detail::AsymmetricParkLot::notify_slot(cons_park_);
  }

  void push(const T& val) noexcept(std::is_nothrow_copy_constructible_v<T>)
    requires std::copy_constructible<T>
  {
    emplace(val);
  }

  void push(T&& val) noexcept { emplace(std::move(val)); }

  std::optional<T> try_pop() noexcept {
    const std::size_t h = head_.load(std::memory_order_relaxed);
    if (h == tail_cache_) {
      const std::size_t tc = tail_.load(std::memory_order_acquire);
      if (h == tc) {
        return std::nullopt;
      }
      tail_cache_ = tc;
    }
    std::optional<T> out{slots_[h].move_and_destroy()};
    head_.store((h + 1U) & ring_mask(), std::memory_order_release);
    queue::detail::AsymmetricParkLot::notify_slot(prod_park_);
    return out;
  }

  bool try_pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
    const std::size_t h = head_.load(std::memory_order_relaxed);
    if (h == tail_cache_) {
      const std::size_t tc = tail_.load(std::memory_order_acquire);
      if (h == tc) {
        return false;
      }
      tail_cache_ = tc;
    }
    slots_[h].move_into_and_destroy(out);
    head_.store((h + 1U) & ring_mask(), std::memory_order_release);
    queue::detail::AsymmetricParkLot::notify_slot(prod_park_);
    return true;
  }

  template <typename OutputIt>
  std::size_t
  try_pop_bulk(OutputIt out, std::size_t max_count = SIZE_MAX) noexcept(
      std::is_nothrow_move_assignable_v<T>) {
    if (max_count == 0U) {
      return 0U;
    }
    const std::size_t m = ring_mask();
    const std::size_t ring_len = m + 1U;
    const std::size_t h = head_.load(std::memory_order_relaxed);
    std::size_t avail = (tail_cache_ - h) & m;
    if (avail == 0U) {
      tail_cache_ = tail_.load(std::memory_order_acquire);
      avail = (tail_cache_ - h) & m;
      if (avail == 0U) {
        return 0U;
      }
    }
    const std::size_t n = std::min(max_count, avail);
    const std::size_t first_span = std::min(n, ring_len - h);
    const std::size_t second_span = n - first_span;

    if constexpr (std::is_trivially_copyable_v<T> &&
                  std::same_as<OutputIt, T*>) {
      std::memcpy(out, slots_ + h, first_span * sizeof(T));
      if (second_span > 0U) {
        std::memcpy(out + first_span, slots_, second_span * sizeof(T));
      }
    } else {
      for (std::size_t i = 0; i < first_span; ++i, ++out) {
        *out = slots_[h + i].move_and_destroy();
      }
      for (std::size_t i = 0; i < second_span; ++i, ++out) {
        *out = slots_[i].move_and_destroy();
      }
    }

    head_.store((h + n) & m, std::memory_order_release);
    queue::detail::AsymmetricParkLot::notify_slot(prod_park_);
    return n;
  }

  template <typename Func>
  std::size_t drain(Func&& fn, std::size_t max_count = SIZE_MAX) noexcept(
      std::is_nothrow_invocable_v<Func&, T&&>) {
    if (max_count == 0U) {
      return 0U;
    }
    const std::size_t m = ring_mask();
    const std::size_t h = head_.load(std::memory_order_relaxed);
    std::size_t avail = (tail_cache_ - h) & m;
    if (avail == 0U) {
      tail_cache_ = tail_.load(std::memory_order_acquire);
      avail = (tail_cache_ - h) & m;
      if (avail == 0U) {
        return 0U;
      }
    }
    const std::size_t limit = std::min(max_count, avail);
    std::size_t consumed = 0U;

    const struct CommitGuard {
      SPSCQueue* self;
      std::size_t start_h;
      std::size_t mask;
      std::size_t* count;
      ~CommitGuard() noexcept {
        if (*count > 0U) {
          self->head_.store((start_h + *count) & mask,
                            std::memory_order_release);
          queue::detail::AsymmetricParkLot::notify_slot(self->prod_park_);
        }
      }
    } commit{this, h, m, &consumed};

    for (; consumed < limit; ++consumed) {
      Slot& slot = slots_[(h + consumed) & m];
      T* p = slot.ptr();
      const bool cont = queue::detail::invoke_drain_visitor(fn, std::move(*p));
      std::destroy_at(p);
      if (!cont) {
        ++consumed;
        break;
      }
    }
    return consumed;
  }

  T pop() noexcept {
    const std::size_t h = head_.load(std::memory_order_relaxed);
    if (h == tail_cache_) {
      tail_cache_ = tail_.load(std::memory_order_acquire);
      if (h == tail_cache_) {
        const std::atomic<std::size_t>* const tail_ptr = &tail_;
        queue::detail::AsymmetricParkLot::wait_slot_until(
            cons_park_, [tail_ptr, h]() noexcept {
              return tail_ptr->load(std::memory_order_acquire) != h;
            });
        tail_cache_ = tail_.load(std::memory_order_acquire);
      }
    }
    T out{slots_[h].move_and_destroy()};
    head_.store((h + 1U) & ring_mask(), std::memory_order_release);
    queue::detail::AsymmetricParkLot::notify_slot(prod_park_);
    return out;
  }

  void pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
    out = pop();
  }

  [[nodiscard]] std::size_t size() const noexcept {
    const std::size_t h = head_.load(std::memory_order_acquire);
    const std::size_t t = tail_.load(std::memory_order_acquire);
    return (t - h) & ring_mask();
  }

  [[nodiscard]] bool empty() const noexcept {
    return head_.load(std::memory_order_acquire) ==
           tail_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::size_t capacity() const noexcept { return ring_mask(); }

 private:
  [[nodiscard]] THETA_FORCE_INLINE constexpr std::size_t ring_mask()
      const noexcept {
    if constexpr (kStaticBufferSize != 0) {
      return kStaticBufferSize - 1U;
    } else {
      return mask_;
    }
  }

  const std::size_t mask_;
  Slot* const slots_;
  std::atomic<queue::detail::AsymmetricParkState*> prod_park_{nullptr};
  std::atomic<queue::detail::AsymmetricParkState*> cons_park_{nullptr};

  alignas(kHardwareDestructiveInterferenceSize) std::atomic<std::size_t> tail_{
      0};
  std::size_t head_cache_{0};

  alignas(kHardwareDestructiveInterferenceSize) std::atomic<std::size_t> head_{
      0};
  std::size_t tail_cache_{0};
};

}  // namespace theta
