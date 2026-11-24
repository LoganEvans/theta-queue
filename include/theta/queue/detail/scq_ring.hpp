// FILENAME: include/theta/queue/detail/scq_ring.hpp
// Copyright (c) 2023-2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <theta/arch.hpp>
#include <theta/queue/detail/ring_common.hpp>
#include <theta/queue/queue_opts.hpp>
#include <type_traits>
#include <utility>

namespace theta::queue::detail {

// ============================================================================
// Nikolaev SCQ Index Ring (SPAA 2019: 2N-entry Safe-Bit + Threshold Ring)
//
// Entry layout (64-bit unsigned):
//   [15:0]  index in [0 .. N - 1], or `kEmptyIdx = 0xFFFF` when empty
//   [16]    is_safe (1 = safe, 0 = unsafe)
//   [63:17] cycle   (47-bit monotonic lap counter)
//
// Each entry occupies a 64B cache line (`EntryLine`) with `remap_slot_index`
// permutation so consecutive `fetch_add` tickets land >= 128B apart.
// ============================================================================
class SCQIndexRing {
 public:
  static constexpr std::uint64_t kEmptyIdx = 0xFFFFULL;
  static constexpr std::uint64_t kIdxMask = 0xFFFFULL;
  static constexpr std::uint64_t kSafeBit = 1ULL << 16U;
  static constexpr unsigned kCycleShift = 17U;

  struct alignas(64) EntryLine {
    std::atomic<std::uint64_t> word{0};
  };
  static_assert(sizeof(EntryLine) == 64);

  [[nodiscard]] static constexpr std::uint64_t pack(
      std::uint64_t cycle, bool safe, std::uint64_t idx) noexcept {
    return (cycle << kCycleShift) | (safe ? kSafeBit : 0ULL) | (idx & kIdxMask);
  }

  [[nodiscard]] static constexpr std::uint64_t ent_cycle(
      std::uint64_t ent) noexcept {
    return ent >> kCycleShift;
  }

  [[nodiscard]] static constexpr bool ent_safe(std::uint64_t ent) noexcept {
    return (ent & kSafeBit) != 0ULL;
  }

  [[nodiscard]] static constexpr std::uint64_t ent_idx(
      std::uint64_t ent) noexcept {
    return ent & kIdxMask;
  }

  explicit SCQIndexRing(std::size_t cap, bool populate_initial_tokens)
      : cap_(cap),
        ring_cap_(2U * cap),
        ring_mask_((2U * cap) - 1U),
        ring_shift_(static_cast<std::size_t>(std::countr_zero(2U * cap))),
        thresh_max_(static_cast<std::int64_t>(3U * cap) - 1),
        entries_(aligned_alloc_uninit<EntryLine>(2U * cap)) {
    for (std::size_t raw = 0; raw < ring_cap_; ++raw) {
      const std::size_t phys = remap_slot_index(raw, ring_shift_);
      if (populate_initial_tokens && raw < cap_) {
        // Cycle 1 tokens at positions [2N .. 2N + N - 1]
        std::construct_at(&entries_[phys].word, pack(1ULL, true, raw));
      } else {
        std::construct_at(&entries_[phys].word, pack(0ULL, true, kEmptyIdx));
      }
    }
    head_.store(ring_cap_, std::memory_order_relaxed);
    tail_.store(populate_initial_tokens ? (ring_cap_ + cap_) : ring_cap_,
                std::memory_order_relaxed);
    threshold_.store(populate_initial_tokens ? thresh_max_ : -1,
                     std::memory_order_relaxed);
  }

  ~SCQIndexRing() { aligned_free_uninit(entries_); }

  SCQIndexRing(const SCQIndexRing&) = delete;
  SCQIndexRing& operator=(const SCQIndexRing&) = delete;

  // Enqueues a guaranteed-valid token `idx` in `[0 .. cap - 1]`.
  THETA_FORCE_INLINE void enqueue(std::uint64_t idx) noexcept {
    const std::size_t m = ring_mask_;
    const std::size_t sh = ring_shift_;
    for (;;) {
      const std::uint64_t t = tail_.fetch_add(1U, std::memory_order_relaxed);
      const std::size_t phys =
          remap_slot_index(static_cast<std::size_t>(t) & m, sh);
      const std::uint64_t cyc_t = t >> sh;
      std::uint64_t ent = entries_[phys].word.load(std::memory_order_acquire);
      for (;;) {
        const std::uint64_t cyc_e = ent_cycle(ent);
        if (cyc_e < cyc_t && ent_idx(ent) == kEmptyIdx &&
            (ent_safe(ent) || head_.load(std::memory_order_acquire) <= t))
            [[likely]] {
          const std::uint64_t desired = pack(cyc_t, true, idx);
          if (entries_[phys].word.compare_exchange_weak(
                  ent, desired, std::memory_order_acq_rel,
                  std::memory_order_acquire)) [[likely]] {
            if (threshold_.load(std::memory_order_relaxed) != thresh_max_) {
              threshold_.store(thresh_max_, std::memory_order_release);
            }
            return;
          }
        } else {
          break;
        }
      }
    }
  }

  // Attempts to dequeue a token; returns `kEmptyIdx` if empty.
  [[nodiscard]] THETA_FORCE_INLINE std::uint64_t dequeue() noexcept {
    if (threshold_.load(std::memory_order_acquire) < 0) {
      return kEmptyIdx;
    }
    const std::size_t m = ring_mask_;
    const std::size_t sh = ring_shift_;
    for (;;) {
      const std::uint64_t h = head_.fetch_add(1U, std::memory_order_relaxed);
      const std::size_t phys =
          remap_slot_index(static_cast<std::size_t>(h) & m, sh);
      const std::uint64_t cyc_h = h >> sh;
      std::uint64_t ent = entries_[phys].word.load(std::memory_order_acquire);
      for (;;) {
        const std::uint64_t cyc_e = ent_cycle(ent);
        if (cyc_e == cyc_h) [[likely]] {
          const std::uint64_t prev = entries_[phys].word.fetch_or(
              kEmptyIdx, std::memory_order_acq_rel);
          return ent_idx(prev);
        }
        if (cyc_e > cyc_h) {
          break;
        }
        const std::uint64_t next_ent =
            (ent_idx(ent) == kEmptyIdx) ? pack(cyc_h, ent_safe(ent), kEmptyIdx)
                                        : pack(cyc_e, false, ent_idx(ent));
        if (entries_[phys].word.compare_exchange_weak(
                ent, next_ent, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
          break;
        }
      }

      const std::uint64_t t = tail_.load(std::memory_order_acquire);
      if (t <= h + 1U) {
        catchup(t, h + 1U);
        threshold_.fetch_sub(1, std::memory_order_acq_rel);
        return kEmptyIdx;
      }
      if (threshold_.fetch_sub(1, std::memory_order_acq_rel) <= 0) {
        return kEmptyIdx;
      }
    }
  }

  [[nodiscard]] std::size_t approx_size() const noexcept {
    const std::uint64_t h = head_.load(std::memory_order_acquire);
    const std::uint64_t t = tail_.load(std::memory_order_acquire);
    if (t <= h) {
      return 0U;
    }
    const auto d = static_cast<std::size_t>(t - h);
    return d < cap_ ? d : cap_;
  }

 private:
  THETA_ALIGNED_NOINLINE void catchup(std::uint64_t t,
                                      std::uint64_t h_next) noexcept {
    while (!tail_.compare_exchange_weak(t, h_next, std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
      if (t >= h_next) {
        break;
      }
    }
  }

  const std::size_t cap_;
  const std::size_t ring_cap_;
  const std::size_t ring_mask_;
  const std::size_t ring_shift_;
  const std::int64_t thresh_max_;
  EntryLine* const entries_;

  alignas(
      kHardwareDestructiveInterferenceSize) std::atomic<std::uint64_t> head_{0};
  alignas(
      kHardwareDestructiveInterferenceSize) std::atomic<std::uint64_t> tail_{0};
  alignas(kHardwareDestructiveInterferenceSize)
      std::atomic<std::int64_t> threshold_{-1};
};

// ============================================================================
// Two-Ring Nikolaev SCQ Queue (`fq` + `aq` + `RawSlotStorage<T>[]`)
// Supports arbitrary move-only & large payloads under StrictFIFO ordering.
// ============================================================================
template <QueueElement T, std::size_t kStaticBufferSize = 0>
class SCQTwoRingQueue {
  static constexpr std::size_t resolve_cap(const QueueOpts& opts) noexcept {
    if constexpr (kStaticBufferSize != 0) {
      return kStaticBufferSize;
    } else {
      return std::bit_ceil(std::max<std::size_t>(opts.max_size(), 2U));
    }
  }

  struct alignas(64) AlignedSlot {
    RawSlotStorage<T> storage;
  };

 public:
  using value_type = T;

  explicit SCQTwoRingQueue(const QueueOpts& opts = QueueOpts{})
      : cap_(resolve_cap(opts)),
        fq_(cap_, /*populate_initial_tokens=*/true),
        aq_(cap_, /*populate_initial_tokens=*/false),
        data_(aligned_alloc_uninit<AlignedSlot>(cap_)) {}

  ~SCQTwoRingQueue() {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      while (try_pop().has_value()) {
      }
    }
    aligned_free_uninit(data_);
  }

  SCQTwoRingQueue(const SCQTwoRingQueue&) = delete;
  SCQTwoRingQueue& operator=(const SCQTwoRingQueue&) = delete;

  template <typename... Args>
    requires std::constructible_from<T, Args...>
  bool try_emplace(Args&&... args) noexcept(
      std::is_nothrow_constructible_v<T, Args...>) {
    const std::uint64_t idx = fq_.dequeue();
    if (idx == SCQIndexRing::kEmptyIdx) {
      return false;
    }
    data_[idx].storage.construct(std::forward<Args>(args)...);
    aq_.enqueue(idx);
    return true;
  }

  bool try_push(const T& val) noexcept(std::is_nothrow_copy_constructible_v<T>)
    requires std::copy_constructible<T>
  {
    return try_emplace(val);
  }

  bool try_push(T&& val) noexcept { return try_emplace(std::move(val)); }

  std::optional<T> try_pop() noexcept {
    const std::uint64_t idx = aq_.dequeue();
    if (idx == SCQIndexRing::kEmptyIdx) {
      return std::nullopt;
    }
    std::optional<T> out{data_[idx].storage.move_and_destroy()};
    fq_.enqueue(idx);
    return out;
  }

  bool try_pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
    const std::uint64_t idx = aq_.dequeue();
    if (idx == SCQIndexRing::kEmptyIdx) {
      return false;
    }
    data_[idx].storage.move_into_and_destroy(out);
    fq_.enqueue(idx);
    return true;
  }

  [[nodiscard]] std::size_t size() const noexcept { return aq_.approx_size(); }
  [[nodiscard]] bool empty() const noexcept { return size() == 0U; }
  [[nodiscard]] std::size_t capacity() const noexcept { return cap_; }

 private:
  const std::size_t cap_;
  SCQIndexRing fq_;
  SCQIndexRing aq_;
  AlignedSlot* const data_;
};

// ============================================================================
// Single-Ring Bounded SCQ for 4B-aligned Canonical Pointers (`int*`)
// Eliminates `fq_` completely (halving atomic RMWs from 4 to 2 per op) by
// packing `(uintptr_t >> 2)` in `[0..45]`, `is_safe` in `[46]`, and 17-bit
// `cycle` in `[47..63]` inside a single `2N`-entry SCQ ring.
// ============================================================================
template <typename PtrT, std::size_t kCapacity = 1024>
  requires(std::is_pointer_v<PtrT> && alignof(std::remove_pointer_t<PtrT>) >= 4)
class SCQSingleRingPtrQueue {
  static constexpr std::size_t kRingCap = 2U * kCapacity;
  static constexpr std::size_t kRingMask = kRingCap - 1U;
  static constexpr std::size_t kRingShift =
      static_cast<std::size_t>(std::countr_zero(kRingCap));
  static constexpr std::uint64_t kValMask = (1ULL << 46U) - 1U;
  static constexpr std::uint64_t kEmptyVal = kValMask;
  static constexpr std::uint64_t kSafeBit = 1ULL << 46U;
  static constexpr unsigned kCycleShift = 47U;
  static constexpr std::int64_t kThreshMax =
      static_cast<std::int64_t>(4U * kCapacity) - 1;

  struct alignas(64) EntryLine {
    std::atomic<std::uint64_t> word{0};
  };

  [[nodiscard]] static constexpr std::uint64_t pack(std::uint64_t cycle,
                                                    bool safe,
                                                    std::uint64_t v) noexcept {
    return (cycle << kCycleShift) | (safe ? kSafeBit : 0ULL) | (v & kValMask);
  }

  [[nodiscard]] static constexpr std::uint64_t ent_cycle(
      std::uint64_t ent) noexcept {
    return ent >> kCycleShift;
  }

  [[nodiscard]] static constexpr bool ent_safe(std::uint64_t ent) noexcept {
    return (ent & kSafeBit) != 0ULL;
  }

  [[nodiscard]] static constexpr std::uint64_t ent_val(
      std::uint64_t ent) noexcept {
    return ent & kValMask;
  }

  [[nodiscard]] static THETA_FORCE_INLINE std::uint64_t encode_ptr(
      PtrT p) noexcept {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(p) >>
                                      2U);
  }

  [[nodiscard]] static THETA_FORCE_INLINE PtrT
  decode_ptr(std::uint64_t v) noexcept {
    return reinterpret_cast<PtrT>(static_cast<std::uintptr_t>(v << 2U));
  }

 public:
  using value_type = PtrT;

  SCQSingleRingPtrQueue()
      : entries_(aligned_alloc_uninit<EntryLine>(kRingCap)) {
    for (std::size_t i = 0; i < kRingCap; ++i) {
      std::construct_at(&entries_[i].word, pack(0ULL, true, kEmptyVal));
    }
    head_.store(kRingCap, std::memory_order_relaxed);
    tail_.store(kRingCap, std::memory_order_relaxed);
    threshold_.store(-1, std::memory_order_relaxed);
  }

  ~SCQSingleRingPtrQueue() { aligned_free_uninit(entries_); }

  SCQSingleRingPtrQueue(const SCQSingleRingPtrQueue&) = delete;
  SCQSingleRingPtrQueue& operator=(const SCQSingleRingPtrQueue&) = delete;

  bool try_push(PtrT ptr) noexcept {
    const std::uint64_t val = encode_ptr(ptr);
    for (;;) {
      const std::uint64_t cur_h = head_.load(std::memory_order_acquire);
      const std::uint64_t cur_t = tail_.load(std::memory_order_relaxed);
      if (static_cast<std::int64_t>(cur_t - cur_h) >=
          static_cast<std::int64_t>(kCapacity)) {
        return false;
      }
      const std::uint64_t t = tail_.fetch_add(1U, std::memory_order_relaxed);
      if (static_cast<std::int64_t>(t -
                                    head_.load(std::memory_order_acquire)) >=
          static_cast<std::int64_t>(kCapacity)) {
        // Leave `t` empty so future `dequeue` skips it cleanly via threshold.
        if (threshold_.load(std::memory_order_relaxed) < kThreshMax) {
          threshold_.store(kThreshMax, std::memory_order_release);
        }
        return false;
      }
      const std::size_t phys =
          remap_slot_index(static_cast<std::size_t>(t) & kRingMask, kRingShift);
      const std::uint64_t cyc_t = t >> kRingShift;
      std::uint64_t ent = entries_[phys].word.load(std::memory_order_acquire);
      for (;;) {
        const std::uint64_t cyc_e = ent_cycle(ent);
        if (cyc_e < cyc_t && ent_val(ent) == kEmptyVal &&
            (ent_safe(ent) || head_.load(std::memory_order_acquire) <= t))
            [[likely]] {
          if (entries_[phys].word.compare_exchange_weak(
                  ent, pack(cyc_t, true, val), std::memory_order_acq_rel,
                  std::memory_order_acquire)) [[likely]] {
            if (threshold_.load(std::memory_order_relaxed) != kThreshMax) {
              threshold_.store(kThreshMax, std::memory_order_release);
            }
            return true;
          }
        } else {
          break;
        }
      }
    }
  }

  std::optional<PtrT> try_pop() noexcept {
    if (threshold_.load(std::memory_order_acquire) < 0) {
      return std::nullopt;
    }
    for (;;) {
      const std::uint64_t h = head_.fetch_add(1U, std::memory_order_relaxed);
      const std::size_t phys =
          remap_slot_index(static_cast<std::size_t>(h) & kRingMask, kRingShift);
      const std::uint64_t cyc_h = h >> kRingShift;
      std::uint64_t ent = entries_[phys].word.load(std::memory_order_acquire);
      for (;;) {
        const std::uint64_t cyc_e = ent_cycle(ent);
        if (cyc_e == cyc_h) [[likely]] {
          const std::uint64_t prev = entries_[phys].word.fetch_or(
              kEmptyVal, std::memory_order_acq_rel);
          const std::uint64_t v = ent_val(prev);
          if (v != kEmptyVal) [[likely]] {
            return decode_ptr(v);
          }
          break;
        }
        if (cyc_e > cyc_h) {
          break;
        }
        const std::uint64_t next_ent =
            (ent_val(ent) == kEmptyVal) ? pack(cyc_h, ent_safe(ent), kEmptyVal)
                                        : pack(cyc_e, false, ent_val(ent));
        if (entries_[phys].word.compare_exchange_weak(
                ent, next_ent, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
          break;
        }
      }

      const std::uint64_t t = tail_.load(std::memory_order_acquire);
      if (t <= h + 1U) {
        catchup(t, h + 1U);
        threshold_.fetch_sub(1, std::memory_order_acq_rel);
        return std::nullopt;
      }
      if (threshold_.fetch_sub(1, std::memory_order_acq_rel) <= 0) {
        return std::nullopt;
      }
    }
  }

 private:
  THETA_ALIGNED_NOINLINE void catchup(std::uint64_t t,
                                      std::uint64_t h_next) noexcept {
    while (!tail_.compare_exchange_weak(t, h_next, std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
      if (t >= h_next) {
        break;
      }
    }
  }

  EntryLine* const entries_;
  alignas(
      kHardwareDestructiveInterferenceSize) std::atomic<std::uint64_t> head_{0};
  alignas(
      kHardwareDestructiveInterferenceSize) std::atomic<std::uint64_t> tail_{0};
  alignas(kHardwareDestructiveInterferenceSize)
      std::atomic<std::int64_t> threshold_{-1};
};

}  // namespace theta::queue::detail
