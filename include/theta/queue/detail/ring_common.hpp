// FILENAME: include/theta/queue/detail/ring_common.hpp
// Copyright (c) 2023-2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <theta/arch.hpp>
#include <theta/queue/detail/common.hpp>

#if THETA_ARCH_X86_64
#include <theta/queue/detail/backend_x86_64.hpp>
#elif THETA_ARCH_AARCH64
#include <theta/queue/detail/backend_aarch64.hpp>
#else
#include <theta/queue/detail/backend_fallback.hpp>
#endif

namespace theta::queue::detail {

THETA_FORCE_INLINE void spin_pause(std::uint32_t& iter) noexcept {
  if (++iter <= 24U) [[likely]] {
    ArchBackend::stlf_micro_pause();
  } else {
    ArchBackend::cpu_relax();
  }
}

// Suppresses `PREFETCHW` on single-CPU `1T` L1-hot workloads (saving 2 AGU ops)
// and latches `multi_cpu_ = 1` the instant a second CPU touches the queue.
class alignas(kHardwareDestructiveInterferenceSize) ContentionPrefetchGate {
 public:
  [[nodiscard]] THETA_FORCE_INLINE bool should_prefetch() noexcept {
    if (multi_cpu_.load(std::memory_order_relaxed) != 0U) [[likely]] {
      return true;
    }
    const std::int32_t cpu = ArchBackend::current_cpu();
    const std::int32_t prev = first_cpu_.load(std::memory_order_relaxed);
    if (prev == -2) [[unlikely]] {
      first_cpu_.store(cpu, std::memory_order_relaxed);
      return false;
    }
    if (prev != cpu) [[unlikely]] {
      multi_cpu_.store(1U, std::memory_order_relaxed);
      return true;
    }
    return false;
  }

 private:
  std::atomic<std::int32_t> first_cpu_{-2};
  std::atomic<std::uint8_t> multi_cpu_{0};
};

template <typename StopPred>
THETA_ALIGNED_NOINLINE void AsymmetricParkLot::wait_until(
    StopPred&& pred) noexcept {
  if (waiters_.load(std::memory_order_relaxed) == 0U) {
    THETA_DISABLE_UNROLL
    for (int spin = 0; spin < 20; ++spin) {
      if (pred()) {
        return;
      }
      ArchBackend::stlf_micro_pause();
    }
    THETA_DISABLE_UNROLL
    for (int spin = 0; spin < 16; ++spin) {
      if (pred()) {
        return;
      }
      ArchBackend::cpu_relax();
    }
  }
  while (!pred()) {
    const std::uint32_t ep = epoch_.load(std::memory_order_relaxed);
    waiters_.fetch_add(1U, std::memory_order_seq_cst);
    std::uint8_t mode = blocking_active_.load(std::memory_order_acquire);
    if (mode < 2U) [[unlikely]] {
      mode = activate_blocking_once();
    }
    if (!pred()) {
      park_epoch(ep, mode);
    }
    waiters_.fetch_sub(1U, std::memory_order_seq_cst);
  }
}

template <std::size_t kStripes>
template <typename StopPred>
THETA_ALIGNED_NOINLINE void StripedParkLot<kStripes>::wait_ticket(
    std::uint64_t ticket, StopPred&& pred) noexcept {
  ParkStripe& st = stripes_[static_cast<std::size_t>(ticket) & kStripeMask];
  if (st.waiters.load(std::memory_order_relaxed) == 0U) {
    THETA_DISABLE_UNROLL
    for (int spin = 0; spin < 20; ++spin) {
      if (pred()) {
        return;
      }
      ArchBackend::stlf_micro_pause();
    }
    THETA_DISABLE_UNROLL
    for (int spin = 0; spin < 16; ++spin) {
      if (pred()) {
        return;
      }
      ArchBackend::cpu_relax();
    }
  }
  while (!pred()) {
    const std::uint32_t ep = st.epoch.load(std::memory_order_relaxed);
    st.waiters.fetch_add(1U, std::memory_order_seq_cst);
    std::uint8_t mode = blocking_active_.load(std::memory_order_acquire);
    if (mode < 2U) [[unlikely]] {
      mode = StripedParkLotBase::activate_blocking_once(blocking_active_,
                                                        activate_lock_);
    }
    if (!pred()) {
      StripedParkLotBase::park_stripe_epoch(st, ep, mode);
    }
    st.waiters.fetch_sub(1U, std::memory_order_seq_cst);
  }
}

// Hybrid turn-sequenced slot storage:
// - Fuses `{atomic<uint64_t> turn, T}` into a single cache-local struct when
//   `sizeof(T) <= 56 && alignof(T) <= 32` (1 coherence hop brings both turn &
//   T).
// - Separates `turns[]` and `vals[]` when `T` is larger or >= 64B-aligned to
//   avoid alignment padding bloat while preserving full `alignof(T)` alignment.
template <typename T, bool kUseFused = (sizeof(T) <= 56 && alignof(T) <= 32)>
class TurnSlotArray;

template <typename T>
class TurnSlotArray<T, true> {
  static constexpr std::size_t kSlotAlign =
      (sizeof(T) <= 8 && alignof(T) <= 8) ? 16U : 64U;

  struct alignas(kSlotAlign) FusedSlot {
    std::atomic<std::uint64_t> turn{0};
    RawSlotStorage<T> val;
  };

 public:
  explicit TurnSlotArray(std::size_t cap)
      : slots_(aligned_alloc_uninit<FusedSlot>(cap)) {
    for (std::size_t i = 0; i < cap; ++i) {
      std::construct_at(&slots_[i].turn, std::uint64_t{0});
    }
  }

  ~TurnSlotArray() { aligned_free_uninit(slots_); }

  TurnSlotArray(const TurnSlotArray&) = delete;
  TurnSlotArray& operator=(const TurnSlotArray&) = delete;

  THETA_FORCE_INLINE void prefetch_write(std::size_t phys) noexcept {
    ArchBackend::prefetch_for_write(&slots_[phys]);
  }

  [[nodiscard]] THETA_FORCE_INLINE std::atomic<std::uint64_t>& turn(
      std::size_t phys) noexcept {
    return slots_[phys].turn;
  }

  [[nodiscard]] THETA_FORCE_INLINE const std::atomic<std::uint64_t>& turn(
      std::size_t phys) const noexcept {
    return slots_[phys].turn;
  }

  [[nodiscard]] THETA_FORCE_INLINE RawSlotStorage<T>& storage(
      std::size_t phys) noexcept {
    return slots_[phys].val;
  }

 private:
  FusedSlot* const slots_;
};

template <typename T>
class TurnSlotArray<T, false> {
 public:
  explicit TurnSlotArray(std::size_t cap)
      : turns_(aligned_alloc_uninit<std::atomic<std::uint64_t>>(cap)),
        vals_(aligned_alloc_uninit<RawSlotStorage<T>>(cap)) {
    for (std::size_t i = 0; i < cap; ++i) {
      std::construct_at(&turns_[i], std::uint64_t{0});
    }
  }

  ~TurnSlotArray() {
    aligned_free_uninit(turns_);
    aligned_free_uninit(vals_);
  }

  TurnSlotArray(const TurnSlotArray&) = delete;
  TurnSlotArray& operator=(const TurnSlotArray&) = delete;

  THETA_FORCE_INLINE void prefetch_write(std::size_t phys) noexcept {
    ArchBackend::prefetch_for_write(&turns_[phys]);
    ArchBackend::prefetch_for_write(&vals_[phys]);
  }

  [[nodiscard]] THETA_FORCE_INLINE std::atomic<std::uint64_t>& turn(
      std::size_t phys) noexcept {
    return turns_[phys];
  }

  [[nodiscard]] THETA_FORCE_INLINE const std::atomic<std::uint64_t>& turn(
      std::size_t phys) const noexcept {
    return turns_[phys];
  }

  [[nodiscard]] THETA_FORCE_INLINE RawSlotStorage<T>& storage(
      std::size_t phys) noexcept {
    return vals_[phys];
  }

 private:
  std::atomic<std::uint64_t>* const turns_;
  RawSlotStorage<T>* const vals_;
};

}  // namespace theta::queue::detail
