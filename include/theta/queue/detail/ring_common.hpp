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
THETA_ALIGNED_NOINLINE void AsymmetricParkLot::wait_slot_until(
    std::atomic<AsymmetricParkState*>& slot, StopPred&& pred) noexcept {
  AsymmetricParkState* s = slot.load(std::memory_order_acquire);
  if (s == nullptr) {
    THETA_DISABLE_UNROLL
    for (int spin = 0; spin < 20; ++spin) {
      if (pred()) {
        return;
      }
      ArchBackend::stlf_micro_pause();
    }
    constexpr int kInitRelax = park_relax_budget(kMaxParkSpinShift);
    THETA_DISABLE_UNROLL
    for (int spin = 0; spin < kInitRelax; ++spin) {
      if (pred()) {
        return;
      }
      ArchBackend::cpu_relax();
    }
  } else if (s->waiters.load(std::memory_order_relaxed) == 0U) {
    THETA_DISABLE_UNROLL
    for (int spin = 0; spin < 20; ++spin) {
      if (pred()) {
        s->spin_shift.store(kMaxParkSpinShift, std::memory_order_relaxed);
        return;
      }
      ArchBackend::stlf_micro_pause();
    }
    const int relax_budget = park_relax_budget(static_cast<std::uint8_t>(
        s->spin_shift.load(std::memory_order_relaxed)));
    THETA_DISABLE_UNROLL
    for (int spin = 0; spin < relax_budget; ++spin) {
      if (pred()) {
        s->spin_shift.store(kMaxParkSpinShift, std::memory_order_relaxed);
        return;
      }
      ArchBackend::cpu_relax();
    }
  }
  if (pred()) {
    return;
  }
  std::uint8_t mode = 0U;
  if (s == nullptr) [[unlikely]] {
    s = get_or_activate_state_slow(slot, mode);
  } else {
    mode = static_cast<std::uint8_t>(s->mode.load(std::memory_order_acquire));
    if (mode < 2U) [[unlikely]] {
      s = get_or_activate_state_slow(slot, mode);
    }
  }
  s->waiters.fetch_add(1U, std::memory_order_relaxed);
  do {
    const std::uint32_t ep = s->epoch.load(std::memory_order_acquire);
    (void)s->needs_wake.exchange(1U, std::memory_order_seq_cst);
#if !THETA_ARCH_X86_64 || THETA_TSAN_ENABLED
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
    if (pred()) {
      break;
    }
    park_epoch(s, ep, mode);
  } while (!pred());
  s->waiters.fetch_sub(1U, std::memory_order_relaxed);
}

template <std::size_t kStripes>
template <typename StopPred>
THETA_ALIGNED_NOINLINE void StripedParkLot<kStripes>::wait_ticket(
    std::uint64_t ticket, StopPred&& pred,
    bool enable_work_conserving) noexcept {
  const std::size_t stripe_idx = static_cast<std::size_t>(ticket) & kStripeMask;
  StripedParkState* s = state_.load(std::memory_order_acquire);
  if (s == nullptr ||
      s->stripes[stripe_idx].waiters.load(std::memory_order_relaxed) == 0U) {
    THETA_DISABLE_UNROLL
    for (int spin = 0; spin < 20; ++spin) {
      if (pred()) {
        return;
      }
      ArchBackend::stlf_micro_pause();
    }
    THETA_DISABLE_UNROLL
    for (int spin = 0; spin < kStripeTicketRelaxIters; ++spin) {
      if (pred()) {
        return;
      }
      ArchBackend::cpu_relax();
    }
  }
  if (pred()) {
    return;
  }
  std::uint8_t mode = 0U;
  if (s == nullptr) [[unlikely]] {
    s = StripedParkLotBase::get_or_activate_state_slow(state_, mode);
  } else {
    mode = static_cast<std::uint8_t>(s->mode.load(std::memory_order_acquire));
    if (mode < 2U) [[unlikely]] {
      s = StripedParkLotBase::get_or_activate_state_slow(state_, mode);
    }
  }
  // Activate multi-consumer work-conserving mode only when >= 2 consumers
  // concurrently wait on an empty queue (`ticket_waiters >= 1`), without
  // touching `StripedParkState`'s header cache line on producer parks or
  // non-empty floods (`BM_MPMC_Blocking`, `BM_CrossCorePingPong`).
  bool tracked_tw = false;
  if (enable_work_conserving &&
      s->work_conserving.load(std::memory_order_relaxed) == 0U) [[unlikely]] {
    tracked_tw = true;
    if (s->ticket_waiters.fetch_add(1U, std::memory_order_acq_rel) >= 1U) {
      s->work_conserving.store(1U, std::memory_order_release);
    }
  }
  ParkStripe& st = s->stripes[stripe_idx];
  st.waiters.fetch_add(1U, std::memory_order_relaxed);
  do {
    const std::uint32_t ep = st.epoch.load(std::memory_order_acquire);
    (void)st.needs_wake.exchange(1U, std::memory_order_seq_cst);
#if !THETA_ARCH_X86_64 || THETA_TSAN_ENABLED
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
    if (pred()) {
      break;
    }
    StripedParkLotBase::park_stripe_epoch(st, ep, mode);
  } while (!pred());
  st.waiters.fetch_sub(1U, std::memory_order_relaxed);
  if (tracked_tw) [[unlikely]] {
    s->ticket_waiters.fetch_sub(1U, std::memory_order_release);
  }
}

template <std::size_t kStripes>
template <typename TryClaimFn, typename HasReadyFn>
THETA_ALIGNED_NOINLINE void StripedParkLot<kStripes>::wait_unclaimed(
    TryClaimFn&& try_claim, HasReadyFn&& has_ready) noexcept {
  std::uint8_t mode = 0U;
  StripedParkState* s = state_.load(std::memory_order_acquire);
  if (s == nullptr) [[unlikely]] {
    s = StripedParkLotBase::get_or_activate_state_slow(state_, mode);
  } else {
    mode = static_cast<std::uint8_t>(s->mode.load(std::memory_order_acquire));
    if (mode < 2U) [[unlikely]] {
      s = StripedParkLotBase::get_or_activate_state_slow(state_, mode);
    }
  }
  if (s->work_conserving.load(std::memory_order_relaxed) == 0U) [[unlikely]] {
    s->work_conserving.store(1U, std::memory_order_release);
  }

  for (;;) {
    if (try_claim()) {
      return;
    }

    // 1. Single Designated Lead Spinner: spins read-only on `has_ready()` and
    //    clears `unclaimed_spinner` (`seq_cst`) BEFORE advancing `head_` in
    //    `try_claim()`, closing any gap between `unclaimed_spinner == 1` and
    //    `pos != head_` on concurrent producer pushes.
    if (s->unclaimed_spinner.load(std::memory_order_relaxed) == 0U &&
        s->unclaimed_spinner.exchange(1U, std::memory_order_acquire) == 0U) {
      bool ready = false;
      THETA_DISABLE_UNROLL
      for (int spin = 0; spin < 20; ++spin) {
        if (has_ready()) {
          ready = true;
          break;
        }
        ArchBackend::stlf_micro_pause();
      }
      if (!ready) {
        const int relax_budget = park_relax_budget(static_cast<std::uint8_t>(
            s->spin_shift.load(std::memory_order_relaxed)));
        THETA_DISABLE_UNROLL
        for (int spin = 0; spin < relax_budget; ++spin) {
          if (has_ready()) {
            ready = true;
            break;
          }
          ArchBackend::cpu_relax();
        }
      }
      (void)s->unclaimed_spinner.exchange(0U, std::memory_order_seq_cst);
#if !THETA_ARCH_X86_64 || THETA_TSAN_ENABLED
      std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
      if (ready) {
        s->spin_shift.store(kMaxParkSpinShift, std::memory_order_relaxed);
      }
      if (try_claim()) {
        if (s->unclaimed_waiters.load(kParkFastLoadOrder) != 0U && has_ready())
            [[unlikely]] {
          StripedParkLotBase::wake_unclaimed_one_slow(s);
        }
        return;
      }
      if (ready) {
        // Another consumer stole the item right after `has_ready()`; loop to
        // re-elect spinner without parking prematurely.
        continue;
      }
    }

    // 2. Unclaimed Sleeper: zero pre-claimed tickets (`0` head-of-line `p99`
    //    blocking) + exact `unclaimed_waiters` count + single-sleeper
    //    `FUTEX_WAKE(1)` (`0` thundering herd, `0` lost wakeups).
    s->unclaimed_waiters.fetch_add(1U, std::memory_order_seq_cst);
#if !THETA_ARCH_X86_64 || THETA_TSAN_ENABLED
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
    bool claimed = false;
    do {
      const std::uint32_t ep =
          s->unclaimed_epoch.load(std::memory_order_acquire);
      std::atomic_thread_fence(std::memory_order_seq_cst);
      if (try_claim()) {
        claimed = true;
        break;
      }
      StripedParkLotBase::park_unclaimed_epoch(s, ep, mode);
      if (try_claim()) {
        claimed = true;
        break;
      }
    } while (s->unclaimed_spinner.load(std::memory_order_relaxed) != 0U &&
             s->work_conserving.load(std::memory_order_relaxed) != 0U);

    const std::uint32_t rem =
        s->unclaimed_waiters.fetch_sub(1U, std::memory_order_acq_rel) - 1U;
    if (claimed) {
      if (rem != 0U && has_ready()) [[unlikely]] {
        StripedParkLotBase::wake_unclaimed_one_slow(s);
      }
      return;
    }
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
