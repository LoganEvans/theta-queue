// FILENAME: src/queue.cpp
// Copyright (c) 2023-2026 Logan Evans
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <theta/arch.hpp>
#include <theta/queue/detail/common.hpp>
#include <theta/queue/detail/ring_common.hpp>

#if THETA_OS_LINUX
#include <linux/futex.h>
#include <linux/membarrier.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace theta::queue::detail {

namespace {

[[gnu::tls_model(
    "initial-exec")]] constinit thread_local std::uint8_t g_steal_cursor = 0;

#if THETA_OS_LINUX && !THETA_TSAN_ENABLED
constinit std::atomic<std::uint8_t> g_membarrier_exp_state{0};
#endif

#if THETA_OS_LINUX && THETA_HAS_LINUX_RSEQ && !THETA_QUEUE_SANITIZED
constinit std::atomic<std::uint8_t> g_membarrier_rseq_state{0};
#endif

THETA_FORCE_INLINE void futex_wake_all_u32(
    std::atomic<std::uint32_t>* addr) noexcept {
#if THETA_OS_LINUX && !THETA_TSAN_ENABLED
  (void)::syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(addr),
                  FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
#else
  addr->notify_all();
#endif
}

THETA_FORCE_INLINE void futex_park_u32(std::atomic<std::uint32_t>* addr,
                                       std::uint32_t expected,
                                       std::uint8_t mode) noexcept {
#if THETA_OS_LINUX && !THETA_TSAN_ENABLED
  static constexpr struct timespec kBriefParkTimeout{
      .tv_sec = 0,
      .tv_nsec = 100'000L,
  };
  if (mode == 2U) [[likely]] {
    (void)::syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(addr),
                    FUTEX_WAIT_PRIVATE, expected, nullptr, nullptr, 0);
  } else {
    (void)::syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(addr),
                    FUTEX_WAIT_PRIVATE, expected, &kBriefParkTimeout, nullptr,
                    0);
  }
#else
  (void)addr;
  (void)expected;
  (void)mode;
  ::theta::arch::cpu_relax();
#endif
}

}  // namespace

[[gnu::noinline, gnu::aligned(64)]] std::int32_t
current_cpu_os_fallback() noexcept {
#if THETA_OS_LINUX
  const std::int32_t cpu = ::sched_getcpu();
  return cpu >= 0 ? cpu : 0;
#else
  return 0;
#endif
}

[[gnu::noinline, gnu::aligned(64)]] bool
asymmetric_membarrier_expedited() noexcept {
#if THETA_OS_LINUX && !THETA_TSAN_ENABLED
  std::uint8_t st = g_membarrier_exp_state.load(std::memory_order_relaxed);
  if (st == 0U) [[unlikely]] {
    const bool ok =
        ::syscall(SYS_membarrier, MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED, 0,
                  0) == 0;
    st = ok ? 1U : 2U;
    g_membarrier_exp_state.store(st, std::memory_order_relaxed);
  }
  if (st == 1U) [[likely]] {
    return ::syscall(SYS_membarrier, MEMBARRIER_CMD_PRIVATE_EXPEDITED, 0, 0) ==
           0;
  }
#endif
  return false;
}

[[gnu::noinline, gnu::aligned(64)]] bool
asymmetric_membarrier_rseq_expedited() noexcept {
#if THETA_OS_LINUX && THETA_HAS_LINUX_RSEQ && !THETA_QUEUE_SANITIZED
  std::uint8_t st = g_membarrier_rseq_state.load(std::memory_order_relaxed);
  if (st == 0U) [[unlikely]] {
    const bool ok =
        ::syscall(SYS_membarrier,
                  MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED_RSEQ, 0, 0) == 0;
    st = ok ? 1U : 2U;
    g_membarrier_rseq_state.store(st, std::memory_order_relaxed);
  }
  if (st == 1U) [[likely]] {
    return ::syscall(SYS_membarrier, MEMBARRIER_CMD_PRIVATE_EXPEDITED_RSEQ, 0,
                     0) == 0;
  }
#endif
  return false;
}

[[gnu::noinline, gnu::aligned(64)]] void AsymmetricParkLot::notify_active_slow(
    AsymmetricParkState* s) noexcept {
  std::atomic_thread_fence(std::memory_order_seq_cst);
  if (s->waiters.load(std::memory_order_seq_cst) != 0U) [[unlikely]] {
    s->epoch.fetch_add(1U, std::memory_order_seq_cst);
    futex_wake_all_u32(&s->epoch);
  }
}

[[gnu::noinline, gnu::aligned(64)]] AsymmetricParkState*
AsymmetricParkLot::get_or_activate_state_slow(
    std::atomic<AsymmetricParkState*>& slot, std::uint8_t& out_mode) noexcept {
  AsymmetricParkState* s = slot.load(std::memory_order_acquire);
  if (s == nullptr) {
    auto* fresh = aligned_alloc_uninit<AsymmetricParkState>(1U);
    std::construct_at(fresh);
    if (slot.compare_exchange_strong(s, fresh, std::memory_order_acq_rel,
                                     std::memory_order_acquire)) {
      s = fresh;
    } else {
      std::destroy_at(fresh);
      aligned_free_uninit(fresh);
    }
  }
  std::uint8_t m = s->mode.load(std::memory_order_acquire);
  if (m < 2U) {
    const ColdSpinGuard guard(s->activate_lock);
    m = s->mode.load(std::memory_order_relaxed);
    if (m < 2U) {
      std::atomic_thread_fence(std::memory_order_seq_cst);
      const bool ok = asymmetric_membarrier_expedited();
      m = ok ? 2U : 3U;
      s->mode.store(m, std::memory_order_release);
    }
  }
  out_mode = m;
  return s;
}

[[gnu::noinline, gnu::aligned(64)]] void AsymmetricParkLot::park_epoch(
    AsymmetricParkState* s, std::uint32_t ep, std::uint8_t mode) noexcept {
  futex_park_u32(&s->epoch, ep, mode);
}

[[gnu::noinline, gnu::aligned(64)]] void AsymmetricParkLot::destroy_state_slow(
    AsymmetricParkState* s) noexcept {
  std::destroy_at(s);
  aligned_free_uninit(s);
}

[[gnu::noinline, gnu::aligned(64)]] void destroy_spsc_core(
    void* slots, std::size_t align, AsymmetricParkState* prod,
    AsymmetricParkState* cons) noexcept {
  ::operator delete[](slots, std::align_val_t{align});
  if (prod != nullptr) [[unlikely]] {
    std::destroy_at(prod);
    aligned_free_uninit(prod);
  }
  if (cons != nullptr) [[unlikely]] {
    std::destroy_at(cons);
    aligned_free_uninit(cons);
  }
}

[[gnu::noinline, gnu::aligned(64)]] StripedParkState*
StripedParkLotBase::get_or_activate_state_slow(
    std::atomic<StripedParkState*>& slot, std::uint8_t& out_mode) noexcept {
  StripedParkState* s = slot.load(std::memory_order_acquire);
  if (s == nullptr) {
    auto* fresh = aligned_alloc_uninit<StripedParkState>(1U);
    std::construct_at(fresh);
    if (slot.compare_exchange_strong(s, fresh, std::memory_order_acq_rel,
                                     std::memory_order_acquire)) {
      s = fresh;
    } else {
      std::destroy_at(fresh);
      aligned_free_uninit(fresh);
    }
  }
  std::uint8_t m = s->mode.load(std::memory_order_acquire);
  if (m < 2U) {
    const ColdSpinGuard guard(s->activate_lock);
    m = s->mode.load(std::memory_order_relaxed);
    if (m < 2U) {
      std::atomic_thread_fence(std::memory_order_seq_cst);
      const bool ok = asymmetric_membarrier_expedited();
      m = ok ? 2U : 3U;
      s->mode.store(m, std::memory_order_release);
    }
  }
  out_mode = m;
  return s;
}

[[gnu::noinline, gnu::aligned(64)]] void
StripedParkLotBase::notify_ticket_active_slow(StripedParkState* state,
                                              std::size_t stripe_idx) noexcept {
  std::atomic_thread_fence(std::memory_order_seq_cst);
  ParkStripe& stripe = state->stripes[stripe_idx];
  if (stripe.waiters.load(std::memory_order_seq_cst) != 0U) [[unlikely]] {
    stripe.epoch.fetch_add(1U, std::memory_order_seq_cst);
    futex_wake_all_u32(&stripe.epoch);
  }
}

[[gnu::noinline, gnu::aligned(64)]] void
StripedParkLotBase::notify_range_active_slow(StripedParkState* state,
                                             std::size_t ring_mask,
                                             std::size_t stripe_mask,
                                             std::uint64_t base_ticket,
                                             std::size_t count) noexcept {
  std::atomic_thread_fence(std::memory_order_seq_cst);
  ParkStripe* stripes = state->stripes.data();
  const std::size_t max_unique = std::min(ring_mask + 1U, stripe_mask + 1U);
  const std::size_t n = std::min(count, max_unique);
  THETA_SCALAR_LOOP
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t idx =
        (static_cast<std::size_t>(base_ticket + i) & ring_mask) & stripe_mask;
    if (stripes[idx].waiters.load(std::memory_order_relaxed) != 0U)
        [[unlikely]] {
      stripes[idx].epoch.fetch_add(1U, std::memory_order_seq_cst);
      futex_wake_all_u32(&stripes[idx].epoch);
    }
  }
}

[[gnu::noinline, gnu::aligned(64)]] void StripedParkLotBase::park_stripe_epoch(
    ParkStripe& stripe, std::uint32_t ep, std::uint8_t mode) noexcept {
  futex_park_u32(&stripe.epoch, ep, mode);
}

[[gnu::noinline, gnu::aligned(64)]] void StripedParkLotBase::destroy_state_slow(
    StripedParkState* s) noexcept {
  std::destroy_at(s);
  aligned_free_uninit(s);
}

[[gnu::noinline, gnu::aligned(64)]] void RseqLaneCore::init(
    std::size_t cap, bool multi_consumer) {
  lane_cap = cap;
  lane_buf = aligned_alloc_uninit<std::uint64_t>(kMaxRseqPreferred * cap);
  const std::uint64_t m = cap - 1U;
  THETA_SCALAR_LOOP
  for (std::size_t i = 0; i < kMaxRseqPreferred; ++i) {
    meta[i].data_prod = lane_buf + (i * cap);
    meta[i].cap_prod = cap;
    meta[i].mask_prod = m;
    meta[i].data_cons = lane_buf + (i * cap);
    meta[i].mask_cons = m;
    meta[i].cons_exclusive = multi_consumer ? 1U : 0U;
  }
  (void)asymmetric_membarrier_rseq_expedited();
}

[[gnu::noinline, gnu::aligned(64)]] void RseqLaneCore::init_rich_turns(
    std::size_t cap, bool multi_consumer,
    std::uint64_t* base_turn_ptr) noexcept {
  lane_cap = cap;
  lane_buf = nullptr;
  const std::uint64_t m = cap - 1U;
  constexpr std::size_t kU64PerTurnLine = 64U / sizeof(std::uint64_t);
  THETA_SCALAR_LOOP
  for (std::size_t i = 0; i < kMaxRseqPreferred; ++i) {
    std::uint64_t* const lt = base_turn_ptr + (i * cap * kU64PerTurnLine);
    THETA_SCALAR_LOOP
    for (std::size_t j = 0; j < cap; ++j) {
      lt[j * kU64PerTurnLine] = static_cast<std::uint64_t>(j) * 2ULL;
    }
    meta[i].data_prod = lt;
    meta[i].cap_prod = cap;
    meta[i].mask_prod = m;
    meta[i].data_cons = lt;
    meta[i].mask_cons = m;
    meta[i].cons_exclusive = multi_consumer ? 1U : 0U;
  }
  (void)asymmetric_membarrier_rseq_expedited();
}

[[gnu::noinline, gnu::aligned(64)]] void RseqLaneCore::destroy() noexcept {
  aligned_free_uninit(lane_buf);
  lane_buf = nullptr;
  lane_cap = 0U;
  prod_biased = 0U;
  cons_biased = 0U;
}

[[gnu::noinline, gnu::aligned(64)]] std::size_t RseqLaneCore::approx_size()
    const noexcept {
  std::size_t total = 0U;
  THETA_SCALAR_LOOP
  for (std::size_t i = 0; i < kMaxRseqPreferred; ++i) {
    const std::uint64_t h = __atomic_load_n(&meta[i].head, __ATOMIC_ACQUIRE);
    const std::uint64_t t = __atomic_load_n(&meta[i].tail, __ATOMIC_ACQUIRE);
    if (t > h) {
      const auto d = static_cast<std::size_t>(t - h);
      total += (d < lane_cap ? d : lane_cap);
    }
  }
  return total;
}

// Returns: `1` if `head_cache` refreshed with free space, `0` if lane is
// genuinely full, `-1` if current CPU does not own a preferred slot.
[[gnu::noinline, gnu::aligned(64)]] int
RseqLaneCore::refresh_push_state_slow() noexcept {
  const std::int32_t cpu = ArchBackend::current_cpu();
  if (cpu >= 0 && cpu < kMaxTrackedQueueCpus) {
    const auto target = static_cast<std::uint16_t>(cpu + 1);
    THETA_SCALAR_LOOP
    for (std::uint8_t s = 1; s <= kMaxRseqPreferred; ++s) {
      if (__atomic_load_n(&prod_cam[s], __ATOMIC_RELAXED) == target) {
        LaneMeta& m = meta[s - 1U];
        const std::uint64_t h = __atomic_load_n(&m.head, __ATOMIC_ACQUIRE);
        __atomic_store_n(&m.head_cache, h, __ATOMIC_RELAXED);
        if ((__atomic_load_n(&m.tail, __ATOMIC_RELAXED) - h) >= lane_cap) {
          return 0;
        }
        return 1;
      }
    }
  }
  return -1;
}

[[gnu::noinline, gnu::aligned(64)]] bool
RseqLaneCore::try_assign_prod_slot_slow() noexcept {
  const std::int32_t cpu = ArchBackend::current_cpu();
  if (cpu >= 0 && cpu < kMaxTrackedQueueCpus &&
      num_prod.load(std::memory_order_relaxed) < kMaxRseqPreferred) {
    bool assigned = false;
    const ColdSpinGuard guard(cam_lock);
    const auto target = static_cast<std::uint16_t>(cpu + 1);
    THETA_SCALAR_LOOP
    for (std::uint8_t s = 1; s <= kMaxRseqPreferred; ++s) {
      if (prod_cam[s] == target) {
        assigned = true;
        break;
      }
    }
    if (!assigned) {
      const std::uint8_t n = num_prod.load(std::memory_order_relaxed);
      if (n < kMaxRseqPreferred) {
        const std::uint8_t s = n + 1U;
        __atomic_store_n(&prod_cam[s], target, __ATOMIC_RELEASE);
        if (s == 1U) {
          __atomic_store_n(&prod_biased, std::uint8_t{1}, __ATOMIC_RELEASE);
        } else if (__atomic_load_n(&prod_biased, __ATOMIC_RELAXED) != 0U) {
          __atomic_store_n(&prod_biased, std::uint8_t{0}, __ATOMIC_RELEASE);
          (void)asymmetric_membarrier_rseq_expedited();
        }
        num_prod.store(s, std::memory_order_release);
        assigned = true;
      }
    }
    return assigned;
  }
  return false;
}

[[gnu::noinline, gnu::aligned(64)]] void
RseqLaneCore::maybe_recycle_prod_cam() noexcept {
  if (!cam_lock.try_lock()) {
    return;
  }
  std::uint64_t tail_sum = 0;
  THETA_SCALAR_LOOP
  for (std::size_t i = 0; i < kMaxRseqPreferred; ++i) {
    tail_sum += __atomic_load_n(&meta[i].tail, __ATOMIC_RELAXED);
  }
  if (tail_sum == last_recycle_tail_sum &&
      num_prod.load(std::memory_order_relaxed) == kMaxRseqPreferred) {
    auto* cam_words = reinterpret_cast<std::uint64_t*>(prod_cam.data());
    __atomic_store_n(&cam_words[0], std::uint64_t{0}, __ATOMIC_RELEASE);
    __atomic_store_n(&cam_words[1], std::uint64_t{0}, __ATOMIC_RELEASE);
    __atomic_store_n(&prod_biased, std::uint8_t{0}, __ATOMIC_RELEASE);
    (void)asymmetric_membarrier_rseq_expedited();
    num_prod.store(0U, std::memory_order_release);
  }
  last_recycle_tail_sum = tail_sum;
  cam_lock.unlock();
}

[[gnu::noinline, gnu::aligned(64)]] bool
RseqLaneCore::try_assign_cons_slot_slow() noexcept {
  const std::int32_t cpu = ArchBackend::current_cpu();
  if (cpu >= 0 && cpu < kMaxTrackedQueueCpus &&
      num_cons.load(std::memory_order_relaxed) < kMaxRseqPreferred) {
    bool assigned = false;
    const ColdSpinGuard guard(cam_lock);
    const auto target = static_cast<std::uint16_t>(cpu + 1);
    THETA_SCALAR_LOOP
    for (std::uint8_t s = 1; s <= kMaxRseqPreferred; ++s) {
      if (cons_cam[s] == target) {
        assigned = true;
        break;
      }
    }
    if (!assigned) {
      const std::uint8_t n = num_cons.load(std::memory_order_relaxed);
      if (n < kMaxRseqPreferred) {
        const std::uint8_t s = n + 1U;
        __atomic_store_n(&cons_cam[s], target, __ATOMIC_RELEASE);
        if (s == 1U &&
            __atomic_load_n(&meta[0].cons_exclusive, __ATOMIC_RELAXED) != 0U) {
          __atomic_store_n(&cons_biased, std::uint8_t{1}, __ATOMIC_RELEASE);
        } else if (__atomic_load_n(&cons_biased, __ATOMIC_RELAXED) != 0U) {
          __atomic_store_n(&cons_biased, std::uint8_t{0}, __ATOMIC_RELEASE);
          (void)asymmetric_membarrier_rseq_expedited();
        }
        num_cons.store(s, std::memory_order_release);
        assigned = true;
      }
    }
    return assigned;
  }
  return false;
}

// Returns: `1` if item stolen into `out`, `-1` if caller's own exclusive `rseq`
// lane became non-empty (caller retries `rseq` without self-demoting
// `cons_exclusive`), `0` if all preferred lanes are empty.
[[gnu::noinline, gnu::aligned(64)]] int RseqLaneCore::try_steal_lane_word_slow(
    std::uint64_t& out) noexcept {
  const std::int32_t cpu = ArchBackend::current_cpu();
  const std::uint16_t self_target = (cpu >= 0 && cpu < kMaxTrackedQueueCpus)
                                        ? static_cast<std::uint16_t>(cpu + 1)
                                        : std::uint16_t{0};

  THETA_SCALAR_LOOP
  for (std::uint8_t step = 0; step < kMaxRseqPreferred; ++step) {
    const auto idx =
        static_cast<std::uint8_t>((g_steal_cursor + step) % kMaxRseqPreferred);
    LaneMeta& m = meta[idx];
    std::uint64_t h = __atomic_load_n(&m.head, __ATOMIC_ACQUIRE);
    std::uint64_t t = __atomic_load_n(&m.tail, __ATOMIC_ACQUIRE);
    if (h >= t) {
      continue;
    }

    if (__atomic_load_n(&m.cons_exclusive, __ATOMIC_ACQUIRE) != 0U) {
      // Never self-demote our own exclusive home lane!
      if (self_target != 0U &&
          __atomic_load_n(&cons_cam[idx + 1U], __ATOMIC_RELAXED) ==
              self_target) {
        return -1;
      }
      const ColdSpinGuard guard(cam_lock);
      if (__atomic_load_n(&m.cons_exclusive, __ATOMIC_RELAXED) != 0U) {
        __atomic_store_n(&cons_biased, std::uint8_t{0}, __ATOMIC_RELEASE);
        __atomic_store_n(&m.cons_exclusive, std::uint8_t{0}, __ATOMIC_RELEASE);
        (void)asymmetric_membarrier_rseq_expedited();
      }
    }

    h = __atomic_load_n(&m.head, __ATOMIC_ACQUIRE);
    for (;;) {
      t = __atomic_load_n(&m.tail, __ATOMIC_ACQUIRE);
      if (h >= t) {
        break;
      }
      const std::uint64_t val =
          __atomic_load_n(&m.data_cons[h & m.mask_cons], __ATOMIC_RELAXED);
      const std::uint64_t next_h = h + 1U;
      if (__atomic_compare_exchange_n(&m.head, &h, next_h, true,
                                      __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        out = val;
        g_steal_cursor = idx;
        return 1;
      }
    }
  }
  return 0;
}

[[gnu::noinline, gnu::aligned(64)]] int RseqLaneCore::try_steal_rich_slot_slow(
    std::uint32_t& out_lane, std::uint64_t& out_ticket) noexcept {
  const std::int32_t cpu = ArchBackend::current_cpu();
  const std::uint16_t self_target = (cpu >= 0 && cpu < kMaxTrackedQueueCpus)
                                        ? static_cast<std::uint16_t>(cpu + 1)
                                        : std::uint16_t{0};
  constexpr std::size_t kU64PerTurnLine = 64U / sizeof(std::uint64_t);

  THETA_SCALAR_LOOP
  for (std::uint8_t step = 0; step < kMaxRseqPreferred; ++step) {
    const auto idx =
        static_cast<std::uint8_t>((g_steal_cursor + step) % kMaxRseqPreferred);
    LaneMeta& m = meta[idx];
    std::uint64_t h = __atomic_load_n(&m.head, __ATOMIC_ACQUIRE);
    const std::size_t off0 =
        static_cast<std::size_t>(h & m.mask_cons) * kU64PerTurnLine;
    if (__atomic_load_n(&m.data_cons[off0], __ATOMIC_ACQUIRE) !=
        (h * 2ULL) + 1ULL) {
      continue;
    }

    if (__atomic_load_n(&m.cons_exclusive, __ATOMIC_ACQUIRE) != 0U) {
      if (self_target != 0U &&
          __atomic_load_n(&cons_cam[idx + 1U], __ATOMIC_RELAXED) ==
              self_target) {
        return -1;
      }
      const ColdSpinGuard guard(cam_lock);
      if (__atomic_load_n(&m.cons_exclusive, __ATOMIC_RELAXED) != 0U) {
        __atomic_store_n(&cons_biased, std::uint8_t{0}, __ATOMIC_RELEASE);
        __atomic_store_n(&m.cons_exclusive, std::uint8_t{0}, __ATOMIC_RELEASE);
        (void)asymmetric_membarrier_rseq_expedited();
      }
    }

    h = __atomic_load_n(&m.head, __ATOMIC_ACQUIRE);
    for (;;) {
      const std::size_t off =
          static_cast<std::size_t>(h & m.mask_cons) * kU64PerTurnLine;
      if (__atomic_load_n(&m.data_cons[off], __ATOMIC_ACQUIRE) !=
          (h * 2ULL) + 1ULL) {
        break;
      }
      const std::uint64_t next_h = h + 1U;
      if (__atomic_compare_exchange_n(&m.head, &h, next_h, true,
                                      __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        out_lane = idx;
        out_ticket = h;
        g_steal_cursor = idx;
        return 1;
      }
    }
  }
  return 0;
}

}  // namespace theta::queue::detail
