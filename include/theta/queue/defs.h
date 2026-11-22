// FILENAME: include/theta/queue/defs.h
// Copyright (c) 2023-2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(__linux__)
#include <linux/futex.h>
#include <linux/membarrier.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#if __has_include(<sys/rseq.h>)
#include <sys/rseq.h>
#define THETA_QUEUE_HAS_LINUX_RSEQ 1
#else
#define THETA_QUEUE_HAS_LINUX_RSEQ 0
#endif

#include <ctime>
#else
#define THETA_QUEUE_HAS_LINUX_RSEQ 0
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#if defined(__SANITIZE_THREAD__)
#define THETA_QUEUE_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define THETA_QUEUE_TSAN 1
#else
#define THETA_QUEUE_TSAN 0
#endif
#else
#define THETA_QUEUE_TSAN 0
#endif

#if defined(__SANITIZE_ADDRESS__)
#define THETA_QUEUE_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define THETA_QUEUE_ASAN 1
#else
#define THETA_QUEUE_ASAN 0
#endif
#else
#define THETA_QUEUE_ASAN 0
#endif

#if THETA_QUEUE_TSAN || THETA_QUEUE_ASAN
#define THETA_QUEUE_SANITIZED 1
#else
#define THETA_QUEUE_SANITIZED 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#define THETA_QUEUE_FORCE_INLINE __attribute__((always_inline)) inline
#define THETA_QUEUE_NOINLINE __attribute__((noinline)) inline
#define THETA_QUEUE_COLD_NOINLINE __attribute__((noinline, cold)) inline
#define THETA_QUEUE_ALIGNED_COLD \
  __attribute__((noinline, cold, aligned(64))) inline
#else
#define THETA_QUEUE_FORCE_INLINE inline
#define THETA_QUEUE_NOINLINE inline
#define THETA_QUEUE_COLD_NOINLINE inline
#define THETA_QUEUE_ALIGNED_COLD inline
#endif

namespace theta {

inline constexpr std::size_t kHardwareConstructiveInterferenceSize = 64;
inline constexpr std::size_t kHardwareDestructiveInterferenceSize = 128;

namespace queue_detail {

template <typename T>
inline constexpr bool kIsAlwaysLockFreeAtomic = false;

template <typename T>
  requires(std::is_trivially_copyable_v<T>)
inline constexpr bool kIsAlwaysLockFreeAtomic<std::atomic<T>> =
    std::atomic<T>::is_always_lock_free;

template <typename T>
inline constexpr bool kCanBeAtomic = kIsAlwaysLockFreeAtomic<std::atomic<T>>;

THETA_QUEUE_FORCE_INLINE void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  _mm_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

// ~15-18 cycle core-local Blocked Store-Forwarding (`ld_blocks.store_forward`)
// micro-pause on the thread's private stack red-zone cache line (`0` shared
// cache-bus traffic, avoids Skylake+'s stretched ~140-cycle `PAUSE` overshoot
// while yielding backend execution ports to SMT siblings).
THETA_QUEUE_FORCE_INLINE void stlf_micro_pause() noexcept {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && \
    !THETA_QUEUE_SANITIZED
  std::uint64_t out = 0;
  asm volatile(
      "movw %w[out], -6(%%rsp)\n\t"
      "movq -8(%%rsp), %[out]\n\t"
      : [out] "+r"(out)
      :
      : "memory");
#else
  cpu_relax();
#endif
}

THETA_QUEUE_FORCE_INLINE void spin_pause(std::uint32_t& iter) noexcept {
  if (++iter <= 24U) [[likely]] {
    stlf_micro_pause();
  } else {
    cpu_relax();
  }
}

// Explicit exclusive RFO write-intent prefetch (`0f 0d /1` `PREFETCHW` on
// x86-64). Avoids `__builtin_prefetch(..., 1, 3)` silently lowering to Shared
// `PREFETCHT0` (`I -> S -> M` double coherence hop) when `-mprfchw` is absent.
THETA_QUEUE_FORCE_INLINE void prefetch_for_write(const void* addr) noexcept {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  asm volatile("prefetchw (%0)" : : "r"(addr) : "memory");
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
  asm volatile("prfm pstl1keep, [%0]" : : "r"(addr) : "memory");
#elif defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(addr, 1, 3);
#else
  (void)addr;
#endif
}

// Full StoreLoad barrier that drains both x86 Store Buffers and AMD Zen 5's
// L2 Write-Coalescing Buffer (`WCB`) without dirtying any shared cache line.
THETA_QUEUE_FORCE_INLINE void dekker_store_load_fence() noexcept {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && \
    !THETA_QUEUE_SANITIZED
  asm volatile("lock orl $0, -8(%%rsp)" ::: "memory", "cc");
#else
  std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

[[nodiscard]] THETA_QUEUE_FORCE_INLINE std::int32_t
read_rseq_cpu_id() noexcept {
#if THETA_QUEUE_HAS_LINUX_RSEQ && defined(__x86_64__) && !THETA_QUEUE_SANITIZED
  std::int32_t cpu = -1;
  const std::ptrdiff_t off = __rseq_offset;
  asm volatile("movl %%fs:4(%[off]), %[cpu]"
               : [cpu] "=r"(cpu)
               : [off] "r"(off));
  return cpu;
#elif defined(__linux__)
  return ::sched_getcpu();
#else
  return 0;
#endif
}

// Suppresses `PREFETCHW` on single-CPU `1T` L1-hot workloads (saving 2 AGU ops)
// and latches `multi_cpu_ = 1` the instant a second CPU touches the queue.
class alignas(kHardwareDestructiveInterferenceSize) ContentionPrefetchGate {
 public:
  [[nodiscard]] THETA_QUEUE_FORCE_INLINE bool should_prefetch() noexcept {
    if (multi_cpu_.load(std::memory_order_relaxed) != 0U) [[likely]] {
      return true;
    }
    const std::int32_t cpu = read_rseq_cpu_id();
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

// Page-local cache-line index permutation: swaps low `kBits` [0..kBits-1] with
// [kBits..2*kBits-1] so consecutive tickets bounce across >= 128B (defeating
// both 64B false sharing and 128B L2 adjacent-line spatial prefetching) while
// staying inside a single 4 KiB TLB page.
template <std::size_t kBits = 3>
THETA_QUEUE_FORCE_INLINE constexpr std::size_t remap_slot_index(
    std::size_t raw_idx, std::size_t cap_log2) noexcept {
  if (cap_log2 < 2 * kBits) {
    return raw_idx;
  }
  constexpr std::size_t kMask = (std::size_t{1} << kBits) - 1U;
  const std::size_t mix = (raw_idx ^ (raw_idx >> kBits)) & kMask;
  return raw_idx ^ mix ^ (mix << kBits);
}

// Uninitialized aligned storage for `T` so empty queue slots never require `T`
// to be default-constructible and never run redundant constructors/destructors.
template <typename T>
struct RawSlotStorage {
  alignas(T) std::array<std::byte, sizeof(T)> bytes;

  template <typename... Args>
  THETA_QUEUE_FORCE_INLINE T* construct(Args&&... args) noexcept(
      std::is_nothrow_constructible_v<T, Args...>) {
    return std::construct_at(reinterpret_cast<T*>(bytes.data()),
                             std::forward<Args>(args)...);
  }

  [[nodiscard]] THETA_QUEUE_FORCE_INLINE T* ptr() noexcept {
    return std::launder(reinterpret_cast<T*>(bytes.data()));
  }

  [[nodiscard]] THETA_QUEUE_FORCE_INLINE const T* ptr() const noexcept {
    return std::launder(reinterpret_cast<const T*>(bytes.data()));
  }

  THETA_QUEUE_FORCE_INLINE T move_and_destroy() noexcept {
    T* p = ptr();
    T out{std::move(*p)};
    std::destroy_at(p);
    return out;
  }

  THETA_QUEUE_FORCE_INLINE void move_into_and_destroy(T& dst) noexcept(
      std::is_nothrow_move_assignable_v<T>) {
    T* p = ptr();
    dst = std::move(*p);
    std::destroy_at(p);
  }

  THETA_QUEUE_FORCE_INLINE void destroy() noexcept { std::destroy_at(ptr()); }
};

template <typename Elem>
Elem* aligned_alloc_uninit(std::size_t count) {
  constexpr std::size_t kAlign = std::max<std::size_t>(
      kHardwareDestructiveInterferenceSize, alignof(Elem));
  return static_cast<Elem*>(
      ::operator new[](count * sizeof(Elem), std::align_val_t{kAlign}));
}

template <typename Elem>
void aligned_free_uninit(Elem* ptr) noexcept {
  constexpr std::size_t kAlign = std::max<std::size_t>(
      kHardwareDestructiveInterferenceSize, alignof(Elem));
  ::operator delete[](ptr, std::align_val_t{kAlign});
}

inline void futex_wait_u32(std::atomic<std::uint32_t>* addr,
                           std::uint32_t expected) noexcept {
#if defined(__linux__) && !THETA_QUEUE_TSAN
  (void)::syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(addr),
                  FUTEX_WAIT_PRIVATE, expected, nullptr, nullptr, 0);
#else
  addr->wait(expected, std::memory_order_relaxed);
#endif
}

inline void futex_wake_all_u32(std::atomic<std::uint32_t>* addr) noexcept {
#if defined(__linux__) && !THETA_QUEUE_TSAN
  (void)::syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(addr),
                  FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
#else
  addr->notify_all();
#endif
}

inline bool asymmetric_membarrier_expedited() noexcept {
#if defined(__linux__) && !THETA_QUEUE_TSAN
  static const bool registered = []() noexcept {
    return ::syscall(SYS_membarrier, MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED,
                     0, 0) == 0;
  }();
  if (registered) [[likely]] {
    return ::syscall(SYS_membarrier, MEMBARRIER_CMD_PRIVATE_EXPEDITED, 0, 0) ==
           0;
  }
#endif
  return false;
}

inline bool asymmetric_membarrier_rseq_expedited() noexcept {
#if defined(__linux__) && THETA_QUEUE_HAS_LINUX_RSEQ && !THETA_QUEUE_SANITIZED
  static const bool registered = []() noexcept {
    return ::syscall(SYS_membarrier,
                     MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED_RSEQ, 0, 0) == 0;
  }();
  if (registered) [[likely]] {
    return ::syscall(SYS_membarrier, MEMBARRIER_CMD_PRIVATE_EXPEDITED_RSEQ, 0,
                     0) == 0;
  }
#endif
  return false;
}

// Tiny cache-line-friendly TTAS spinlock used only on cold 1-in-N slab/lane
// maintenance paths so `theta-queue` remains zero-external-dependency.
class ColdSpinLock {
 public:
  THETA_QUEUE_FORCE_INLINE void lock() noexcept {
    while (flag_.exchange(1U, std::memory_order_acquire) != 0U) [[unlikely]] {
      while (flag_.load(std::memory_order_relaxed) != 0U) {
        cpu_relax();
      }
    }
  }

  [[nodiscard]] THETA_QUEUE_FORCE_INLINE bool try_lock() noexcept {
    return flag_.load(std::memory_order_relaxed) == 0U &&
           flag_.exchange(1U, std::memory_order_acquire) == 0U;
  }

  THETA_QUEUE_FORCE_INLINE void unlock() noexcept {
    flag_.store(0U, std::memory_order_release);
  }

 private:
  std::atomic<std::uint8_t> flag_{0};
};

class ColdSpinGuard {
 public:
  explicit THETA_QUEUE_FORCE_INLINE ColdSpinGuard(ColdSpinLock& lk) noexcept
      : lk_(lk) {
    lk_.lock();
  }
  THETA_QUEUE_FORCE_INLINE ~ColdSpinGuard() noexcept { lk_.unlock(); }
  ColdSpinGuard(const ColdSpinGuard&) = delete;
  ColdSpinGuard& operator=(const ColdSpinGuard&) = delete;

 private:
  ColdSpinLock& lk_;
};

// 128B-isolated 2-Tier eventcount:
// - Tier 1 (`blocking_active_ == 0`, default until a thread actually parks):
//   2-instruction `cmpb $0 + je` (`0` `LOCK`s, `0` `StoreLoad` fences).
// - Tier 2 (`blocking_active_ != 0`, promoted once via a single
// `SYS_membarrier`
//   on the very first park): pairs private-stack `lock orl $0, -8(%rsp)` on
//   notifiers with `waiters_.fetch_add` on sleepers so subsequent parks need
//   ZERO process-wide `SYS_membarrier` IPIs (~15 ns park vs ~40 us IPI!).
class alignas(kHardwareDestructiveInterferenceSize) AsymmetricParkLot {
 public:
  THETA_QUEUE_FORCE_INLINE void notify_if_waiters() noexcept {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && \
    !THETA_QUEUE_SANITIZED
    std::uint32_t w = 0;
    const auto* bp = reinterpret_cast<const std::uint8_t*>(&blocking_active_);
    const auto* wp = reinterpret_cast<const std::uint32_t*>(&waiters_);
    asm volatile goto(
        "cmpb $0, (%[bp])\n\t"
        "je 1f\n\t"
        "lock orl $0, -8(%%rsp)\n\t"
        "movl (%[wp]), %[w]\n\t"
        "testl %[w], %[w]\n\t"
        "jne %l[slow]\n\t"
        "1:\n\t"
        : [w] "=&r"(w)
        : [bp] "r"(bp), [wp] "r"(wp)
        : "memory", "cc"
        : slow);
    return;
  slow:
    notify_slow();
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (blocking_active_.load(std::memory_order_relaxed) != 0U) [[unlikely]] {
      dekker_store_load_fence();
      if (waiters_.load(std::memory_order_seq_cst) != 0U) [[unlikely]] {
        notify_slow();
      }
    }
#endif
  }

  template <typename StopPred>
  THETA_QUEUE_ALIGNED_COLD void wait_until(StopPred&& pred) noexcept {
    if (waiters_.load(std::memory_order_relaxed) == 0U) {
      for (int spin = 0; spin < 20; ++spin) {
        if (pred()) {
          return;
        }
        stlf_micro_pause();
      }
      for (int spin = 0; spin < 16; ++spin) {
        if (pred()) {
          return;
        }
        cpu_relax();
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
#if defined(__linux__) && !THETA_QUEUE_TSAN
        if (mode == 2U) [[likely]] {
          (void)::syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(&epoch_),
                          FUTEX_WAIT_PRIVATE, ep, nullptr, nullptr, 0);
        } else {
          const struct timespec ts{.tv_sec = 0, .tv_nsec = 100'000L};
          (void)::syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(&epoch_),
                          FUTEX_WAIT_PRIVATE, ep, &ts, nullptr, 0);
        }
#else
        (void)mode;
        (void)ep;
        cpu_relax();
#endif
      }
      waiters_.fetch_sub(1U, std::memory_order_seq_cst);
    }
  }

 private:
  THETA_QUEUE_ALIGNED_COLD std::uint8_t activate_blocking_once() noexcept {
    const ColdSpinGuard guard(activate_lock_);
    std::uint8_t mode = blocking_active_.load(std::memory_order_relaxed);
    if (mode < 2U) {
      blocking_active_.store(1U, std::memory_order_seq_cst);
      const bool ok = asymmetric_membarrier_expedited();
      mode = ok ? 2U : 3U;
      blocking_active_.store(mode, std::memory_order_release);
    }
    return mode;
  }

  THETA_QUEUE_ALIGNED_COLD void notify_slow() noexcept {
    epoch_.fetch_add(1U, std::memory_order_seq_cst);
#if defined(__linux__) && !THETA_QUEUE_TSAN
    futex_wake_all_u32(&epoch_);
#endif
  }

  std::atomic<std::uint32_t> waiters_{0};
  std::atomic<std::uint32_t> epoch_{0};
  std::atomic<std::uint8_t> blocking_active_{0};
  ColdSpinLock activate_lock_;
};

// Ticket-striped 2-Tier eventcount: eliminates BOTH thundering-herd wakeups on
// oversubscribed MPMC/MPSC queues AND ongoing `SYS_membarrier` IPIs on park.
template <std::size_t kStripes = 16>
class alignas(kHardwareDestructiveInterferenceSize) StripedParkLot {
  static_assert((kStripes & (kStripes - 1U)) == 0U && kStripes >= 2U);
  static constexpr std::size_t kStripeMask = kStripes - 1U;

  struct ParkStripe {
    std::atomic<std::uint32_t> waiters{0};
    std::atomic<std::uint32_t> epoch{0};
  };

 public:
  THETA_QUEUE_FORCE_INLINE void notify_ticket(std::uint64_t ticket) noexcept {
    const std::size_t idx = static_cast<std::size_t>(ticket) & kStripeMask;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && \
    !THETA_QUEUE_SANITIZED
    std::uint32_t w = 0;
    const auto* bp = reinterpret_cast<const std::uint8_t*>(&blocking_active_);
    const auto* wp =
        reinterpret_cast<const std::uint32_t*>(&stripes_[idx].waiters);
    asm volatile goto(
        "cmpb $0, (%[bp])\n\t"
        "je 1f\n\t"
        "lock orl $0, -8(%%rsp)\n\t"
        "movl (%[wp]), %[w]\n\t"
        "testl %[w], %[w]\n\t"
        "jne %l[slow]\n\t"
        "1:\n\t"
        : [w] "=&r"(w)
        : [bp] "r"(bp), [wp] "r"(wp)
        : "memory", "cc"
        : slow);
    return;
  slow:
    notify_stripe_slow(idx);
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (blocking_active_.load(std::memory_order_relaxed) != 0U) [[unlikely]] {
      dekker_store_load_fence();
      if (stripes_[idx].waiters.load(std::memory_order_seq_cst) != 0U)
          [[unlikely]] {
        notify_stripe_slow(idx);
      }
    }
#endif
  }

  THETA_QUEUE_FORCE_INLINE void notify_range(std::uint64_t base_ticket,
                                             std::size_t count) noexcept {
    if (blocking_active_.load(std::memory_order_relaxed) == 0U) [[likely]] {
      return;
    }
    dekker_store_load_fence();
    const std::size_t n = std::min<std::size_t>(count, kStripes);
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t idx =
          static_cast<std::size_t>(base_ticket + i) & kStripeMask;
      if (stripes_[idx].waiters.load(std::memory_order_relaxed) != 0U)
          [[unlikely]] {
        notify_stripe_slow(idx);
      }
    }
  }

  template <typename StopPred>
  THETA_QUEUE_ALIGNED_COLD void wait_ticket(std::uint64_t ticket,
                                            StopPred&& pred) noexcept {
    ParkStripe& st = stripes_[static_cast<std::size_t>(ticket) & kStripeMask];
    if (st.waiters.load(std::memory_order_relaxed) == 0U) {
      for (int spin = 0; spin < 20; ++spin) {
        if (pred()) {
          return;
        }
        stlf_micro_pause();
      }
      for (int spin = 0; spin < 16; ++spin) {
        if (pred()) {
          return;
        }
        cpu_relax();
      }
    }
    while (!pred()) {
      const std::uint32_t ep = st.epoch.load(std::memory_order_relaxed);
      st.waiters.fetch_add(1U, std::memory_order_seq_cst);
      std::uint8_t mode = blocking_active_.load(std::memory_order_acquire);
      if (mode < 2U) [[unlikely]] {
        mode = activate_blocking_once();
      }
      if (!pred()) {
#if defined(__linux__) && !THETA_QUEUE_TSAN
        if (mode == 2U) [[likely]] {
          (void)::syscall(SYS_futex,
                          reinterpret_cast<std::uint32_t*>(&st.epoch),
                          FUTEX_WAIT_PRIVATE, ep, nullptr, nullptr, 0);
        } else {
          const struct timespec ts{.tv_sec = 0, .tv_nsec = 100'000L};
          (void)::syscall(SYS_futex,
                          reinterpret_cast<std::uint32_t*>(&st.epoch),
                          FUTEX_WAIT_PRIVATE, ep, &ts, nullptr, 0);
        }
#else
        (void)mode;
        (void)ep;
        cpu_relax();
#endif
      }
      st.waiters.fetch_sub(1U, std::memory_order_seq_cst);
    }
  }

 private:
  THETA_QUEUE_ALIGNED_COLD std::uint8_t activate_blocking_once() noexcept {
    const ColdSpinGuard guard(activate_lock_);
    std::uint8_t mode = blocking_active_.load(std::memory_order_relaxed);
    if (mode < 2U) {
      blocking_active_.store(1U, std::memory_order_seq_cst);
      const bool ok = asymmetric_membarrier_expedited();
      mode = ok ? 2U : 3U;
      blocking_active_.store(mode, std::memory_order_release);
    }
    return mode;
  }

  THETA_QUEUE_ALIGNED_COLD void notify_stripe_slow(std::size_t idx) noexcept {
    stripes_[idx].epoch.fetch_add(1U, std::memory_order_seq_cst);
#if defined(__linux__) && !THETA_QUEUE_TSAN
    futex_wake_all_u32(&stripes_[idx].epoch);
#endif
  }

  std::atomic<std::uint8_t> blocking_active_{0};
  ColdSpinLock activate_lock_;
  std::array<ParkStripe, kStripes> stripes_{};
};

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

  THETA_QUEUE_FORCE_INLINE void prefetch_write(std::size_t phys) noexcept {
    prefetch_for_write(&slots_[phys]);
  }

  [[nodiscard]] THETA_QUEUE_FORCE_INLINE std::atomic<std::uint64_t>& turn(
      std::size_t phys) noexcept {
    return slots_[phys].turn;
  }

  [[nodiscard]] THETA_QUEUE_FORCE_INLINE const std::atomic<std::uint64_t>& turn(
      std::size_t phys) const noexcept {
    return slots_[phys].turn;
  }

  [[nodiscard]] THETA_QUEUE_FORCE_INLINE RawSlotStorage<T>& storage(
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

  THETA_QUEUE_FORCE_INLINE void prefetch_write(std::size_t phys) noexcept {
    prefetch_for_write(&turns_[phys]);
    prefetch_for_write(&vals_[phys]);
  }

  [[nodiscard]] THETA_QUEUE_FORCE_INLINE std::atomic<std::uint64_t>& turn(
      std::size_t phys) noexcept {
    return turns_[phys];
  }

  [[nodiscard]] THETA_QUEUE_FORCE_INLINE const std::atomic<std::uint64_t>& turn(
      std::size_t phys) const noexcept {
    return turns_[phys];
  }

  [[nodiscard]] THETA_QUEUE_FORCE_INLINE RawSlotStorage<T>& storage(
      std::size_t phys) noexcept {
    return vals_[phys];
  }

 private:
  std::atomic<std::uint64_t>* const turns_;
  RawSlotStorage<T>* const vals_;
};

// Helper invoking user callback `fn(T&&)` during `drain()`. Supports both
// `void(T&&)` (always continues) and `bool(T&&)` (false = stop draining early).
template <typename Func, typename T>
[[nodiscard]] THETA_QUEUE_FORCE_INLINE bool invoke_drain_visitor(
    Func& fn, T&& item) noexcept(std::is_nothrow_invocable_v<Func&, T&&>) {
  if constexpr (std::is_invocable_r_v<bool, Func&, T&&>) {
    return static_cast<bool>(fn(std::forward<T>(item)));
  } else {
    fn(std::forward<T>(item));
    return true;
  }
}

}  // namespace queue_detail

// Legacy 8-byte atomic value concepts retained for backward compatibility.
template <typename T>
concept AtomType =
    std::default_initializable<T> && std::is_trivially_copyable_v<T> &&
    queue_detail::kCanBeAtomic<T> && (sizeof(T) <= 8);

template <typename T>
concept ZeroableAtomType = AtomType<T> && requires(const T& val) {
  { static_cast<bool>(val) } -> std::same_as<bool>;
} && (!static_cast<bool>(T{}));

// General queue element concept supporting pointers, scalars, small/large
// trivially copyable structs, over-aligned SIMD types, and move-only objects
// (e.g. `std::unique_ptr`, `std::string`, `std::vector`).
template <typename T>
concept QueueElement = (!std::is_reference_v<T>) && (!std::is_const_v<T>) &&
                       (!std::is_volatile_v<T>) && std::move_constructible<T> &&
                       std::is_nothrow_move_constructible_v<T> &&
                       std::is_nothrow_destructible_v<T>;

}  // namespace theta
