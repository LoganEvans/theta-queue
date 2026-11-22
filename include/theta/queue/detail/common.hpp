// FILENAME: include/theta/queue/detail/common.hpp
// Copyright (c) 2023-2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <theta/arch.hpp>
#include <type_traits>
#include <utility>

#if THETA_TSAN_ENABLED || THETA_ASAN_ENABLED
#define THETA_QUEUE_SANITIZED 1
#else
#define THETA_QUEUE_SANITIZED 0
#endif

#if defined(__clang__)
#define THETA_QUEUE_CAPABILITY(x) __attribute__((capability(x)))
#define THETA_QUEUE_SCOPED_LOCKABLE __attribute__((scoped_lockable))
#define THETA_QUEUE_ACQUIRE(...) \
  __attribute__((acquire_capability(__VA_ARGS__)))
#define THETA_QUEUE_RELEASE(...) \
  __attribute__((release_capability(__VA_ARGS__)))
#define THETA_QUEUE_TRY_ACQUIRE(...) \
  __attribute__((try_acquire_capability(__VA_ARGS__)))
#define THETA_QUEUE_GUARDED_BY(x) __attribute__((guarded_by(x)))
#else
#define THETA_QUEUE_CAPABILITY(x)
#define THETA_QUEUE_SCOPED_LOCKABLE
#define THETA_QUEUE_ACQUIRE(...)
#define THETA_QUEUE_RELEASE(...)
#define THETA_QUEUE_TRY_ACQUIRE(...)
#define THETA_QUEUE_GUARDED_BY(x)
#endif

namespace theta {

inline constexpr std::size_t kHardwareConstructiveInterferenceSize =
    arch::kCacheLineBytes;
inline constexpr std::size_t kHardwareDestructiveInterferenceSize =
    2U * arch::kCacheLineBytes;

namespace queue::detail {

struct QueueTestPeer;

template <typename T>
inline constexpr bool kIsAlwaysLockFreeAtomic = false;

template <typename T>
  requires(std::is_trivially_copyable_v<T>)
inline constexpr bool kIsAlwaysLockFreeAtomic<std::atomic<T>> =
    std::atomic<T>::is_always_lock_free;

template <typename T>
inline constexpr bool kCanBeAtomic = kIsAlwaysLockFreeAtomic<std::atomic<T>>;

// Cold OS & barrier helpers implemented in `src/queue.cpp`.
[[nodiscard]] std::int32_t current_cpu_os_fallback() noexcept;
bool asymmetric_membarrier_expedited() noexcept;
bool asymmetric_membarrier_rseq_expedited() noexcept;

// Cache-line-friendly TTAS spinlock used only on cold 1-in-N slab/lane
// maintenance paths so `theta-queue` remains zero-external-dependency.
class THETA_QUEUE_CAPABILITY("mutex") ColdSpinLock {
 public:
  THETA_FORCE_INLINE void lock() noexcept THETA_QUEUE_ACQUIRE() {
    while (flag_.exchange(1U, std::memory_order_acquire) != 0U) [[unlikely]] {
      while (flag_.load(std::memory_order_relaxed) != 0U) {
        ::theta::arch::cpu_relax();
      }
    }
  }

  [[nodiscard]] THETA_FORCE_INLINE bool try_lock() noexcept
      THETA_QUEUE_TRY_ACQUIRE(true) {
    return flag_.load(std::memory_order_relaxed) == 0U &&
           flag_.exchange(1U, std::memory_order_acquire) == 0U;
  }

  THETA_FORCE_INLINE void unlock() noexcept THETA_QUEUE_RELEASE() {
    flag_.store(0U, std::memory_order_release);
  }

 private:
  std::atomic<std::uint8_t> flag_{0};
};

class THETA_QUEUE_SCOPED_LOCKABLE ColdSpinGuard {
 public:
  explicit THETA_FORCE_INLINE ColdSpinGuard(ColdSpinLock& lk) noexcept
      THETA_QUEUE_ACQUIRE(lk)
      : lk_(lk) {
    lk_.lock();
  }

  THETA_FORCE_INLINE ~ColdSpinGuard() noexcept THETA_QUEUE_RELEASE() {
    lk_.unlock();
  }

  ColdSpinGuard(const ColdSpinGuard&) = delete;
  ColdSpinGuard& operator=(const ColdSpinGuard&) = delete;

 private:
  ColdSpinLock& lk_;
};

struct ParkStripe {
  std::atomic<std::uint32_t> waiters{0};
  std::atomic<std::uint32_t> epoch{0};
};

// 128B-isolated 2-Tier eventcount:
// - Tier 1 (`blocking_active_ == 0`, default until a thread actually parks):
//   frameless `testb + jne` (`0` `LOCK`s, `0` `StoreLoad` fences, `0` `asm`).
// - Tier 2 (`blocking_active_ != 0`, promoted once via a single
// `SYS_membarrier`
//   on the very first park): pairs `seq_cst` StoreLoad fence on notifiers with
//   `waiters_.fetch_add` on sleepers in `src/queue.cpp` so subsequent parks
//   need ZERO ongoing `SYS_membarrier` IPIs.
class alignas(kHardwareDestructiveInterferenceSize) AsymmetricParkLot {
  friend struct QueueTestPeer;

 public:
  THETA_FORCE_INLINE void notify_if_waiters() noexcept {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (blocking_active_.load(std::memory_order_relaxed) != 0U) [[unlikely]] {
      notify_active_slow();
    }
  }

  template <typename StopPred>
  THETA_ALIGNED_NOINLINE void wait_until(StopPred&& pred) noexcept;

 private:
  [[gnu::cold]] void notify_active_slow() noexcept;
  [[gnu::cold]] void notify_slow() noexcept;
  [[gnu::cold]] std::uint8_t activate_blocking_once() noexcept;
  [[gnu::cold]] void park_epoch(std::uint32_t ep, std::uint8_t mode) noexcept;

  std::atomic<std::uint32_t> waiters_{0};
  std::atomic<std::uint32_t> epoch_{0};
  std::atomic<std::uint8_t> blocking_active_{0};
  ColdSpinLock activate_lock_;
};

struct StripedParkLotBase {
  [[gnu::cold]] static std::uint8_t activate_blocking_once(
      std::atomic<std::uint8_t>& blocking_active,
      ColdSpinLock& activate_lock) noexcept;
  [[gnu::cold]] static void notify_stripe_slow(ParkStripe& stripe) noexcept;
  [[gnu::cold]] static void notify_ticket_active_slow(
      ParkStripe& stripe) noexcept;
  [[gnu::cold]] static void notify_range_active_slow(
      ParkStripe* stripes, std::size_t ring_mask, std::size_t stripe_mask,
      std::uint64_t base_ticket, std::size_t count) noexcept;
  [[gnu::cold]] static void park_stripe_epoch(ParkStripe& stripe,
                                              std::uint32_t ep,
                                              std::uint8_t mode) noexcept;
};

// Ticket-striped 2-Tier eventcount: eliminates BOTH thundering-herd wakeups on
// oversubscribed MPMC/MPSC queues AND ongoing `SYS_membarrier` IPIs on park.
template <std::size_t kStripes = 16>
class alignas(kHardwareDestructiveInterferenceSize) StripedParkLot {
  static_assert((kStripes & (kStripes - 1U)) == 0U && kStripes >= 2U);
  static constexpr std::size_t kStripeMask = kStripes - 1U;
  friend struct QueueTestPeer;

 public:
  THETA_FORCE_INLINE void notify_ticket(std::uint64_t ticket) noexcept {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (blocking_active_.load(std::memory_order_relaxed) != 0U) [[unlikely]] {
      StripedParkLotBase::notify_ticket_active_slow(
          stripes_[static_cast<std::size_t>(ticket) & kStripeMask]);
    }
  }

  THETA_FORCE_INLINE void notify_range(std::uint64_t base_ticket,
                                       std::size_t count,
                                       std::size_t ring_mask) noexcept {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (blocking_active_.load(std::memory_order_relaxed) != 0U) [[unlikely]] {
      StripedParkLotBase::notify_range_active_slow(
          stripes_.data(), ring_mask, kStripeMask, base_ticket, count);
    }
  }

  template <typename StopPred>
  THETA_ALIGNED_NOINLINE void wait_ticket(std::uint64_t ticket,
                                          StopPred&& pred) noexcept;

 private:
  std::atomic<std::uint8_t> blocking_active_{0};
  ColdSpinLock activate_lock_;
  std::array<ParkStripe, kStripes> stripes_{};
};

inline constexpr std::uint8_t kMaxRseqPreferred = 7U;
inline constexpr std::uint8_t kRseqScQuantum = 64U;
inline constexpr std::int32_t kMaxTrackedQueueCpus = 8192;

enum class RseqPushResult : std::uint8_t {
  Pushed = 0,
  Full = 1,
  Overflow = 2,
};

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
static_assert(offsetof(LaneMeta, tail) == 0);
static_assert(offsetof(LaneMeta, head_cache) == 8);
static_assert(offsetof(LaneMeta, data_prod) == 16);
static_assert(offsetof(LaneMeta, cap_prod) == 24);
static_assert(offsetof(LaneMeta, mask_prod) == 32);
static_assert(offsetof(LaneMeta, head) == 128);
static_assert(offsetof(LaneMeta, tail_cache) == 136);
static_assert(offsetof(LaneMeta, data_cons) == 144);
static_assert(offsetof(LaneMeta, mask_cons) == 152);
static_assert(offsetof(LaneMeta, cons_exclusive) == 160);

// Type-erased 2-Tier `rseq` + 7-Slot CAM state block; cold slot assignment,
// quiescent CAM recycling (`MEMBARRIER_RSEQ`), and work-stealing state
// transitions live out-of-line in `src/queue.cpp`.
struct RseqLaneCore {
  std::size_t lane_cap{0};
  std::uint64_t* lane_buf{nullptr};

  alignas(kHardwareDestructiveInterferenceSize)
      std::array<std::uint16_t, 8> prod_cam{};
  std::uint8_t prod_biased{0};

  alignas(kHardwareDestructiveInterferenceSize)
      std::array<std::uint16_t, 8> cons_cam{};
  std::uint8_t cons_biased{0};
  std::uint8_t sc_cursor{0};
  std::uint8_t sc_quantum{kRseqScQuantum};

  alignas(kHardwareDestructiveInterferenceSize) ColdSpinLock cam_lock;
  std::atomic<std::uint8_t> num_prod{0};
  std::atomic<std::uint8_t> num_cons{0};
  std::atomic<std::uint32_t> overflow_ops{0};
  std::uint64_t last_recycle_tail_sum THETA_QUEUE_GUARDED_BY(cam_lock){0};

  std::array<LaneMeta, kMaxRseqPreferred> meta{};

  void init(std::size_t cap, bool multi_consumer);
  void destroy() noexcept;
  [[nodiscard]] std::size_t approx_size() const noexcept;

  [[gnu::cold]] int refresh_push_state_slow() noexcept;
  [[gnu::cold]] bool try_assign_prod_slot_slow() noexcept;
  [[gnu::cold]] void maybe_recycle_prod_cam() noexcept;
  [[gnu::cold]] bool try_assign_cons_slot_slow() noexcept;
  [[gnu::cold]] int try_steal_lane_word_slow(std::uint64_t& out) noexcept;
};

// Page-local cache-line index permutation: swaps low `kBits` [0..kBits-1] with
// [kBits..2*kBits-1] so consecutive tickets bounce across >= 128B (defeating
// both 64B false sharing and 128B L2 adjacent-line spatial prefetching) while
// staying inside a single 4 KiB TLB page.
template <std::size_t kBits = 3>
THETA_FORCE_INLINE constexpr std::size_t remap_slot_index(
    std::size_t raw_idx, std::size_t cap_log2) noexcept {
  if (cap_log2 < 2U * kBits) {
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
  THETA_FORCE_INLINE T* construct(Args&&... args) noexcept(
      std::is_nothrow_constructible_v<T, Args...>) {
    return std::construct_at(reinterpret_cast<T*>(bytes.data()),
                             std::forward<Args>(args)...);
  }

  [[nodiscard]] THETA_FORCE_INLINE T* ptr() noexcept {
    return std::launder(reinterpret_cast<T*>(bytes.data()));
  }

  [[nodiscard]] THETA_FORCE_INLINE const T* ptr() const noexcept {
    return std::launder(reinterpret_cast<const T*>(bytes.data()));
  }

  THETA_FORCE_INLINE T move_and_destroy() noexcept {
    T* p = ptr();
    T out{std::move(*p)};
    std::destroy_at(p);
    return out;
  }

  THETA_FORCE_INLINE void move_into_and_destroy(T& dst) noexcept(
      std::is_nothrow_move_assignable_v<T>) {
    T* p = ptr();
    dst = std::move(*p);
    std::destroy_at(p);
  }

  THETA_FORCE_INLINE void destroy() noexcept { std::destroy_at(ptr()); }
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

// Helper invoking user callback `fn(T&&)` during `drain()`. Supports both
// `void(T&&)` (always continues) and `bool(T&&)` (false = stop draining early).
template <typename Func, typename T>
[[nodiscard]] THETA_FORCE_INLINE bool invoke_drain_visitor(
    Func& fn, T&& item) noexcept(std::is_nothrow_invocable_v<Func&, T&&>) {
  if constexpr (std::is_invocable_r_v<bool, Func&, T&&>) {
    return static_cast<bool>(fn(std::forward<T>(item)));
  } else {
    fn(std::forward<T>(item));
    return true;
  }
}

}  // namespace queue::detail
}  // namespace theta
