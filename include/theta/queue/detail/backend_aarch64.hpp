// FILENAME: include/theta/queue/detail/backend_aarch64.hpp
// Copyright (c) 2023-2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <theta/queue/detail/common.hpp>

#if THETA_ARCH_AARCH64

#if THETA_HAS_LINUX_RSEQ
#include <sys/rseq.h>
#endif

namespace theta::queue::detail {

#if THETA_HAS_LINUX_RSEQ
static_assert(offsetof(struct rseq, cpu_id) == 4,
              "Linux struct rseq::cpu_id must reside at byte offset 4.");
#endif

struct ArchBackend {
  static constexpr bool kHasRseqLanes = false;

  static THETA_FORCE_INLINE void cpu_relax() noexcept {
    ::theta::arch::cpu_relax();
  }

  static THETA_FORCE_INLINE void stlf_micro_pause() noexcept { cpu_relax(); }

  static THETA_FORCE_INLINE void prefetch_for_write(const void* addr) noexcept {
    __builtin_prefetch(addr, 1, 3);
  }

  [[nodiscard]] static THETA_FORCE_INLINE std::int32_t current_cpu() noexcept {
#if THETA_HAS_LINUX_RSEQ && !THETA_QUEUE_SANITIZED
    const auto* tp = static_cast<const char*>(__builtin_thread_pointer());
    const auto* cpu_ptr =
        reinterpret_cast<const std::int32_t*>(tp + __rseq_offset + 4);
    return __atomic_load_n(cpu_ptr, __ATOMIC_RELAXED);
#else
    return current_cpu_os_fallback();
#endif
  }

  static RseqPushResult try_push_word_rseq(RseqLaneCore& /*core*/,
                                           AsymmetricParkLot& /*cons_park*/,
                                           std::uint64_t /*val*/) noexcept {
    return RseqPushResult::Overflow;
  }

  template <typename StealFn>
  static bool try_pop_word_mc_rseq(RseqLaneCore& /*core*/,
                                   AsymmetricParkLot& /*prod_park*/,
                                   std::uint64_t& /*out*/,
                                   StealFn&& /*steal_fn*/) noexcept {
    return false;
  }
};

}  // namespace theta::queue::detail

#endif  // THETA_ARCH_AARCH64
