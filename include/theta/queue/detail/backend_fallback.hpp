// FILENAME: include/theta/queue/detail/backend_fallback.hpp
// Copyright (c) 2023-2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <theta/queue/detail/common.hpp>

#if !THETA_ARCH_X86_64 && !THETA_ARCH_AARCH64

namespace theta::queue::detail {

struct ArchBackend {
  static constexpr bool kHasRseqLanes = false;

  static THETA_FORCE_INLINE void cpu_relax() noexcept {
    ::theta::arch::cpu_relax();
  }

  static THETA_FORCE_INLINE void stlf_micro_pause() noexcept { cpu_relax(); }

  static THETA_FORCE_INLINE void polite_relax() noexcept { cpu_relax(); }

  static THETA_FORCE_INLINE void prefetch_for_write(const void* addr) noexcept {
    __builtin_prefetch(addr, 1, 3);
  }

  [[nodiscard]] static THETA_FORCE_INLINE std::int32_t current_cpu() noexcept {
    return current_cpu_os_fallback();
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

  static RseqPushResult try_reserve_rich_slot_rseq(
      RseqLaneCore& /*core*/, std::uint32_t& /*out_lane*/,
      std::uint64_t& /*out_ticket*/) noexcept {
    return RseqPushResult::Overflow;
  }

  template <typename StealFn>
  static bool try_claim_rich_pop_mc_rseq(RseqLaneCore& /*core*/,
                                         std::uint32_t& /*out_lane*/,
                                         std::uint64_t& /*out_ticket*/,
                                         StealFn&& /*steal_fn*/) noexcept {
    return false;
  }
};

}  // namespace theta::queue::detail

#endif  // !THETA_ARCH_X86_64 && !THETA_ARCH_AARCH64
