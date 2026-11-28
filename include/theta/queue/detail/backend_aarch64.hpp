// FILENAME: include/theta/queue/detail/backend_aarch64.hpp
// Copyright (c) 2023-2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <theta/queue/detail/common.hpp>

#if THETA_ARCH_AARCH64

#if THETA_HAS_LINUX_RSEQ
#include <sys/rseq.h>
#endif

namespace theta::queue::detail {

#if THETA_HAS_LINUX_RSEQ
static_assert(offsetof(struct rseq, cpu_id) == 4,
              "Linux struct rseq::cpu_id must reside at byte offset 4.");
static_assert(offsetof(struct rseq, rseq_cs) == 8,
              "Linux struct rseq::rseq_cs must reside at byte offset 8.");
#ifdef RSEQ_SIG
static_assert(
    RSEQ_SIG == 0xd428bc00U,
    "AArch64 RSEQ_SIG must match 0xd428bc00 in inline assembly abort pad.");
#endif
#endif

struct ArchBackend {
#if THETA_HAS_LINUX_RSEQ && !THETA_QUEUE_SANITIZED
  static constexpr bool kHasRseqLanes = true;
#else
  static constexpr bool kHasRseqLanes = false;
#endif

  static THETA_FORCE_INLINE void cpu_relax() noexcept {
    ::theta::arch::cpu_relax();
  }

  static THETA_FORCE_INLINE void stlf_micro_pause() noexcept { cpu_relax(); }

  // Paces Stage-2 user-space spin loops on AArch64 (`yield` is a 1-cycle NOP on
  // Neoverse-N1/V2, expiring 1024 iterations in ~340 ns). `isb` flushes the
  // pipeline frontend cleanly in ~20-30 cycles (~8-10 ns) without `wfe` EL2
  // hypervisor VMEXIT traps (`HCR_EL2.TWE`) or exclusive-monitor contention.
  static THETA_FORCE_INLINE void polite_relax() noexcept {
#if defined(__clang__)
    __builtin_arm_isb(15);
#else
    // WHY ASM: (b) ISA instruction (`isb`) unavailable via header-free GCC
    // builtin, pacing Stage-2 AArch64 spin loops (~8-10 ns vs 1-cycle `yield`).
    asm volatile("isb" ::: "memory");
#endif
  }

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

#if THETA_HAS_LINUX_RSEQ && !THETA_QUEUE_SANITIZED
  // Borrow-free 64-bit GPR SWAR lookup across `cam[1..7]` (`0` NEON-to-GPR
  // cross-domain transfer stall). Valid for all `target` and `cam[i]` in
  // `[0 .. 8192]` where bit 15 (`0x8000`) is guaranteed zero.
  [[nodiscard]] static THETA_FORCE_INLINE std::uint32_t find_cam_slot_fast(
      const std::uint16_t* cam, std::uint16_t target) noexcept {
    std::uint64_t w0 = 0;
    std::uint64_t w1 = 0;
    std::memcpy(&w0, cam, sizeof(w0));
    std::memcpy(&w1, cam + 4, sizeof(w1));
    const std::uint64_t b =
        static_cast<std::uint64_t>(target) * 0x0001000100010001ULL;

    const std::uint64_t v0 = w0 ^ b;
    const std::uint64_t z0 =
        (v0 | 0x8000800080008000ULL) - 0x0001000100010001ULL;
    const std::uint64_t m0 = (~z0) & 0x8000800080000000ULL;
    if (m0 != 0U) [[likely]] {
      return static_cast<std::uint32_t>(std::countr_zero(m0) >> 4U);
    }

    const std::uint64_t v1 = w1 ^ b;
    const std::uint64_t z1 =
        (v1 | 0x8000800080008000ULL) - 0x0001000100010001ULL;
    const std::uint64_t m1 = (~z1) & 0x8000800080008000ULL;
    if (m1 != 0U) [[likely]] {
      return 4U + static_cast<std::uint32_t>(std::countr_zero(m1) >> 4U);
    }
    return 0U;
  }

  static THETA_FORCE_INLINE RseqPushResult
  try_push_word_rseq(RseqLaneCore& core, AsymmetricParkLot& cons_park,
                     std::uint64_t val) noexcept {
    const auto* tp = static_cast<const char*>(__builtin_thread_pointer());
    const char* const rseq_base = tp + __rseq_offset;

  retry_push:
    if (__atomic_load_n(&core.prod_biased, __ATOMIC_RELAXED) != 0U) [[likely]] {
      std::uint64_t tmp = 0;
      std::uint64_t slot = 0;
      std::uint64_t cur_t = 0;
      std::uint64_t next_t = 0;
      // WHY ASM: (a) OS/ABI control-flow contract — Tier-1 biased producer
      // enqueue inside a restartable `[2f, 3f)` AArch64 `rseq` critical section
      // with `RSEQ_SIG` (`0xd428bc00`) preceding abort pad `4f` and `stlr`
      // release-commit to `meta[0].tail`.
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw?\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "adrp %[tmp], 1b\n\t"
          "add %[tmp], %[tmp], :lo12:1b\n\t"
          "str %[tmp], [%[rseq], #8]\n\t"
          "2:\n\t"
          "ldrb %w[tmp], [%[biased]]\n\t"
          "cbz %w[tmp], %l[tier2_7slot]\n\t"
          "ldr %w[tmp], [%[rseq], #4]\n\t"
          "ldrh %w[slot], [%[cam], #2]\n\t"
          "add %w[tmp], %w[tmp], #1\n\t"
          "cmp %w[tmp], %w[slot]\n\t"
          "b.ne %l[slow_push]\n\t"
          "ldr %[cur_t], [%[meta]]\n\t"
          "ldr %[tmp], [%[meta], #8]\n\t"
          "sub %[next_t], %[cur_t], %[tmp]\n\t"
          "ldr %[tmp], [%[meta], #24]\n\t"
          "cmp %[next_t], %[tmp]\n\t"
          "b.hs %l[refresh_push]\n\t"
          "add %[next_t], %[cur_t], #1\n\t"
          "ldr %[tmp], [%[meta], #32]\n\t"
          "and %[cur_t], %[cur_t], %[tmp]\n\t"
          "ldr %[tmp], [%[meta], #16]\n\t"
          "str %[val], [%[tmp], %[cur_t], lsl #3]\n\t"
          "stlr %[next_t], [%[meta]]\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax?\"\n\t"
          ".inst 0xd428bc00\n\t"
          "4:\n\t"
          "b 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [slot] "=&r"(slot), [cur_t] "=&r"(cur_t),
            [next_t] "=&r"(next_t)
          : [rseq] "r"(rseq_base), [biased] "r"(&core.prod_biased),
            [cam] "r"(core.prod_cam.data()), [meta] "r"(core.meta.data()),
            [val] "r"(val)
          : "memory", "cc"
          : tier2_7slot, refresh_push, slow_push);
      cons_park.notify_if_waiters();
      return RseqPushResult::Pushed;
    }

  tier2_7slot:
    for (int mig = 0; mig < 2; ++mig) {
      const std::int32_t approx_cpu =
          __atomic_load_n(reinterpret_cast<const std::int32_t*>(rseq_base + 4),
                          __ATOMIC_RELAXED);
      if (approx_cpu < 0 || approx_cpu >= kMaxTrackedQueueCpus) [[unlikely]] {
        goto slow_push;
      }
      const std::uint32_t slot_u32 = find_cam_slot_fast(
          core.prod_cam.data(), static_cast<std::uint16_t>(approx_cpu + 1));
      if (slot_u32 == 0U) {
        goto slow_push;
      }
      const auto slot_idx = static_cast<std::uint64_t>(slot_u32);
      LaneMeta* const lp = &core.meta[slot_u32 - 1U];
      std::uint64_t tmp = 0;
      std::uint64_t chk = 0;
      std::uint64_t cur_t = 0;
      std::uint64_t next_t = 0;
      // WHY ASM: (a) OS/ABI control-flow contract — Tier-2 7-slot CAM-verified
      // producer enqueue with `stlr` release-commit inside a restartable
      // `[2f, 3f)` AArch64 `rseq` critical section.
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw?\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "adrp %[tmp], 1b\n\t"
          "add %[tmp], %[tmp], :lo12:1b\n\t"
          "str %[tmp], [%[rseq], #8]\n\t"
          "2:\n\t"
          "ldr %w[tmp], [%[rseq], #4]\n\t"
          "add %w[tmp], %w[tmp], #1\n\t"
          "ldrh %w[chk], [%[cam], %[slot], lsl #1]\n\t"
          "cmp %w[tmp], %w[chk]\n\t"
          "b.ne %l[recheck_push]\n\t"
          "ldr %[cur_t], [%[lp]]\n\t"
          "ldr %[tmp], [%[lp], #8]\n\t"
          "sub %[next_t], %[cur_t], %[tmp]\n\t"
          "ldr %[tmp], [%[lp], #24]\n\t"
          "cmp %[next_t], %[tmp]\n\t"
          "b.hs %l[refresh_push]\n\t"
          "add %[next_t], %[cur_t], #1\n\t"
          "ldr %[tmp], [%[lp], #32]\n\t"
          "and %[cur_t], %[cur_t], %[tmp]\n\t"
          "ldr %[tmp], [%[lp], #16]\n\t"
          "str %[val], [%[tmp], %[cur_t], lsl #3]\n\t"
          "stlr %[next_t], [%[lp]]\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax?\"\n\t"
          ".inst 0xd428bc00\n\t"
          "4:\n\t"
          "b 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [chk] "=&r"(chk), [cur_t] "=&r"(cur_t),
            [next_t] "=&r"(next_t)
          : [rseq] "r"(rseq_base), [cam] "r"(core.prod_cam.data()),
            [slot] "r"(slot_idx), [lp] "r"(lp), [val] "r"(val)
          : "memory", "cc"
          : recheck_push, refresh_push);
      cons_park.notify_if_waiters();
      return RseqPushResult::Pushed;
    recheck_push:;
    }
    goto slow_push;

  refresh_push:
    {
      const int status = core.refresh_push_state_slow();
      if (status > 0) {
        goto retry_push;
      }
      if (status == 0) {
        return RseqPushResult::Full;
      }
    }
  slow_push:
    if (core.try_assign_prod_slot_slow()) {
      goto retry_push;
    }
    return RseqPushResult::Overflow;
  }

  template <typename StealFn>
  static THETA_FORCE_INLINE bool try_pop_word_mc_rseq(
      RseqLaneCore& core, AsymmetricParkLot& prod_park, std::uint64_t& out,
      StealFn&& steal_fn) noexcept {
    const auto* tp = static_cast<const char*>(__builtin_thread_pointer());
    const char* const rseq_base = tp + __rseq_offset;

  retry_pop:
    if (__atomic_load_n(&core.cons_biased, __ATOMIC_RELAXED) != 0U) [[likely]] {
      LaneMeta* const meta0 = core.meta.data();
      std::uint64_t* const head0 = &meta0->head;
      std::uint64_t tmp = 0;
      std::uint64_t slot = 0;
      std::uint64_t cur_h = 0;
      std::uint64_t next_h = 0;
      std::uint64_t val_out = 0;
      // WHY ASM: (a) OS/ABI control-flow contract — Tier-1 biased consumer
      // dequeue with inline `ldar` `tail_cache` refresh and `stlr` head commit
      // inside a restartable `[2f, 3f)` AArch64 `rseq` critical section.
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw?\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "adrp %[tmp], 1b\n\t"
          "add %[tmp], %[tmp], :lo12:1b\n\t"
          "str %[tmp], [%[rseq], #8]\n\t"
          "2:\n\t"
          "ldarb %w[tmp], [%[biased]]\n\t"
          "cbz %w[tmp], %l[tier2_pop]\n\t"
          "ldr %w[tmp], [%[rseq], #4]\n\t"
          "ldrh %w[slot], [%[cam], #2]\n\t"
          "add %w[tmp], %w[tmp], #1\n\t"
          "cmp %w[tmp], %w[slot]\n\t"
          "b.ne %l[slow_pop]\n\t"
          "ldr %[cur_h], [%[head]]\n\t"
          "ldr %[tmp], [%[meta], #136]\n\t"
          "cmp %[cur_h], %[tmp]\n\t"
          "b.ne 7f\n\t"
          "ldar %[tmp], [%[meta]]\n\t"
          "cmp %[cur_h], %[tmp]\n\t"
          "b.eq %l[steal_pop]\n\t"
          "str %[tmp], [%[meta], #136]\n\t"
          "7:\n\t"
          "add %[next_h], %[cur_h], #1\n\t"
          "ldr %[tmp], [%[meta], #152]\n\t"
          "and %[cur_h], %[cur_h], %[tmp]\n\t"
          "ldr %[tmp], [%[meta], #144]\n\t"
          "ldr %[val_out], [%[tmp], %[cur_h], lsl #3]\n\t"
          "stlr %[next_h], [%[head]]\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax?\"\n\t"
          ".inst 0xd428bc00\n\t"
          "4:\n\t"
          "b 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [slot] "=&r"(slot), [cur_h] "=&r"(cur_h),
            [next_h] "=&r"(next_h), [val_out] "=&r"(val_out)
          : [rseq] "r"(rseq_base), [biased] "r"(&core.cons_biased),
            [cam] "r"(core.cons_cam.data()), [meta] "r"(meta0),
            [head] "r"(head0)
          : "memory", "cc"
          : tier2_pop, steal_pop, slow_pop);
      out = val_out;
      prod_park.notify_if_waiters();
      return true;
    }

  tier2_pop:
    for (int mig = 0; mig < 2; ++mig) {
      const std::int32_t approx_cpu =
          __atomic_load_n(reinterpret_cast<const std::int32_t*>(rseq_base + 4),
                          __ATOMIC_RELAXED);
      if (approx_cpu < 0 || approx_cpu >= kMaxTrackedQueueCpus) [[unlikely]] {
        goto slow_pop;
      }
      const std::uint32_t slot_u32 = find_cam_slot_fast(
          core.cons_cam.data(), static_cast<std::uint16_t>(approx_cpu + 1));
      if (slot_u32 == 0U) {
        goto slow_pop;
      }
      const auto slot_idx = static_cast<std::uint64_t>(slot_u32);
      LaneMeta* const lp = &core.meta[slot_u32 - 1U];
      std::uint64_t* const head_ptr = &lp->head;
      std::uint64_t tmp = 0;
      std::uint64_t chk = 0;
      std::uint64_t cur_h = 0;
      std::uint64_t next_h = 0;
      std::uint64_t val_out = 0;
      // WHY ASM: (a) OS/ABI control-flow contract — Tier-2 7-slot CAM-verified
      // consumer dequeue + `cons_exclusive` check + `stlr` head commit inside a
      // restartable `[2f, 3f)` AArch64 `rseq` critical section.
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw?\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "adrp %[tmp], 1b\n\t"
          "add %[tmp], %[tmp], :lo12:1b\n\t"
          "str %[tmp], [%[rseq], #8]\n\t"
          "2:\n\t"
          "ldr %w[tmp], [%[rseq], #4]\n\t"
          "add %w[tmp], %w[tmp], #1\n\t"
          "ldrh %w[chk], [%[cam], %[slot], lsl #1]\n\t"
          "cmp %w[tmp], %w[chk]\n\t"
          "b.ne %l[recheck_pop]\n\t"
          "ldarb %w[tmp], [%[ex]]\n\t"
          "cbz %w[tmp], %l[steal_pop]\n\t"
          "ldr %[cur_h], [%[head]]\n\t"
          "ldr %[tmp], [%[lp], #136]\n\t"
          "cmp %[cur_h], %[tmp]\n\t"
          "b.ne 7f\n\t"
          "ldar %[tmp], [%[lp]]\n\t"
          "cmp %[cur_h], %[tmp]\n\t"
          "b.eq %l[steal_pop]\n\t"
          "str %[tmp], [%[lp], #136]\n\t"
          "7:\n\t"
          "add %[next_h], %[cur_h], #1\n\t"
          "ldr %[tmp], [%[lp], #152]\n\t"
          "and %[cur_h], %[cur_h], %[tmp]\n\t"
          "ldr %[tmp], [%[lp], #144]\n\t"
          "ldr %[val_out], [%[tmp], %[cur_h], lsl #3]\n\t"
          "stlr %[next_h], [%[head]]\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax?\"\n\t"
          ".inst 0xd428bc00\n\t"
          "4:\n\t"
          "b 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [chk] "=&r"(chk), [cur_h] "=&r"(cur_h),
            [next_h] "=&r"(next_h), [val_out] "=&r"(val_out)
          : [rseq] "r"(rseq_base), [cam] "r"(core.cons_cam.data()),
            [slot] "r"(slot_idx), [lp] "r"(lp), [head] "r"(head_ptr),
            [ex] "r"(&lp->cons_exclusive)
          : "memory", "cc"
          : recheck_pop, steal_pop);
      out = val_out;
      prod_park.notify_if_waiters();
      return true;
    recheck_pop:;
    }
    goto slow_pop;

  steal_pop:
    {
      const int st = steal_fn(out);
      if (st < 0) {
        goto retry_pop;
      }
      return st > 0;
    }
  slow_pop:
    if (core.try_assign_cons_slot_slow()) {
      goto retry_pop;
    }
    goto steal_pop;
  }

  static THETA_FORCE_INLINE RseqPushResult
  try_reserve_rich_slot_rseq(RseqLaneCore& core, std::uint32_t& out_lane,
                             std::uint64_t& out_ticket) noexcept {
    const auto* tp = static_cast<const char*>(__builtin_thread_pointer());
    const char* const rseq_base = tp + __rseq_offset;

  retry_rich_push:
    if (__atomic_load_n(&core.prod_biased, __ATOMIC_RELAXED) != 0U) [[likely]] {
      std::uint64_t tmp = 0;
      std::uint64_t slot = 0;
      std::uint64_t cur_t = 0;
      std::uint64_t next_t = 0;
      // WHY ASM: (a) OS/ABI control-flow contract — Tier-1 biased AArch64
      // `rseq` slot-index reservation with `ldar` turn acquire and `stlr` tail
      // commit (`[2f, 3f)`, payload constructed outside `rseq`).
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw?\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "adrp %[tmp], 1b\n\t"
          "add %[tmp], %[tmp], :lo12:1b\n\t"
          "str %[tmp], [%[rseq], #8]\n\t"
          "2:\n\t"
          "ldrb %w[tmp], [%[biased]]\n\t"
          "cbz %w[tmp], %l[tier2_rich_push]\n\t"
          "ldr %w[tmp], [%[rseq], #4]\n\t"
          "ldrh %w[slot], [%[cam], #2]\n\t"
          "add %w[tmp], %w[tmp], #1\n\t"
          "cmp %w[tmp], %w[slot]\n\t"
          "b.ne %l[slow_rich_push]\n\t"
          "ldr %[cur_t], [%[meta]]\n\t"
          "ldr %[tmp], [%[meta], #32]\n\t"
          "and %[next_t], %[cur_t], %[tmp]\n\t"
          "ldr %[tmp], [%[meta], #16]\n\t"
          "add %[tmp], %[tmp], %[next_t], lsl #6\n\t"
          "ldar %[tmp], [%[tmp]]\n\t"
          "cmp %[tmp], %[cur_t], lsl #1\n\t"
          "b.ne %l[full_rich_push]\n\t"
          "add %[next_t], %[cur_t], #1\n\t"
          "stlr %[next_t], [%[meta]]\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax?\"\n\t"
          ".inst 0xd428bc00\n\t"
          "4:\n\t"
          "b 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [slot] "=&r"(slot), [cur_t] "=&r"(cur_t),
            [next_t] "=&r"(next_t)
          : [rseq] "r"(rseq_base), [biased] "r"(&core.prod_biased),
            [cam] "r"(core.prod_cam.data()), [meta] "r"(core.meta.data())
          : "memory", "cc"
          : tier2_rich_push, full_rich_push, slow_rich_push);
      out_lane = 0U;
      out_ticket = cur_t;
      return RseqPushResult::Pushed;
    }

  tier2_rich_push:
    for (int mig = 0; mig < 2; ++mig) {
      const std::int32_t approx_cpu =
          __atomic_load_n(reinterpret_cast<const std::int32_t*>(rseq_base + 4),
                          __ATOMIC_RELAXED);
      if (approx_cpu < 0 || approx_cpu >= kMaxTrackedQueueCpus) [[unlikely]] {
        goto slow_rich_push;
      }
      const std::uint32_t slot_u32 = find_cam_slot_fast(
          core.prod_cam.data(), static_cast<std::uint16_t>(approx_cpu + 1));
      if (slot_u32 == 0U) {
        goto slow_rich_push;
      }
      const auto slot_idx = static_cast<std::uint64_t>(slot_u32);
      LaneMeta* const lp = &core.meta[slot_u32 - 1U];
      std::uint64_t tmp = 0;
      std::uint64_t chk = 0;
      std::uint64_t cur_t = 0;
      std::uint64_t next_t = 0;
      // WHY ASM: (a) OS/ABI control-flow contract — Tier-2 7-slot CAM AArch64
      // `rseq` slot-index reservation inside `[2f, 3f)`.
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw?\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "adrp %[tmp], 1b\n\t"
          "add %[tmp], %[tmp], :lo12:1b\n\t"
          "str %[tmp], [%[rseq], #8]\n\t"
          "2:\n\t"
          "ldr %w[tmp], [%[rseq], #4]\n\t"
          "add %w[tmp], %w[tmp], #1\n\t"
          "ldrh %w[chk], [%[cam], %[slot], lsl #1]\n\t"
          "cmp %w[tmp], %w[chk]\n\t"
          "b.ne %l[recheck_rich_push]\n\t"
          "ldr %[cur_t], [%[lp]]\n\t"
          "ldr %[tmp], [%[lp], #32]\n\t"
          "and %[next_t], %[cur_t], %[tmp]\n\t"
          "ldr %[tmp], [%[lp], #16]\n\t"
          "add %[tmp], %[tmp], %[next_t], lsl #6\n\t"
          "ldar %[tmp], [%[tmp]]\n\t"
          "cmp %[tmp], %[cur_t], lsl #1\n\t"
          "b.ne %l[full_rich_push]\n\t"
          "add %[next_t], %[cur_t], #1\n\t"
          "stlr %[next_t], [%[lp]]\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax?\"\n\t"
          ".inst 0xd428bc00\n\t"
          "4:\n\t"
          "b 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [chk] "=&r"(chk), [cur_t] "=&r"(cur_t),
            [next_t] "=&r"(next_t)
          : [rseq] "r"(rseq_base), [cam] "r"(core.prod_cam.data()),
            [slot] "r"(slot_idx), [lp] "r"(lp)
          : "memory", "cc"
          : recheck_rich_push, full_rich_push);
      out_lane = slot_u32 - 1U;
      out_ticket = cur_t;
      return RseqPushResult::Pushed;
    recheck_rich_push:;
    }
    goto slow_rich_push;

  full_rich_push:
    return RseqPushResult::Full;
  slow_rich_push:
    if (core.try_assign_prod_slot_slow()) {
      goto retry_rich_push;
    }
    return RseqPushResult::Overflow;
  }

  template <typename StealFn>
  static THETA_FORCE_INLINE bool try_claim_rich_pop_mc_rseq(
      RseqLaneCore& core, std::uint32_t& out_lane, std::uint64_t& out_ticket,
      StealFn&& steal_fn) noexcept {
    const auto* tp = static_cast<const char*>(__builtin_thread_pointer());
    const char* const rseq_base = tp + __rseq_offset;

  retry_rich_pop:
    if (__atomic_load_n(&core.cons_biased, __ATOMIC_RELAXED) != 0U) [[likely]] {
      LaneMeta* const meta0 = core.meta.data();
      std::uint64_t* const head0 = &meta0->head;
      std::uint64_t tmp = 0;
      std::uint64_t slot = 0;
      std::uint64_t cur_h = 0;
      std::uint64_t next_h = 0;
      // WHY ASM: (a) OS/ABI control-flow contract — Tier-1 biased consumer
      // AArch64 `rseq` slot-index claim with `ldar` turn acquire and `stlr`
      // head commit (`[2f, 3f)`).
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw?\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "adrp %[tmp], 1b\n\t"
          "add %[tmp], %[tmp], :lo12:1b\n\t"
          "str %[tmp], [%[rseq], #8]\n\t"
          "2:\n\t"
          "ldarb %w[tmp], [%[biased]]\n\t"
          "cbz %w[tmp], %l[tier2_rich_pop]\n\t"
          "ldr %w[tmp], [%[rseq], #4]\n\t"
          "ldrh %w[slot], [%[cam], #2]\n\t"
          "add %w[tmp], %w[tmp], #1\n\t"
          "cmp %w[tmp], %w[slot]\n\t"
          "b.ne %l[slow_rich_pop]\n\t"
          "ldr %[cur_h], [%[head]]\n\t"
          "ldr %[tmp], [%[meta], #152]\n\t"
          "and %[next_h], %[cur_h], %[tmp]\n\t"
          "ldr %[tmp], [%[meta], #144]\n\t"
          "add %[tmp], %[tmp], %[next_h], lsl #6\n\t"
          "ldar %[tmp], [%[tmp]]\n\t"
          "lsl %[next_h], %[cur_h], #1\n\t"
          "orr %[next_h], %[next_h], #1\n\t"
          "cmp %[tmp], %[next_h]\n\t"
          "b.ne %l[steal_rich_pop]\n\t"
          "add %[next_h], %[cur_h], #1\n\t"
          "stlr %[next_h], [%[head]]\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax?\"\n\t"
          ".inst 0xd428bc00\n\t"
          "4:\n\t"
          "b 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [slot] "=&r"(slot), [cur_h] "=&r"(cur_h),
            [next_h] "=&r"(next_h)
          : [rseq] "r"(rseq_base), [biased] "r"(&core.cons_biased),
            [cam] "r"(core.cons_cam.data()), [meta] "r"(meta0),
            [head] "r"(head0)
          : "memory", "cc"
          : tier2_rich_pop, steal_rich_pop, slow_rich_pop);
      out_lane = 0U;
      out_ticket = cur_h;
      return true;
    }

  tier2_rich_pop:
    for (int mig = 0; mig < 2; ++mig) {
      const std::int32_t approx_cpu =
          __atomic_load_n(reinterpret_cast<const std::int32_t*>(rseq_base + 4),
                          __ATOMIC_RELAXED);
      if (approx_cpu < 0 || approx_cpu >= kMaxTrackedQueueCpus) [[unlikely]] {
        goto slow_rich_pop;
      }
      const std::uint32_t slot_u32 = find_cam_slot_fast(
          core.cons_cam.data(), static_cast<std::uint16_t>(approx_cpu + 1));
      if (slot_u32 == 0U) {
        goto slow_rich_pop;
      }
      const auto slot_idx = static_cast<std::uint64_t>(slot_u32);
      LaneMeta* const lp = &core.meta[slot_u32 - 1U];
      std::uint64_t* const head_ptr = &lp->head;
      std::uint64_t tmp = 0;
      std::uint64_t chk = 0;
      std::uint64_t cur_h = 0;
      std::uint64_t next_h = 0;
      // WHY ASM: (a) OS/ABI control-flow contract — Tier-2 7-slot CAM consumer
      // AArch64 `rseq` slot-index claim inside `[2f, 3f)`.
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw?\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "adrp %[tmp], 1b\n\t"
          "add %[tmp], %[tmp], :lo12:1b\n\t"
          "str %[tmp], [%[rseq], #8]\n\t"
          "2:\n\t"
          "ldr %w[tmp], [%[rseq], #4]\n\t"
          "add %w[tmp], %w[tmp], #1\n\t"
          "ldrh %w[chk], [%[cam], %[slot], lsl #1]\n\t"
          "cmp %w[tmp], %w[chk]\n\t"
          "b.ne %l[recheck_rich_pop]\n\t"
          "ldarb %w[tmp], [%[ex]]\n\t"
          "cbz %w[tmp], %l[steal_rich_pop]\n\t"
          "ldr %[cur_h], [%[head]]\n\t"
          "ldr %[tmp], [%[lp], #152]\n\t"
          "and %[next_h], %[cur_h], %[tmp]\n\t"
          "ldr %[tmp], [%[lp], #144]\n\t"
          "add %[tmp], %[tmp], %[next_h], lsl #6\n\t"
          "ldar %[tmp], [%[tmp]]\n\t"
          "lsl %[next_h], %[cur_h], #1\n\t"
          "orr %[next_h], %[next_h], #1\n\t"
          "cmp %[tmp], %[next_h]\n\t"
          "b.ne %l[steal_rich_pop]\n\t"
          "add %[next_h], %[cur_h], #1\n\t"
          "stlr %[next_h], [%[head]]\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax?\"\n\t"
          ".inst 0xd428bc00\n\t"
          "4:\n\t"
          "b 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [chk] "=&r"(chk), [cur_h] "=&r"(cur_h),
            [next_h] "=&r"(next_h)
          : [rseq] "r"(rseq_base), [cam] "r"(core.cons_cam.data()),
            [slot] "r"(slot_idx), [lp] "r"(lp), [head] "r"(head_ptr),
            [ex] "r"(&lp->cons_exclusive)
          : "memory", "cc"
          : recheck_rich_pop, steal_rich_pop);
      out_lane = slot_u32 - 1U;
      out_ticket = cur_h;
      return true;
    recheck_rich_pop:;
    }
    goto slow_rich_pop;

  steal_rich_pop:
    {
      const int st = steal_fn(out_lane, out_ticket);
      if (st < 0) {
        goto retry_rich_pop;
      }
      return st > 0;
    }
  slow_rich_pop:
    if (core.try_assign_cons_slot_slow()) {
      goto retry_rich_pop;
    }
    goto steal_rich_pop;
  }
#else
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
#endif
};

}  // namespace theta::queue::detail

#endif  // THETA_ARCH_AARCH64
