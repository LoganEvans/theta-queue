// FILENAME: include/theta/queue/detail/backend_x86_64.hpp
// Copyright (c) 2023-2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <theta/queue/detail/common.hpp>

#if THETA_ARCH_X86_64

#include <immintrin.h>
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
    RSEQ_SIG == 0x53053053,
    "x86-64 RSEQ_SIG must match 0x53053053 in inline assembly abort pad.");
#endif
#endif

struct ArchBackend {
  static constexpr bool kHasRseqLanes =
      (THETA_HAS_LINUX_RSEQ != 0) && (THETA_QUEUE_SANITIZED == 0);

  static THETA_FORCE_INLINE void cpu_relax() noexcept {
    ::theta::arch::cpu_relax();
  }

  static THETA_FORCE_INLINE void stlf_micro_pause() noexcept {
    if constexpr (::theta::arch::kHasRedZoneStlfPause) {
      std::uint64_t dep = 0;
      // WHY ASM: (b) Missing ISA/C++ construct — deliberate misaligned partial
      // Store-Buffer forwarding reject (`-6(%rsp)` -> `-8(%rsp)`) inside the
      // x86-64 System V 128B red-zone (~15-18 cycles vs ~140-cycle Skylake /
      // Golden Cove `pause`). Pure C++ would DCE or legalize the overlap.
      asm volatile(
          "movw %w[dep], -6(%%rsp)\n\t"
          "movq -8(%%rsp), %[dep]\n\t"
          : [dep] "+r"(dep)
          :
          : "memory");
    } else {
      cpu_relax();
    }
  }

  static THETA_FORCE_INLINE void prefetch_for_write(const void* addr) noexcept {
    // WHY ASM: (b) Broken builtin lowering — without `-mprfchw` on consumer
    // TUs, Clang lowers `__builtin_prefetch(addr, 1, 3)` to Shared `prefetcht0`
    // (`I -> S -> M` double coherence upgrade, `-39%` MPMC regression) instead
    // of exclusive RFO `prefetchw` (`0f 0d /1`, architectural NOP on legacy).
    asm volatile("prefetchw (%0)" : : "r"(addr) : "memory");
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
  static THETA_FORCE_INLINE RseqPushResult
  try_push_word_rseq(RseqLaneCore& core, AsymmetricParkLot& cons_park,
                     std::uint64_t val) noexcept {
    const std::ptrdiff_t rseq_off = __rseq_offset;

  retry_push:
    // Tier 1 (1-CPU Biased): `movzwl` bijection + offset-0 `meta[0]`
    // (`0` SSE2, `0` shift, `0` LOCK instructions).
    {
      std::uint64_t tmp = 0;
      std::uint64_t slot = 0;
      std::uint64_t cur_t = 0;
      std::uint64_t next_t = 0;
      // WHY ASM: (a) OS/ABI control-flow contract — Linux `rseq` requires an
      // exact `[2f, 3f)` instruction window registered in `__rseq_cs` with
      // `RSEQ_SIG` (`0x53053053`) immediately preceding abort pad `4f`.
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw?\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "leaq 1b(%%rip), %[tmp]\n\t"
          "movq %[tmp], %%fs:8(%[rseq_off])\n\t"
          "2:\n\t"
          "cmpb $0, (%[biased])\n\t"
          "je %l[tier2_7slot]\n\t"
          "movl %%fs:4(%[rseq_off]), %k[tmp]\n\t"
          "movzwl 2(%[cam]), %k[slot]\n\t"
          "incl %k[tmp]\n\t"
          "cmpl %k[slot], %k[tmp]\n\t"
          "jne %l[slow_push]\n\t"
          "movq (%[meta]), %[cur_t]\n\t"
          "movq %[cur_t], %[next_t]\n\t"
          "subq 8(%[meta]), %[next_t]\n\t"
          "cmpq 24(%[meta]), %[next_t]\n\t"
          "jae %l[refresh_push]\n\t"
          "leaq 1(%[cur_t]), %[next_t]\n\t"
          "andq 32(%[meta]), %[cur_t]\n\t"
          "movq 16(%[meta]), %[tmp]\n\t"
          "movq %[val], (%[tmp], %[cur_t], 8)\n\t"
          "movq %[next_t], (%[meta])\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax?\"\n\t"
          ".long 0x53053053\n\t"
          "4:\n\t"
          "jmp 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [slot] "=&r"(slot), [cur_t] "=&r"(cur_t),
            [next_t] "=&r"(next_t)
          : [rseq_off] "r"(rseq_off), [biased] "r"(&core.prod_biased),
            [cam] "r"(core.prod_cam.data()), [meta] "r"(core.meta.data()),
            [val] "r"(val)
          : "memory", "cc"
          : tier2_7slot, refresh_push, slow_push);
      cons_park.notify_if_waiters();
      return RseqPushResult::Pushed;
    }

  tier2_7slot:
    {
      __m128i v_cpu{};  // NOLINT(misc-const-correctness)
      std::uint64_t tmp = 0;
      std::uint64_t slot = 0;
      std::uint64_t cur_t = 0;
      std::uint64_t next_t = 0;
      std::uintptr_t lp = 0;
      // WHY ASM: (a) OS/ABI control-flow contract — Tier-2 7-slot SSE2
      // `pcmpeqw` inverse-CAM lookup + single-commit ring enqueue inside a
      // restartable `[2f, 3f)` Linux `rseq` critical section.
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw?\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "leaq 1b(%%rip), %[tmp]\n\t"
          "movq %[tmp], %%fs:8(%[rseq_off])\n\t"
          "2:\n\t"
          "movl %%fs:4(%[rseq_off]), %k[tmp]\n\t"
          "cmpl $8192, %k[tmp]\n\t"
          "jae %l[slow_push]\n\t"
          "incl %k[tmp]\n\t"
          "movd %k[tmp], %[v]\n\t"
          "pshuflw $0, %[v], %[v]\n\t"
          "punpcklqdq %[v], %[v]\n\t"
          "pcmpeqw (%[cam]), %[v]\n\t"
          "pmovmskb %[v], %k[slot]\n\t"
          "bsfl %k[slot], %k[slot]\n\t"
          "je %l[slow_push]\n\t"
          "shrl $1, %k[slot]\n\t"
          "leaq -1(%[slot]), %[tmp]\n\t"
          "shlq $8, %[tmp]\n\t"
          "leaq (%[meta], %[tmp], 1), %[lp]\n\t"
          "movq (%[lp]), %[cur_t]\n\t"
          "movq %[cur_t], %[next_t]\n\t"
          "subq 8(%[lp]), %[next_t]\n\t"
          "cmpq 24(%[lp]), %[next_t]\n\t"
          "jae %l[refresh_push]\n\t"
          "leaq 1(%[cur_t]), %[next_t]\n\t"
          "andq 32(%[lp]), %[cur_t]\n\t"
          "movq 16(%[lp]), %[tmp]\n\t"
          "movq %[val], (%[tmp], %[cur_t], 8)\n\t"
          "movq %[next_t], (%[lp])\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax?\"\n\t"
          ".long 0x53053053\n\t"
          "4:\n\t"
          "jmp 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [slot] "=&r"(slot), [cur_t] "=&r"(cur_t),
            [next_t] "=&r"(next_t), [lp] "=&r"(lp), [v] "=&x"(v_cpu)
          : [rseq_off] "r"(rseq_off), [cam] "r"(core.prod_cam.data()),
            [meta] "r"(core.meta.data()), [val] "r"(val)
          : "memory", "cc"
          : refresh_push, slow_push);
      cons_park.notify_if_waiters();
      return RseqPushResult::Pushed;
    }

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
    const std::ptrdiff_t rseq_off = __rseq_offset;

  retry_pop:
    // Tier 1 Consumer (`cons_biased == 1`):
    {
      std::uint64_t tmp = 0;
      std::uint64_t slot = 0;
      std::uint64_t cur_h = 0;
      std::uint64_t next_h = 0;
      std::uint64_t val_out = 0;
      // WHY ASM: (a) OS/ABI control-flow contract — Tier-1 biased consumer
      // dequeue with inline `tail_cache` refresh inside a restartable
      // `[2f, 3f)` Linux `rseq` critical section.
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw?\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "leaq 1b(%%rip), %[tmp]\n\t"
          "movq %[tmp], %%fs:8(%[rseq_off])\n\t"
          "2:\n\t"
          "cmpb $0, (%[biased])\n\t"
          "je %l[tier2_pop]\n\t"
          "movl %%fs:4(%[rseq_off]), %k[tmp]\n\t"
          "movzwl 2(%[cam]), %k[slot]\n\t"
          "incl %k[tmp]\n\t"
          "cmpl %k[slot], %k[tmp]\n\t"
          "jne %l[slow_pop]\n\t"
          "movq 128(%[meta]), %[cur_h]\n\t"
          "cmpq 136(%[meta]), %[cur_h]\n\t"
          "jne 7f\n\t"
          "movq (%[meta]), %[tmp]\n\t"
          "cmpq %[tmp], %[cur_h]\n\t"
          "je %l[steal_pop]\n\t"
          "movq %[tmp], 136(%[meta])\n\t"
          "7:\n\t"
          "leaq 1(%[cur_h]), %[next_h]\n\t"
          "andq 152(%[meta]), %[cur_h]\n\t"
          "movq 144(%[meta]), %[tmp]\n\t"
          "movq (%[tmp], %[cur_h], 8), %[val_out]\n\t"
          "movq %[next_h], 128(%[meta])\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax?\"\n\t"
          ".long 0x53053053\n\t"
          "4:\n\t"
          "jmp 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [slot] "=&r"(slot), [cur_h] "=&r"(cur_h),
            [next_h] "=&r"(next_h), [val_out] "=&r"(val_out)
          : [rseq_off] "r"(rseq_off), [biased] "r"(&core.cons_biased),
            [cam] "r"(core.cons_cam.data()), [meta] "r"(core.meta.data())
          : "memory", "cc"
          : tier2_pop, steal_pop, slow_pop);
      out = val_out;
      prod_park.notify_if_waiters();
      return true;
    }

  tier2_pop:
    {
      __m128i v_cpu{};  // NOLINT(misc-const-correctness)
      std::uint64_t tmp = 0;
      std::uint64_t slot = 0;
      std::uint64_t cur_h = 0;
      std::uint64_t next_h = 0;
      std::uint64_t val_out = 0;
      std::uintptr_t lp = 0;
      // WHY ASM: (a) OS/ABI control-flow contract — Tier-2 7-slot SSE2
      // consumer CAM + `cons_exclusive` guard + single-commit head advance
      // inside a restartable `[2f, 3f)` Linux `rseq` critical section.
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw?\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "leaq 1b(%%rip), %[tmp]\n\t"
          "movq %[tmp], %%fs:8(%[rseq_off])\n\t"
          "2:\n\t"
          "movl %%fs:4(%[rseq_off]), %k[tmp]\n\t"
          "cmpl $8192, %k[tmp]\n\t"
          "jae %l[slow_pop]\n\t"
          "incl %k[tmp]\n\t"
          "movd %k[tmp], %[v]\n\t"
          "pshuflw $0, %[v], %[v]\n\t"
          "punpcklqdq %[v], %[v]\n\t"
          "pcmpeqw (%[cam]), %[v]\n\t"
          "pmovmskb %[v], %k[slot]\n\t"
          "bsfl %k[slot], %k[slot]\n\t"
          "je %l[slow_pop]\n\t"
          "shrl $1, %k[slot]\n\t"
          "leaq -1(%[slot]), %[tmp]\n\t"
          "shlq $8, %[tmp]\n\t"
          "leaq (%[meta], %[tmp], 1), %[lp]\n\t"
          "cmpb $0, 160(%[lp])\n\t"
          "je %l[steal_pop]\n\t"
          "movq 128(%[lp]), %[cur_h]\n\t"
          "cmpq 136(%[lp]), %[cur_h]\n\t"
          "jne 7f\n\t"
          "movq (%[lp]), %[tmp]\n\t"
          "cmpq %[tmp], %[cur_h]\n\t"
          "je %l[steal_pop]\n\t"
          "movq %[tmp], 136(%[lp])\n\t"
          "7:\n\t"
          "leaq 1(%[cur_h]), %[next_h]\n\t"
          "andq 152(%[lp]), %[cur_h]\n\t"
          "movq 144(%[lp]), %[tmp]\n\t"
          "movq (%[tmp], %[cur_h], 8), %[val_out]\n\t"
          "movq %[next_h], 128(%[lp])\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax?\"\n\t"
          ".long 0x53053053\n\t"
          "4:\n\t"
          "jmp 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [slot] "=&r"(slot), [cur_h] "=&r"(cur_h),
            [next_h] "=&r"(next_h), [lp] "=&r"(lp), [val_out] "=&r"(val_out),
            [v] "=&x"(v_cpu)
          : [rseq_off] "r"(rseq_off), [cam] "r"(core.cons_cam.data()),
            [meta] "r"(core.meta.data())
          : "memory", "cc"
          : steal_pop, slow_pop);
      out = val_out;
      prod_park.notify_if_waiters();
      return true;
    }

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
    const std::ptrdiff_t rseq_off = __rseq_offset;

  retry_rich_push:
    {
      std::uint64_t tmp = 0;
      std::uint64_t slot = 0;
      std::uint64_t cur_t = 0;
      std::uint64_t next_t = 0;
      // WHY ASM: (a) OS/ABI control-flow contract — Tier-1 biased slot-index
      // reservation (64B-strided turn check + tail commit; payload constructed
      // outside `[2f, 3f)`).
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw?\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "leaq 1b(%%rip), %[tmp]\n\t"
          "movq %[tmp], %%fs:8(%[rseq_off])\n\t"
          "2:\n\t"
          "cmpb $0, (%[biased])\n\t"
          "je %l[tier2_rich_push]\n\t"
          "movl %%fs:4(%[rseq_off]), %k[tmp]\n\t"
          "movzwl 2(%[cam]), %k[slot]\n\t"
          "incl %k[tmp]\n\t"
          "cmpl %k[slot], %k[tmp]\n\t"
          "jne %l[slow_rich_push]\n\t"
          "movq (%[meta]), %[cur_t]\n\t"
          "leaq (%[cur_t], %[cur_t]), %[next_t]\n\t"
          "movq %[cur_t], %[tmp]\n\t"
          "andq 32(%[meta]), %[tmp]\n\t"
          "shlq $6, %[tmp]\n\t"
          "addq 16(%[meta]), %[tmp]\n\t"
          "cmpq %[next_t], (%[tmp])\n\t"
          "jne %l[full_rich_push]\n\t"
          "incq %[cur_t]\n\t"
          "movq %[cur_t], (%[meta])\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax?\"\n\t"
          ".long 0x53053053\n\t"
          "4:\n\t"
          "jmp 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [slot] "=&r"(slot), [cur_t] "=&r"(cur_t),
            [next_t] "=&r"(next_t)
          : [rseq_off] "r"(rseq_off), [biased] "r"(&core.prod_biased),
            [cam] "r"(core.prod_cam.data()), [meta] "r"(core.meta.data())
          : "memory", "cc"
          : tier2_rich_push, full_rich_push, slow_rich_push);
      out_lane = 0U;
      out_ticket = cur_t - 1U;
      return RseqPushResult::Pushed;
    }

  tier2_rich_push:
    {
      __m128i v_cpu{};  // NOLINT(misc-const-correctness)
      std::uint64_t tmp = 0;
      std::uint64_t slot = 0;
      std::uint64_t cur_t = 0;
      std::uint64_t next_t = 0;
      std::uintptr_t lp = 0;
      // WHY ASM: (a) OS/ABI control-flow contract — Tier-2 7-slot SSE2 CAM
      // slot-index reservation inside `[2f, 3f)`.
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw?\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "leaq 1b(%%rip), %[tmp]\n\t"
          "movq %[tmp], %%fs:8(%[rseq_off])\n\t"
          "2:\n\t"
          "movl %%fs:4(%[rseq_off]), %k[tmp]\n\t"
          "cmpl $8192, %k[tmp]\n\t"
          "jae %l[slow_rich_push]\n\t"
          "incl %k[tmp]\n\t"
          "movd %k[tmp], %[v]\n\t"
          "pshuflw $0, %[v], %[v]\n\t"
          "punpcklqdq %[v], %[v]\n\t"
          "pcmpeqw (%[cam]), %[v]\n\t"
          "pmovmskb %[v], %k[slot]\n\t"
          "bsfl %k[slot], %k[slot]\n\t"
          "je %l[slow_rich_push]\n\t"
          "shrl $1, %k[slot]\n\t"
          "leaq -1(%[slot]), %[tmp]\n\t"
          "shlq $8, %[tmp]\n\t"
          "leaq (%[meta], %[tmp], 1), %[lp]\n\t"
          "movq (%[lp]), %[cur_t]\n\t"
          "leaq (%[cur_t], %[cur_t]), %[next_t]\n\t"
          "movq %[cur_t], %[tmp]\n\t"
          "andq 32(%[lp]), %[tmp]\n\t"
          "shlq $6, %[tmp]\n\t"
          "addq 16(%[lp]), %[tmp]\n\t"
          "cmpq %[next_t], (%[tmp])\n\t"
          "jne %l[full_rich_push]\n\t"
          "incq %[cur_t]\n\t"
          "movq %[cur_t], (%[lp])\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax?\"\n\t"
          ".long 0x53053053\n\t"
          "4:\n\t"
          "jmp 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [slot] "=&r"(slot), [cur_t] "=&r"(cur_t),
            [next_t] "=&r"(next_t), [lp] "=&r"(lp), [v] "=&x"(v_cpu)
          : [rseq_off] "r"(rseq_off), [cam] "r"(core.prod_cam.data()),
            [meta] "r"(core.meta.data())
          : "memory", "cc"
          : full_rich_push, slow_rich_push);
      out_lane = static_cast<std::uint32_t>(slot - 1U);
      out_ticket = cur_t - 1U;
      return RseqPushResult::Pushed;
    }

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
    const std::ptrdiff_t rseq_off = __rseq_offset;

  retry_rich_pop:
    {
      std::uint64_t tmp = 0;
      std::uint64_t slot = 0;
      std::uint64_t cur_h = 0;
      std::uint64_t next_h = 0;
      // WHY ASM: (a) OS/ABI control-flow contract — Tier-1 biased consumer
      // slot-index claim inside `[2f, 3f)` (payload moved outside `rseq`).
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw?\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "leaq 1b(%%rip), %[tmp]\n\t"
          "movq %[tmp], %%fs:8(%[rseq_off])\n\t"
          "2:\n\t"
          "cmpb $0, (%[biased])\n\t"
          "je %l[tier2_rich_pop]\n\t"
          "movl %%fs:4(%[rseq_off]), %k[tmp]\n\t"
          "movzwl 2(%[cam]), %k[slot]\n\t"
          "incl %k[tmp]\n\t"
          "cmpl %k[slot], %k[tmp]\n\t"
          "jne %l[slow_rich_pop]\n\t"
          "movq 128(%[meta]), %[cur_h]\n\t"
          "leaq 1(%[cur_h], %[cur_h]), %[next_h]\n\t"
          "movq %[cur_h], %[tmp]\n\t"
          "andq 152(%[meta]), %[tmp]\n\t"
          "shlq $6, %[tmp]\n\t"
          "addq 144(%[meta]), %[tmp]\n\t"
          "cmpq %[next_h], (%[tmp])\n\t"
          "jne %l[steal_rich_pop]\n\t"
          "incq %[cur_h]\n\t"
          "movq %[cur_h], 128(%[meta])\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax?\"\n\t"
          ".long 0x53053053\n\t"
          "4:\n\t"
          "jmp 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [slot] "=&r"(slot), [cur_h] "=&r"(cur_h),
            [next_h] "=&r"(next_h)
          : [rseq_off] "r"(rseq_off), [biased] "r"(&core.cons_biased),
            [cam] "r"(core.cons_cam.data()), [meta] "r"(core.meta.data())
          : "memory", "cc"
          : tier2_rich_pop, steal_rich_pop, slow_rich_pop);
      out_lane = 0U;
      out_ticket = cur_h - 1U;
      return true;
    }

  tier2_rich_pop:
    {
      __m128i v_cpu{};  // NOLINT(misc-const-correctness)
      std::uint64_t tmp = 0;
      std::uint64_t slot = 0;
      std::uint64_t cur_h = 0;
      std::uint64_t next_h = 0;
      std::uintptr_t lp = 0;
      // WHY ASM: (a) OS/ABI control-flow contract — Tier-2 7-slot SSE2 CAM
      // consumer slot-index claim inside `[2f, 3f)`.
      asm volatile goto(
          ".pushsection __rseq_cs, \"aw?\"\n\t"
          ".balign 32\n\t"
          "1:\n\t"
          ".long 0, 0\n\t"
          ".quad 2f, (3f - 2f), 4f\n\t"
          ".popsection\n\t"
          "6:\n\t"
          "leaq 1b(%%rip), %[tmp]\n\t"
          "movq %[tmp], %%fs:8(%[rseq_off])\n\t"
          "2:\n\t"
          "movl %%fs:4(%[rseq_off]), %k[tmp]\n\t"
          "cmpl $8192, %k[tmp]\n\t"
          "jae %l[slow_rich_pop]\n\t"
          "incl %k[tmp]\n\t"
          "movd %k[tmp], %[v]\n\t"
          "pshuflw $0, %[v], %[v]\n\t"
          "punpcklqdq %[v], %[v]\n\t"
          "pcmpeqw (%[cam]), %[v]\n\t"
          "pmovmskb %[v], %k[slot]\n\t"
          "bsfl %k[slot], %k[slot]\n\t"
          "je %l[slow_rich_pop]\n\t"
          "shrl $1, %k[slot]\n\t"
          "leaq -1(%[slot]), %[tmp]\n\t"
          "shlq $8, %[tmp]\n\t"
          "leaq (%[meta], %[tmp], 1), %[lp]\n\t"
          "cmpb $0, 160(%[lp])\n\t"
          "je %l[steal_rich_pop]\n\t"
          "movq 128(%[lp]), %[cur_h]\n\t"
          "leaq 1(%[cur_h], %[cur_h]), %[next_h]\n\t"
          "movq %[cur_h], %[tmp]\n\t"
          "andq 152(%[lp]), %[tmp]\n\t"
          "shlq $6, %[tmp]\n\t"
          "addq 144(%[lp]), %[tmp]\n\t"
          "cmpq %[next_h], (%[tmp])\n\t"
          "jne %l[steal_rich_pop]\n\t"
          "incq %[cur_h]\n\t"
          "movq %[cur_h], 128(%[lp])\n\t"
          "3:\n\t"
          ".pushsection .text.unlikely, \"ax?\"\n\t"
          ".long 0x53053053\n\t"
          "4:\n\t"
          "jmp 6b\n\t"
          ".popsection\n\t"
          : [tmp] "=&r"(tmp), [slot] "=&r"(slot), [cur_h] "=&r"(cur_h),
            [next_h] "=&r"(next_h), [lp] "=&r"(lp), [v] "=&x"(v_cpu)
          : [rseq_off] "r"(rseq_off), [cam] "r"(core.cons_cam.data()),
            [meta] "r"(core.meta.data())
          : "memory", "cc"
          : steal_rich_pop, slow_rich_pop);
      out_lane = static_cast<std::uint32_t>(slot - 1U);
      out_ticket = cur_h - 1U;
      return true;
    }

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

#endif  // THETA_ARCH_X86_64
