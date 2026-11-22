// FILENAME: tests/queue_odr_helper.cpp
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT

#include "queue_odr_helper.hpp"

#include <cstdint>
#include <theta/queue.hpp>

namespace theta::test_odr {

std::uint64_t exercise_queues_in_second_tu(std::uint64_t seed) noexcept {
  SPSCQueue<std::uint64_t, 16> spsc;
  MPSCQueue<std::uint64_t, 32> mpsc;
  FastMPSCQueue<std::uint64_t, 32> fast_mpsc;
  MPMCQueue<std::uint64_t, 32> mpmc;
  FastMPMCQueue<std::uint64_t, 32> fast_mpmc;
  USPSCQueue<std::uint64_t, 4> uspsc;
  UMPSCQueue<std::uint64_t, 4> umpsc;
  UMPMCQueue<std::uint64_t, 4> umpmc;

  spsc.push(seed + 1U);
  mpsc.push(spsc.pop() + 2U);
  fast_mpsc.push(mpsc.pop() + 3U);
  mpmc.push(fast_mpsc.pop() + 4U);
  fast_mpmc.push(mpmc.pop() + 5U);
  uspsc.push(fast_mpmc.pop() + 6U);
  umpsc.push(uspsc.pop() + 7U);
  umpmc.push(umpsc.pop() + 8U);
  return umpmc.pop();
}

}  // namespace theta::test_odr
