// FILENAME: include/theta/queue/queue-opts.h
// Copyright (c) 2023-2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>

#include "theta/queue/defs.h"

namespace theta {

// Ordering policy for multi-producer bounded queues (`MPSCQueue`, `MPMCQueue`):
// - `StrictFIFO`: Global linearizable ticket FIFO (`HybridFaaRing`).
// - `PerProducer`: Per-producer FIFO using 2-Tier Linux `rseq` + 7-slot SIMD
//   CPU CAM lanes with automatic `MEMBARRIER_RSEQ` migration & MPMC fallback.
enum class QueueOrder : std::uint8_t {
  StrictFIFO = 0,
  PerProducer = 1,
};

class QueueOpts {
 public:
  [[nodiscard]] constexpr std::size_t max_size() const noexcept {
    return max_size_;
  }

  constexpr QueueOpts& set_max_size(std::size_t val) noexcept {
    max_size_ = val;
    return *this;
  }

 private:
  std::size_t max_size_{kHardwareDestructiveInterferenceSize};
};

}  // namespace theta
