// FILENAME: include/theta/queue/queue_opts.hpp
// Copyright (c) 2023-2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <theta/queue/detail/common.hpp>
#include <type_traits>

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

// Legacy 8-byte atomic value concepts retained for backward compatibility.
template <typename T>
concept AtomType =
    std::default_initializable<T> && std::is_trivially_copyable_v<T> &&
    queue::detail::kCanBeAtomic<T> && (sizeof(T) <= 8);

template <typename T>
concept ZeroableAtomType = AtomType<T> && requires(const T& val) {
  { static_cast<bool>(val) } -> std::same_as<bool>;
} && (!static_cast<bool>(T{}));

// General queue element concept supporting pointers, scalars, small/large
// trivially copyable structs, over-aligned SIMD types, and move-only objects
// (e.g., `std::unique_ptr`, `std::string`, `std::vector`).
template <typename T>
concept QueueElement = (!std::is_reference_v<T>) && (!std::is_const_v<T>) &&
                       (!std::is_volatile_v<T>) && std::move_constructible<T> &&
                       std::is_nothrow_move_constructible_v<T> &&
                       std::is_nothrow_destructible_v<T>;

}  // namespace theta
