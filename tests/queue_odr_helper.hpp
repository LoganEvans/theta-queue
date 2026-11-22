// FILENAME: tests/queue_odr_helper.hpp
// Copyright (c) 2026 Logan Evans
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace theta::test_odr {

[[nodiscard]] std::uint64_t exercise_queues_in_second_tu(
    std::uint64_t seed) noexcept;

}  // namespace theta::test_odr
