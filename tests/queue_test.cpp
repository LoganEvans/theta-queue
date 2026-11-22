// FILENAME: tests/queue_test.cpp
// Copyright (c) 2023-2026 Logan Evans
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <sched.h>
#include <sys/time.h>

#include <array>
#include <atomic>
#include <barrier>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <theta/queue.hpp>
#include <thread>
#include <utility>
#include <vector>

#include "queue_odr_helper.hpp"
#include "queue_test_peer.hpp"

namespace theta {
namespace {

// ============================================================================
// Compile-time concept checks (`AtomType`, `ZeroableAtomType`, `QueueElement`)
// ============================================================================
struct TooLarge {
  uint64_t a{0};
  uint64_t b{0};
};

struct NonZeroDefault {
  uint64_t v{1};
  constexpr explicit operator bool() const noexcept { return v != 0; }
};

struct TaggedPtr {
  uintptr_t raw{0};
  constexpr explicit operator bool() const noexcept { return raw != 0; }
};

struct Payload256 {
  std::array<uint64_t, 32> words{};
};

struct alignas(64) AlignedSimd64 {
  std::array<uint64_t, 8> lanes{};
};

// Non-default-constructible, move-only type that tracks live instances.
class TrackedMoveOnly {
 public:
  explicit TrackedMoveOnly(std::atomic<int64_t>* live, uint64_t val) noexcept
      : live_(live), val_(val) {
    if (live_ != nullptr) {
      live_->fetch_add(1, std::memory_order_relaxed);
    }
  }

  TrackedMoveOnly(TrackedMoveOnly&& other) noexcept
      : live_(std::exchange(other.live_, nullptr)),
        val_(std::exchange(other.val_, 0U)) {}

  TrackedMoveOnly& operator=(TrackedMoveOnly&& other) noexcept {
    if (this != &other) {
      if (live_ != nullptr) {
        live_->fetch_sub(1, std::memory_order_relaxed);
      }
      live_ = std::exchange(other.live_, nullptr);
      val_ = std::exchange(other.val_, 0U);
    }
    return *this;
  }

  TrackedMoveOnly(const TrackedMoveOnly&) = delete;
  TrackedMoveOnly& operator=(const TrackedMoveOnly&) = delete;

  ~TrackedMoveOnly() noexcept {
    if (live_ != nullptr) {
      live_->fetch_sub(1, std::memory_order_relaxed);
    }
  }

  [[nodiscard]] uint64_t value() const noexcept { return val_; }

 private:
  std::atomic<int64_t>* live_;
  uint64_t val_;
};

static_assert(AtomType<uint64_t*>);
static_assert(AtomType<uint64_t>);
static_assert(AtomType<uint32_t>);
static_assert(AtomType<TaggedPtr>);
static_assert(!AtomType<TooLarge>);
static_assert(!AtomType<std::string>);

static_assert(ZeroableAtomType<uint64_t*>);
static_assert(ZeroableAtomType<uint64_t>);
static_assert(ZeroableAtomType<TaggedPtr>);
static_assert(!ZeroableAtomType<NonZeroDefault>);
static_assert(!ZeroableAtomType<TooLarge>);

static_assert(QueueElement<uint64_t>);
static_assert(QueueElement<TooLarge>);
static_assert(QueueElement<Payload256>);
static_assert(QueueElement<AlignedSimd64>);
static_assert(QueueElement<std::unique_ptr<uint64_t>>);
static_assert(QueueElement<std::string>);
static_assert(QueueElement<TrackedMoveOnly>);
static_assert(!QueueElement<const uint64_t>);
static_assert(!QueueElement<uint64_t&>);

// ============================================================================
// Typed Single-Threaded FIFO + Wraparound Tests Across All Bounded Queues
// ============================================================================
template <typename Q>
class BoundedFifoTests : public ::testing::Test {
 public:
  static Q make_uut(QueueOpts opts = QueueOpts{}) { return Q{opts}; }
};

using AllBoundedQueueTypes =
    ::testing::Types<SPSCQueue<uint64_t*>, SPSCQueue<uint64_t*, 32>,
                     MPSCQueue<uint64_t*>, MPSCQueue<uint64_t*, 32>,
                     MPSCQueue<uint64_t*, 512>, FastMPSCQueue<uint64_t*>,
                     FastMPSCQueue<uint64_t*, 32>, MPMCQueue<uint64_t*>,
                     MPMCQueue<uint64_t*, 32>, MPMCQueue<uint64_t*, 512>,
                     FastMPMCQueue<uint64_t*>, FastMPMCQueue<uint64_t*, 32>>;
TYPED_TEST_SUITE(BoundedFifoTests, AllBoundedQueueTypes);

TYPED_TEST(BoundedFifoTests, TryPushPopFifo) {
  constexpr std::size_t kSize = 10;
  std::array<uint64_t, kSize> storage{};
  auto queue = this->make_uut();

  for (std::size_t i = 0; i < kSize; ++i) {
    storage[i] = 100 + i;
    EXPECT_EQ(queue.size(), i);
    ASSERT_TRUE(queue.try_push(&storage[i]));
  }

  uint64_t expected = 100;
  std::size_t expected_size = kSize;
  while (queue.size() > 0) {
    EXPECT_EQ(queue.size(), expected_size--);
    const auto v = queue.try_pop();
    if (!v.has_value()) {
      FAIL() << "Expected item in queue";
    }
    EXPECT_EQ(**v, expected++);
  }
  EXPECT_EQ(expected, 110U);
  EXPECT_FALSE(queue.try_pop().has_value());
}

TYPED_TEST(BoundedFifoTests, TryPushFullAndWraparound) {
  constexpr std::size_t kRequestedSize = 32;
  auto queue = this->make_uut(QueueOpts{}.set_max_size(kRequestedSize));
  const std::size_t cap = queue.capacity();
  ASSERT_GE(cap, 31U);
  std::vector<uint64_t> storage(cap, 0);
  uint64_t extra = 1000;
  uint64_t single = 2000;

  for (int rev = 0; rev < 32; ++rev) {
    for (std::size_t i = 0; i < cap; ++i) {
      storage[i] = 100 + i;
      EXPECT_EQ(queue.size(), i);
      ASSERT_TRUE(queue.try_push(&storage[i]));
    }

    EXPECT_EQ(queue.size(), cap);
    EXPECT_FALSE(queue.try_push(&extra));

    uint64_t expected = 100;
    std::size_t expected_size = cap;
    while (true) {
      const auto v = queue.try_pop();
      if (!v.has_value()) {
        break;
      }
      EXPECT_EQ(queue.size(), --expected_size);
      EXPECT_EQ(**v, expected++);
    }

    EXPECT_EQ(expected_size, 0U);
    EXPECT_EQ(queue.size(), 0U);

    ASSERT_TRUE(queue.try_push(&single));
    const auto popped = queue.try_pop();
    if (!popped.has_value()) {
      FAIL() << "Expected single item";
    }
    EXPECT_EQ(**popped, 2000U);
    EXPECT_FALSE(queue.try_pop().has_value());
  }
}

// ============================================================================
// Multi-Producer / Multi-Consumer Stress Tests
// ============================================================================
template <typename Q>
class MpmcTryStressTests : public ::testing::Test {
 public:
  static Q make_uut(QueueOpts opts = QueueOpts{}) { return Q{opts}; }
};

using MpmcQueueTypes =
    ::testing::Types<MPMCQueue<uint64_t*>, MPMCQueue<uint64_t*, 32>,
                     MPMCQueue<uint64_t*, 512>, FastMPMCQueue<uint64_t*>,
                     FastMPMCQueue<uint64_t*, 32>>;
TYPED_TEST_SUITE(MpmcTryStressTests, MpmcQueueTypes);

TYPED_TEST(MpmcTryStressTests, MultithreadedTryStress) {
  constexpr uint64_t kPushesPerThread = arch::kTsanEnabled ? 10'000 : 50'000;
  constexpr std::size_t kNumThreads = 4;
  std::array<std::thread, kNumThreads> threads;
  std::array<std::atomic<uint64_t>, kNumThreads> sums{};

  auto queue = this->make_uut(QueueOpts{}.set_max_size(32));

  for (std::size_t tx = 0; tx < kNumThreads; ++tx) {
    threads[tx] = std::thread([&, tx]() {
      std::mt19937_64 gen(tx + 1);
      std::uniform_real_distribution<double> unif(0.0, 1.0);

      uint64_t sum = 0;
      uint64_t num_pushes = 0;

      while (num_pushes < kPushesPerThread) {
        if (unif(gen) < 0.5) {
          if (auto v = queue.try_pop(); v.has_value()) {
            sum += **v;
            delete *v;
          }
        } else {
          auto* v = new uint64_t{num_pushes++};
          while (!queue.try_push(v)) {
            if (auto other = queue.try_pop(); other.has_value()) {
              sum += **other;
              delete *other;
            }
          }
        }
      }

      sums[tx].store(sum, std::memory_order_relaxed);
    });
  }

  uint64_t total_sum = 0;
  for (std::size_t tx = 0; tx < kNumThreads; ++tx) {
    threads[tx].join();
    total_sum += sums[tx].load(std::memory_order_relaxed);
  }

  while (auto v = queue.try_pop()) {
    total_sum += **v;
    delete *v;
  }

  EXPECT_EQ(total_sum,
            kNumThreads * kPushesPerThread * (kPushesPerThread - 1) / 2);
}

TEST(SPSCQueueTest, MultithreadedExactFifoAndBlocking) {
  constexpr uint64_t kCount = arch::kTsanEnabled ? 40'000 : 200'000;
  SPSCQueue<uint64_t, 64> queue{};
  EXPECT_EQ(queue.capacity(), 63U);

  std::thread prod([&]() {
    for (uint64_t i = 0; i < kCount; ++i) {
      queue.push(i);
    }
  });

  std::thread cons([&]() {
    for (uint64_t expected = 0; expected < kCount; ++expected) {
      const uint64_t got = queue.pop();
      ASSERT_EQ(got, expected);
    }
  });

  prod.join();
  cons.join();
  EXPECT_TRUE(queue.empty());
}

TEST(MPMCQueueTest, BlockingPushPopAndSizeUnderWaiters) {
  MPMCQueue<uint64_t> queue{QueueOpts{}.set_max_size(4)};
  EXPECT_EQ(queue.capacity(), 4U);

  std::atomic<bool> consumer_entered{false};
  uint64_t popped_value = 999;

  std::thread consumer([&]() {
    consumer_entered.store(true, std::memory_order_release);
    popped_value = queue.pop();
  });

  while (!consumer_entered.load(std::memory_order::acquire)) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(5));

  EXPECT_EQ(queue.size(), 0U);
  EXPECT_FALSE(queue.try_pop().has_value());

  queue.push(0U);
  consumer.join();
  EXPECT_EQ(popped_value, 0U);

  for (uint64_t i = 1; i <= 4; ++i) {
    ASSERT_TRUE(queue.try_push(i));
  }
  EXPECT_FALSE(queue.try_push(5U));

  std::atomic<bool> producer_finished{false};
  std::thread producer([&]() {
    queue.push(5U);
    producer_finished.store(true, std::memory_order_release);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_FALSE(producer_finished.load(std::memory_order::acquire));

  EXPECT_EQ(queue.pop(), 1U);
  producer.join();
  EXPECT_TRUE(producer_finished.load(std::memory_order::acquire));

  EXPECT_EQ(queue.pop(), 2U);
  EXPECT_EQ(queue.pop(), 3U);
  EXPECT_EQ(queue.pop(), 4U);
  EXPECT_EQ(queue.pop(), 5U);
  EXPECT_EQ(queue.size(), 0U);
}

TEST(MPMCQueueTest, BlockingMultithreadedExactConservation) {
  constexpr std::size_t kProducers = 4;
  constexpr std::size_t kConsumers = 4;
  constexpr std::size_t kItemsPerProducer = 12000;
  constexpr std::size_t kTotalItems = kProducers * kItemsPerProducer;

  MPMCQueue<uint32_t, 16> queue{};
  std::vector<std::atomic<uint32_t>> seen(kTotalItems);
  for (auto& s : seen) {
    s.store(0, std::memory_order_relaxed);
  }

  std::vector<std::thread> consumers;
  consumers.reserve(kConsumers);
  for (std::size_t c = 0; c < kConsumers; ++c) {
    consumers.emplace_back([&]() {
      for (std::size_t i = 0; i < kItemsPerProducer; ++i) {
        const uint32_t id = queue.pop();
        ASSERT_LT(id, kTotalItems);
        seen[id].fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  std::vector<std::thread> producers;
  producers.reserve(kProducers);
  for (std::size_t p = 0; p < kProducers; ++p) {
    producers.emplace_back([&, p]() {
      const auto base = static_cast<uint32_t>(p * kItemsPerProducer);
      for (uint32_t i = 0; i < kItemsPerProducer; ++i) {
        queue.push(base + i);
      }
    });
  }

  for (auto& t : producers) {
    t.join();
  }
  for (auto& t : consumers) {
    t.join();
  }

  for (std::size_t i = 0; i < kTotalItems; ++i) {
    ASSERT_EQ(seen[i].load(std::memory_order_relaxed), 1U)
        << "Item " << i << " was lost or duplicated";
  }
}

TEST(MPSCQueueTest, MultiProducerSingleConsumerCountReporting) {
  constexpr std::size_t kProducers = 4;
  constexpr uint64_t kItemsPerProducer = 8000;
  constexpr uint64_t kTotalItems = kProducers * kItemsPerProducer;

  MPSCQueue<uint64_t> queue{QueueOpts{}.set_max_size(64)};
  EXPECT_EQ(queue.capacity(), 64U);

  // Zero values are now supported cleanly in MPSCQueue too.
  std::size_t reported_size = 0;
  ASSERT_TRUE(queue.try_push(0U, &reported_size));
  EXPECT_EQ(reported_size, 1U);
  ASSERT_EQ(queue.try_pop(), std::optional<uint64_t>{0U});

  std::vector<std::thread> producers;
  producers.reserve(kProducers);
  for (std::size_t p = 0; p < kProducers; ++p) {
    producers.emplace_back([&, p]() {
      const uint64_t base = p * kItemsPerProducer;
      for (uint64_t i = 0; i < kItemsPerProducer; ++i) {
        while (!queue.try_push(base + i)) {
          std::this_thread::yield();
        }
      }
    });
  }

  std::vector<uint8_t> seen(kTotalItems, 0);
  uint64_t popped_count = 0;
  while (popped_count < kTotalItems) {
    if (auto item = queue.try_pop(); item.has_value()) {
      ASSERT_LT(*item, kTotalItems);
      ++seen[*item];
      ++popped_count;
    } else {
      std::this_thread::yield();
    }
  }

  for (auto& t : producers) {
    t.join();
  }

  for (uint64_t id = 0; id < kTotalItems; ++id) {
    ASSERT_EQ(seen[id], 1U);
  }
}

// ============================================================================
// Unbounded Queue Suite (`USPSCQueue`, `UMPSCQueue`, `UMPMCQueue`)
// ============================================================================
template <typename Q>
class UnboundedBasicTests : public ::testing::Test {};

// Use small `kLgSlab = 4` (16 items/slab) to exercise circular wrap, slab
// chaining, and slab pool recycling heavily.
using AllUnboundedTypes =
    ::testing::Types<USPSCQueue<uint64_t, 4>, UMPSCQueue<uint64_t, 4>,
                     UMPMCQueue<uint64_t, 4>, UnboundedQueue<uint64_t>>;
TYPED_TEST_SUITE(UnboundedBasicTests, AllUnboundedTypes);

TYPED_TEST(UnboundedBasicTests, ShallowAndDeepBurstFifo) {
  TypeParam q;
  EXPECT_TRUE(q.empty());
  EXPECT_EQ(q.size(), 0U);

  uint64_t next_in = 0;
  uint64_t next_out = 0;

  for (const std::size_t burst : {5UZ, 16UZ, 17UZ, 4096UZ, 9UZ, 500UZ}) {
    for (std::size_t i = 0; i < burst; ++i) {
      q.push(next_in++);
    }
    EXPECT_FALSE(q.empty());

    for (std::size_t i = 0; i < burst / 2U; ++i) {
      const auto v = q.try_pop();
      if (!v.has_value()) {
        FAIL() << "Expected item during partial drain";
      }
      ASSERT_EQ(*v, next_out++);
    }
  }

  while (const auto v = q.try_pop()) {
    ASSERT_EQ(*v, next_out++);
  }
  EXPECT_EQ(next_out, next_in);
  EXPECT_TRUE(q.empty());
  EXPECT_EQ(q.size(), 0U);
}

TEST(UnboundedQueueTest, USPSC_CrossThreadExactOrderAndBlocking) {
  constexpr uint64_t kTotal = arch::kTsanEnabled ? 40'000 : 250'000;
  USPSCQueue<uint64_t, 5> q;  // 32-item slabs

  std::thread cons([&]() {
    for (uint64_t expected = 0; expected < kTotal; ++expected) {
      ASSERT_EQ(q.pop(), expected);
    }
  });

  std::thread prod([&]() {
    for (uint64_t i = 0; i < kTotal; ++i) {
      q.push(i);
    }
  });

  prod.join();
  cons.join();
  EXPECT_TRUE(q.empty());
}

TEST(UnboundedQueueTest, UMPSC_MultiProducerSingleConsumerConservation) {
  constexpr std::size_t kProducers = 4;
  constexpr uint64_t kPerProducer = arch::kTsanEnabled ? 8'000 : 30'000;
  constexpr uint64_t kTotal = kProducers * kPerProducer;

  UMPSCQueue<uint64_t, 5> q;
  std::vector<std::thread> producers;
  producers.reserve(kProducers);
  for (std::size_t p = 0; p < kProducers; ++p) {
    producers.emplace_back([&, p]() {
      const uint64_t base = p * kPerProducer;
      for (uint64_t i = 0; i < kPerProducer; ++i) {
        q.push(base + i);
      }
    });
  }

  std::vector<uint8_t> seen(kTotal, 0);
  for (uint64_t i = 0; i < kTotal; ++i) {
    const uint64_t id = q.pop();
    ASSERT_LT(id, kTotal);
    ++seen[id];
  }

  for (auto& t : producers) {
    t.join();
  }
  for (uint64_t i = 0; i < kTotal; ++i) {
    ASSERT_EQ(seen[i], 1U);
  }
  EXPECT_TRUE(q.empty());
}

TEST(UnboundedQueueTest, UMPMC_MultithreadedExactConservation) {
  constexpr std::size_t kProducers = 4;
  constexpr std::size_t kConsumers = 4;
  constexpr uint64_t kPerThread = arch::kTsanEnabled ? 8'000 : 25'000;
  constexpr uint64_t kTotal = kProducers * kPerThread;

  UMPMCQueue<uint64_t, 5> q;  // 32-slot slabs force rapid slab recycling
  std::vector<std::atomic<uint32_t>> seen(kTotal);
  for (auto& s : seen) {
    s.store(0, std::memory_order_relaxed);
  }

  std::vector<std::thread> consumers;
  consumers.reserve(kConsumers);
  for (std::size_t c = 0; c < kConsumers; ++c) {
    consumers.emplace_back([&]() {
      for (uint64_t i = 0; i < kPerThread; ++i) {
        const uint64_t id = q.pop();
        ASSERT_LT(id, kTotal);
        seen[id].fetch_add(1U, std::memory_order_relaxed);
      }
    });
  }

  std::vector<std::thread> producers;
  producers.reserve(kProducers);
  for (std::size_t p = 0; p < kProducers; ++p) {
    producers.emplace_back([&, p]() {
      const uint64_t base = p * kPerThread;
      for (uint64_t i = 0; i < kPerThread; ++i) {
        q.push(base + i);
      }
    });
  }

  for (auto& t : producers) {
    t.join();
  }
  for (auto& t : consumers) {
    t.join();
  }

  for (uint64_t i = 0; i < kTotal; ++i) {
    ASSERT_EQ(seen[i].load(std::memory_order_relaxed), 1U);
  }
  EXPECT_TRUE(q.empty());
}

// ============================================================================
// Rich Payload Tests: Move-Only Lifetime Tracking, 256B Structs & 64B Alignment
// ============================================================================
template <typename Q>
void verify_move_only_lifetime() {
  std::atomic<int64_t> live{0};
  {
    Q q;
    for (uint64_t i = 1; i <= 20; ++i) {
      q.emplace(&live, i);
    }
    EXPECT_EQ(live.load(), 20);

    for (uint64_t i = 1; i <= 12; ++i) {
      const auto popped = q.try_pop();
      if (!popped.has_value()) {
        FAIL() << "Expected TrackedMoveOnly item";
      }
      EXPECT_EQ(popped->value(), i);
    }
    EXPECT_EQ(live.load(), 8);
  }
  EXPECT_EQ(live.load(), 0);
}

TEST(RichPayloadTest, MoveOnlyNoDefaultAndDestructorCleanupOnDestroy) {
  verify_move_only_lifetime<SPSCQueue<TrackedMoveOnly, 32>>();
  verify_move_only_lifetime<MPSCQueue<TrackedMoveOnly, 32>>();
  verify_move_only_lifetime<FastMPSCQueue<TrackedMoveOnly, 32>>();
  verify_move_only_lifetime<MPMCQueue<TrackedMoveOnly, 32>>();
  verify_move_only_lifetime<FastMPMCQueue<TrackedMoveOnly, 32>>();
  verify_move_only_lifetime<USPSCQueue<TrackedMoveOnly, 3>>();
  verify_move_only_lifetime<UMPSCQueue<TrackedMoveOnly, 3>>();
  verify_move_only_lifetime<UMPMCQueue<TrackedMoveOnly, 3>>();
}

TEST(RichPayloadTest, LargeAndOverAlignedPayloads) {
  MPMCQueue<Payload256, 16> large_mpmc;
  UMPMCQueue<Payload256, 4> large_umpmc;
  SPSCQueue<AlignedSimd64, 16> simd_spsc;
  MPMCQueue<AlignedSimd64, 16> simd_mpmc;
  UMPMCQueue<AlignedSimd64, 4> simd_umpmc;

  for (uint64_t i = 1; i <= 12; ++i) {
    Payload256 p{};
    p.words[0] = i;
    p.words[31] = i * 100U;
    ASSERT_TRUE(large_mpmc.try_push(p));
    large_umpmc.push(p);

    AlignedSimd64 a{};
    a.lanes[0] = i;
    a.lanes[7] = i * 7U;
    ASSERT_TRUE(simd_spsc.try_push(a));
    ASSERT_TRUE(simd_mpmc.try_push(a));
    simd_umpmc.push(a);
  }

  for (uint64_t i = 1; i <= 12; ++i) {
    const auto p1 = large_mpmc.pop();
    const auto p2 = large_umpmc.pop();
    EXPECT_EQ(p1.words[0], i);
    EXPECT_EQ(p1.words[31], i * 100U);
    EXPECT_EQ(p2.words[0], i);
    EXPECT_EQ(p2.words[31], i * 100U);

    const auto a1 = simd_spsc.pop();
    const auto a2 = simd_mpmc.pop();
    const auto a3 = simd_umpmc.pop();
    EXPECT_EQ(a1.lanes[7], i * 7U);
    EXPECT_EQ(a2.lanes[7], i * 7U);
    EXPECT_EQ(a3.lanes[7], i * 7U);
  }
}

// ============================================================================
// 2 kHz SIGPROF Preemption Storm Across Bounded & Unbounded MPMC
// ============================================================================
void sigprof_preempt_handler(int /*signo*/) {
  const int saved_errno = errno;
  sched_yield();
  errno = saved_errno;
}

TEST(MPMCQueueTest, PreemptionStormStress) {
  struct sigaction sa{};
  struct sigaction old_sa{};
  sa.sa_handler = sigprof_preempt_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART;
  ASSERT_EQ(sigaction(SIGPROF, &sa, &old_sa), 0);

  itimerval timer{};
  timer.it_interval.tv_sec = 0;
  timer.it_interval.tv_usec = 500;  // 2 kHz preemption storm
  timer.it_value = timer.it_interval;
  ASSERT_EQ(setitimer(ITIMER_PROF, &timer, nullptr), 0);

  constexpr std::size_t kPairs = 4;
  constexpr uint64_t kOpsPerThread = 5000;
  MPMCQueue<uint64_t, 8> bounded_q{};
  MPMCQueue<uint64_t, 512> hybrid_q{};
  FastMPMCQueue<uint64_t, 64> rseq_q{};
  UMPMCQueue<uint64_t, 3> unbounded_q{};
  std::atomic<uint64_t> sum_pushed{0};
  std::atomic<uint64_t> sum_popped_b{0};
  std::atomic<uint64_t> sum_popped_h{0};
  std::atomic<uint64_t> sum_popped_r{0};
  std::atomic<uint64_t> sum_popped_u{0};

  std::vector<std::thread> threads;
  threads.reserve(kPairs * 2);
  for (std::size_t i = 0; i < kPairs; ++i) {
    threads.emplace_back([&, i]() {
      uint64_t local_sum = 0;
      const uint64_t base = i * kOpsPerThread + 1;
      for (uint64_t k = 0; k < kOpsPerThread; ++k) {
        const uint64_t v = base + k;
        bounded_q.push(v);
        hybrid_q.push(v);
        rseq_q.push(v);
        unbounded_q.push(v);
        local_sum += v;
      }
      sum_pushed.fetch_add(local_sum, std::memory_order_relaxed);
    });
    threads.emplace_back([&]() {
      uint64_t local_b = 0;
      uint64_t local_h = 0;
      uint64_t local_r = 0;
      uint64_t local_u = 0;
      for (uint64_t k = 0; k < kOpsPerThread; ++k) {
        local_b += bounded_q.pop();
        local_h += hybrid_q.pop();
        local_r += rseq_q.pop();
        local_u += unbounded_q.pop();
      }
      sum_popped_b.fetch_add(local_b, std::memory_order_relaxed);
      sum_popped_h.fetch_add(local_h, std::memory_order_relaxed);
      sum_popped_r.fetch_add(local_r, std::memory_order_relaxed);
      sum_popped_u.fetch_add(local_u, std::memory_order_relaxed);
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  const itimerval stop_timer{};
  EXPECT_EQ(setitimer(ITIMER_PROF, &stop_timer, nullptr), 0);
  EXPECT_EQ(sigaction(SIGPROF, &old_sa, nullptr), 0);

  EXPECT_EQ(sum_pushed.load(), sum_popped_b.load());
  EXPECT_EQ(sum_pushed.load(), sum_popped_h.load());
  EXPECT_EQ(sum_pushed.load(), sum_popped_r.load());
  EXPECT_EQ(sum_pushed.load(), sum_popped_u.load());
  EXPECT_EQ(bounded_q.size(), 0U);
  EXPECT_EQ(hybrid_q.size(), 0U);
  EXPECT_EQ(rseq_q.size(), 0U);
  EXPECT_EQ(unbounded_q.size(), 0U);
}

// ============================================================================
// 2-Tier `rseq` + 7-Slot CAM Overflow & Phase Migration Tests (> 7 Producers)
// ============================================================================
TEST(FastRseqQueueTest, Overflow10ProducersAndPhaseMigration) {
  constexpr std::size_t kProducers = 10;
  constexpr uint64_t kPerProducer = arch::kTsanEnabled ? 3'000 : 12'000;
  constexpr uint64_t kTotal = kProducers * kPerProducer;

  FastMPSCQueue<uint64_t, 256> mpsc{};
  FastMPMCQueue<uint64_t, 256> mpmc{};
  std::atomic<uint64_t> mpsc_sum{0};
  std::atomic<uint64_t> mpmc_sum{0};

  std::thread mpsc_cons([&]() {
    uint64_t local = 0;
    uint64_t count = 0;
    while (count < kTotal) {
      local += mpsc.pop();
      ++count;
    }
    mpsc_sum.store(local, std::memory_order_relaxed);
  });

  constexpr std::size_t kMpmcCons = 4;
  std::vector<std::thread> mpmc_consumers;
  mpmc_consumers.reserve(kMpmcCons);
  for (std::size_t c = 0; c < kMpmcCons; ++c) {
    mpmc_consumers.emplace_back([&, c]() {
      const uint64_t quota =
          (kTotal / kMpmcCons) + (c == 0 ? (kTotal % kMpmcCons) : 0U);
      uint64_t local = 0;
      for (uint64_t i = 0; i < quota; ++i) {
        local += mpmc.pop();
      }
      mpmc_sum.fetch_add(local, std::memory_order_relaxed);
    });
  }

  std::vector<std::thread> producers;
  producers.reserve(kProducers);
  for (std::size_t p = 0; p < kProducers; ++p) {
    producers.emplace_back([&, p]() {
      const uint64_t base = p * kPerProducer + 1U;
      for (uint64_t i = 0; i < kPerProducer; ++i) {
        mpsc.push(base + i);
        mpmc.push(base + i);
      }
    });
  }

  for (auto& t : producers) {
    t.join();
  }
  mpsc_cons.join();
  for (auto& t : mpmc_consumers) {
    t.join();
  }

  const uint64_t expected_sum = (kTotal * (kTotal + 1U)) / 2U;
  EXPECT_EQ(mpsc_sum.load(), expected_sum);
  EXPECT_EQ(mpmc_sum.load(), expected_sum);
  EXPECT_TRUE(mpsc.empty());
  EXPECT_TRUE(mpmc.empty());
}

// ============================================================================
// First-Class Batch (`try_push_bulk` / `try_pop_bulk`) & Zero-Copy `drain`
// ============================================================================
template <typename Q>
void verify_bulk_and_drain(Q& q, bool exact_cap) {
  std::vector<uint64_t> src(64);
  for (std::size_t i = 0; i < src.size(); ++i) {
    src[i] = 1000U + i;
  }

  const std::size_t pushed = q.try_push_bulk(std::span<const uint64_t>{src});
  if (exact_cap) {
    ASSERT_EQ(pushed, src.size());
  } else {
    ASSERT_GT(pushed, 0U);
  }

  // Drain first 20 items via early-stopping `bool(uint64_t)` callback
  std::vector<uint64_t> seen;
  const std::size_t first_drain = q.drain([&](uint64_t v) {
    seen.push_back(v);
    return seen.size() < 20U;
  });
  EXPECT_EQ(first_drain, std::min<std::size_t>(20U, pushed));

  // Pop next 16 items via `try_pop_bulk`
  std::array<uint64_t, 16> batch_out{};
  const std::size_t popped_bulk = q.try_pop_bulk(batch_out.data(), 16U);
  for (std::size_t i = 0; i < popped_bulk; ++i) {
    seen.push_back(batch_out[i]);
  }

  // Drain remainder via `void(uint64_t)` callback
  const std::size_t rem_drain = q.drain([&](uint64_t v) { seen.push_back(v); });
  EXPECT_EQ(first_drain + popped_bulk + rem_drain, pushed);
  ASSERT_EQ(seen.size(), pushed);
  for (std::size_t i = 0; i < pushed; ++i) {
    EXPECT_EQ(seen[i], 1000U + i);
  }
  EXPECT_TRUE(q.empty());
}

TEST(BulkAndDrainTest, AllBoundedAndUnboundedVariants) {
  SPSCQueue<uint64_t, 128> spsc;
  MPSCQueue<uint64_t, 512> mpsc;
  FastMPSCQueue<uint64_t, 128> fast_mpsc;
  MPMCQueue<uint64_t, 512> mpmc;
  FastMPMCQueue<uint64_t, 128> fast_mpmc;
  USPSCQueue<uint64_t, 5> uspsc;
  UMPSCQueue<uint64_t, 5> umpsc;
  UMPMCQueue<uint64_t, 5> umpmc;

  verify_bulk_and_drain(spsc, true);
  verify_bulk_and_drain(mpsc, true);
  verify_bulk_and_drain(fast_mpsc, true);
  verify_bulk_and_drain(mpmc, true);
  verify_bulk_and_drain(fast_mpmc, true);
  verify_bulk_and_drain(uspsc, true);
  verify_bulk_and_drain(umpsc, true);
  verify_bulk_and_drain(umpmc, true);
}

TEST(BulkAndDrainTest, SPSC_CrossThreadBulkStreamingAndMoveOnlyDrain) {
  std::atomic<int64_t> live{0};
  {
    SPSCQueue<TrackedMoveOnly, 64> move_q;
    for (uint64_t i = 1; i <= 40; ++i) {
      ASSERT_TRUE(move_q.try_emplace(&live, i));
    }
    EXPECT_EQ(live.load(), 40);

    uint64_t acc = 0;
    const std::size_t n = move_q.drain(
        [&](TrackedMoveOnly&& item) noexcept { acc += item.value(); }, 25U);
    EXPECT_EQ(n, 25U);
    EXPECT_EQ(acc, (25U * 26U) / 2U);
    EXPECT_EQ(live.load(), 15);
  }
  EXPECT_EQ(live.load(), 0);
}

TEST(ConventionsAndPeerTest, MultiTuOdrAndTwoTierStateTransitions) {
  EXPECT_EQ(test_odr::exercise_queues_in_second_tu(100U), 136U);

  SPSCQueue<uint64_t, 16> spsc;
  EXPECT_EQ(queue::detail::QueueTestPeer::spsc_cons_blocking_mode(spsc), 0U);
  ASSERT_TRUE(spsc.try_push(7U));
  EXPECT_EQ(spsc.try_pop(), 7U);
  EXPECT_EQ(queue::detail::QueueTestPeer::spsc_cons_blocking_mode(spsc), 0U);

  std::thread delayed_prod([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    spsc.push(42U);
  });
  EXPECT_EQ(spsc.pop(), 42U);
  delayed_prod.join();
  EXPECT_GE(queue::detail::QueueTestPeer::spsc_cons_blocking_mode(spsc), 2U);

  FastMPSCQueue<uint64_t, 64> fast_mpsc;
  ASSERT_TRUE(fast_mpsc.try_push(99U));
  if constexpr (queue::detail::QueueTestPeer::uses_rseq_lanes<
                    FastMPSCQueue<uint64_t, 64>>()) {
    EXPECT_EQ(queue::detail::QueueTestPeer::rseq_prod_biased(fast_mpsc), 1U);
    EXPECT_EQ(queue::detail::QueueTestPeer::rseq_num_prod(fast_mpsc), 1U);
    EXPECT_EQ(queue::detail::QueueTestPeer::rseq_overflow_ops(fast_mpsc), 0U);
  }
  EXPECT_EQ(fast_mpsc.try_pop(), 99U);
}

TEST(FastRseqQueueTest, OversubscribedMultiRoundBarrierConservation) {
  constexpr std::ptrdiff_t kPairs = arch::kTsanEnabled ? 6 : 12;
  constexpr std::size_t kBatch = arch::kTsanEnabled ? 512U : 4096U;
  constexpr int kRounds = arch::kTsanEnabled ? 6 : 40;
  constexpr std::size_t kTotalPerRound =
      static_cast<std::size_t>(kPairs) * kBatch;
  constexpr uint64_t kExpectedRoundSum =
      (static_cast<uint64_t>(kTotalPerRound) * (kTotalPerRound + 1U)) / 2U;

  FastMPMCQueue<uint64_t, 1024> q;
  std::barrier start_batch(2 * kPairs);
  std::barrier end_batch(2 * kPairs);
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> round_sum{0};

  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>((2 * kPairs) - 1));

  for (std::ptrdiff_t p = 1; p < kPairs; ++p) {
    workers.emplace_back([&, p]() {
      while (true) {
        start_batch.arrive_and_wait();
        if (stop.load(std::memory_order_relaxed)) {
          return;
        }
        const std::size_t base = static_cast<std::size_t>(p) * kBatch;
        for (std::size_t i = 0; i < kBatch; ++i) {
          const uint64_t val = base + i + 1U;
          while (!q.try_push(val)) {
            arch::cpu_relax();
          }
        }
        end_batch.arrive_and_wait();
      }
    });
  }

  for (std::ptrdiff_t c = 0; c < kPairs; ++c) {
    workers.emplace_back([&]() {
      while (true) {
        start_batch.arrive_and_wait();
        if (stop.load(std::memory_order_relaxed)) {
          return;
        }
        uint64_t local_sum = 0;
        for (std::size_t i = 0; i < kBatch; ++i) {
          while (true) {
            if (auto v = q.try_pop(); v.has_value()) {
              local_sum += *v;
              break;
            }
            arch::cpu_relax();
          }
        }
        round_sum.fetch_add(local_sum, std::memory_order_relaxed);
        end_batch.arrive_and_wait();
      }
    });
  }

  for (int r = 0; r < kRounds; ++r) {
    round_sum.store(0U, std::memory_order_relaxed);
    start_batch.arrive_and_wait();
    for (std::size_t i = 0; i < kBatch; ++i) {
      const uint64_t val = i + 1U;
      while (!q.try_push(val)) {
        arch::cpu_relax();
      }
    }
    end_batch.arrive_and_wait();
    EXPECT_EQ(round_sum.load(std::memory_order_relaxed), kExpectedRoundSum);
  }

  stop.store(true, std::memory_order_relaxed);
  start_batch.arrive_and_wait();
  for (auto& w : workers) {
    w.join();
  }
  EXPECT_TRUE(q.empty());
}

TEST(BulkAndDrainTest, SmallRingWrapAroundStripedParkWakeup) {
  MPMCQueue<uint64_t, 4> q;
  // Advance head and tail to index 3 (`3 & 3 == 3`) so a 2-item batch wraps
  // across ring slots `3` and `0` (`cap < kStripes`).
  for (uint64_t i = 0; i < 3U; ++i) {
    ASSERT_TRUE(q.try_push(i));
    ASSERT_EQ(q.try_pop(), i);
  }
  std::atomic<uint64_t> sum{0};
  std::thread c0([&]() { sum.fetch_add(q.pop(), std::memory_order_relaxed); });
  std::thread c1([&]() { sum.fetch_add(q.pop(), std::memory_order_relaxed); });
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  const std::array<uint64_t, 2> batch{10U, 20U};
  EXPECT_EQ(q.try_push_bulk(std::span<const uint64_t>{batch}), 2U);
  c0.join();
  c1.join();
  EXPECT_EQ(sum.load(), 30U);
}

TEST(ConventionsAndPeerTest,
     LazyParkConcurrentFirstActivationAndDepthOneDrain) {
  // 1. Concurrent multi-thread first-park race on both AsymmetricParkLot
  //    (`UMPSCQueue`) and StripedParkLot (`MPMCQueue`).
  {
    MPMCQueue<uint64_t, 32> mpmc;
    EXPECT_EQ(queue::detail::QueueTestPeer::mpmc_cons_blocking_mode(mpmc), 0U);
    constexpr std::ptrdiff_t kWaiters = 8;
    std::barrier ready(kWaiters + 1);
    std::atomic<uint64_t> sum{0};
    std::vector<std::thread> consumers;
    consumers.reserve(static_cast<std::size_t>(kWaiters));
    for (std::ptrdiff_t i = 0; i < kWaiters; ++i) {
      consumers.emplace_back([&]() {
        ready.arrive_and_wait();
        sum.fetch_add(mpmc.pop(), std::memory_order_relaxed);
      });
    }
    ready.arrive_and_wait();
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (queue::detail::QueueTestPeer::mpmc_cons_blocking_mode(mpmc) < 2U &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    EXPECT_GE(static_cast<unsigned>(
                  queue::detail::QueueTestPeer::mpmc_cons_blocking_mode(mpmc)),
              2U);
    for (uint64_t v = 1U; v <= static_cast<uint64_t>(kWaiters); ++v) {
      mpmc.push(v);
    }
    for (auto& t : consumers) {
      t.join();
    }
    EXPECT_EQ(sum.load(),
              static_cast<uint64_t>(kWaiters * (kWaiters + 1)) / 2U);
  }

  // 2. Depth-1 push/pop/empty/drain transitions across SPSC, USPSC, FastMPSC.
  SPSCQueue<uint64_t, 16> spsc;
  USPSCQueue<uint64_t, 3> uspsc;
  FastMPSCQueue<uint64_t, 16> fmpsc;
  for (uint64_t i = 1U; i <= 64U; ++i) {
    ASSERT_TRUE(spsc.try_push(i));
    ASSERT_EQ(spsc.try_pop(), i);
    ASSERT_EQ(spsc.try_pop(), std::nullopt);
    ASSERT_EQ(spsc.drain([](uint64_t) {}), 0U);

    ASSERT_TRUE(uspsc.try_push(i));
    ASSERT_EQ(uspsc.try_pop(), i);
    ASSERT_EQ(uspsc.try_pop(), std::nullopt);
    ASSERT_EQ(uspsc.drain([](uint64_t) {}), 0U);

    ASSERT_TRUE(fmpsc.try_push(i));
    ASSERT_EQ(fmpsc.try_pop(), i);
    ASSERT_EQ(fmpsc.try_pop(), std::nullopt);
    ASSERT_EQ(fmpsc.drain([](uint64_t) {}), 0U);
  }
}

}  // namespace
}  // namespace theta
