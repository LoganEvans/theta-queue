// FILENAME: benchmarks/queue_benchmark.cpp
// Copyright (c) 2023-2026 Logan Evans
// SPDX-License-Identifier: MIT

#include <benchmark/benchmark.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <theta/arch.hpp>
#include <theta/queue.hpp>
#include <thread>
#include <utility>
#include <vector>

#if defined(THETA_QUEUE_HAS_ATOMIC_QUEUE)
#include <atomic_queue/atomic_queue.h>
#endif

#if defined(THETA_QUEUE_HAS_ABSL)
#include <absl/synchronization/mutex.h>
#endif

#if defined(THETA_QUEUE_HAS_THETA_MUTEX)
#include <theta/mutex.hpp>
#if __has_include(<theta/condition_variable.hpp>)
#include <theta/condition_variable.hpp>
#define THETA_QUEUE_HAS_THETA_CV 1
#endif
#endif

#if defined(THETA_QUEUE_HAS_FOLLY)
#include <folly/MPMCQueue.h>
#include <folly/ProducerConsumerQueue.h>
#include <folly/concurrency/UnboundedQueue.h>
#endif

#if defined(THETA_QUEUE_HAS_BOOST)
#include <boost/lockfree/queue.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#endif

#if defined(THETA_QUEUE_HAS_MOODYCAMEL)
#if __has_include(<moodycamel/concurrentqueue.h>)
#include <moodycamel/concurrentqueue.h>
#else
#include <concurrentqueue.h>
#endif
#endif

namespace theta {
namespace {

constexpr std::size_t kBenchQueueCapacity = 1024;
constexpr std::size_t kBatchPerProducer = 8192;

inline void cpu_relax() noexcept { ::theta::arch::cpu_relax(); }

// Orders logical CPUs cleanly for hybrid + SMT x86-64 CPUs:
//   1. Primary threads of SMT P-cores
//   2. Secondary SMT sibling threads of P-cores
//   3. Single-threaded E-cores
inline int physical_first_cpu_for_thread(int thread_idx) {
#if defined(__linux__)
  static const std::vector<int> cpu_order = []() {
    const int num_cores = static_cast<int>(std::thread::hardware_concurrency());
    std::vector<int> smt_primaries;
    std::vector<int> smt_siblings;
    std::vector<int> single_cores;

    for (int cpu = 0; cpu < num_cores; ++cpu) {
      std::array<char, 128> path{};
      std::snprintf(
          path.data(), path.size(),
          "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
      std::array<char, 64> buf{};
      int first_sib = cpu;
      bool has_sep = false;
      if (FILE* fp = std::fopen(path.data(), "r")) {
        if (std::fgets(buf.data(), static_cast<int>(buf.size()), fp) !=
            nullptr) {
          char* end_ptr = nullptr;
          const long parsed = std::strtol(buf.data(), &end_ptr, 10);
          if (end_ptr != buf.data()) {
            first_sib = static_cast<int>(parsed);
          }
          has_sep = (std::strchr(buf.data(), ',') != nullptr) ||
                    (std::strchr(buf.data(), '-') != nullptr);
        }
        std::fclose(fp);
      }
      if (has_sep) {
        if (first_sib == cpu) {
          smt_primaries.push_back(cpu);
        } else {
          smt_siblings.push_back(cpu);
        }
      } else {
        single_cores.push_back(cpu);
      }
    }

    std::vector<int> order;
    order.insert(order.end(), smt_primaries.begin(), smt_primaries.end());
    order.insert(order.end(), smt_siblings.begin(), smt_siblings.end());
    order.insert(order.end(), single_cores.begin(), single_cores.end());
    if (order.empty()) {
      order.push_back(0);
    }
    return order;
  }();
  return cpu_order[static_cast<std::size_t>(thread_idx) % cpu_order.size()];
#else
  return thread_idx;
#endif
}

inline void pin_thread_to_core(int thread_idx, int total_threads) {
#if defined(__linux__)
  const int num_cores = static_cast<int>(std::thread::hardware_concurrency());
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  if (num_cores > 0 && total_threads <= num_cores) {
    CPU_SET(static_cast<std::size_t>(physical_first_cpu_for_thread(thread_idx)),
            &cpuset);
  } else {
    for (int c = 0; c < std::max(1, num_cores); ++c) {
      CPU_SET(static_cast<std::size_t>(c), &cpuset);
    }
  }
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
  (void)thread_idx;
  (void)total_threads;
#endif
}

template <typename Q>
concept NonBlockingQueueAdaptor =
    std::default_initializable<Q> && requires(Q q, int* v) {
      { q.try_push(v) } -> std::same_as<bool>;
      { q.try_pop() } -> std::same_as<std::optional<int*>>;
    };

template <typename Q>
concept BlockingQueueAdaptor =
    NonBlockingQueueAdaptor<Q> && requires(Q q, int* v) {
      q.push(v);
      { q.pop() } -> std::same_as<int*>;
    };

// ---------------------------------------------------------------------------
// Queue Adaptors (Capacity = 1024)
// ---------------------------------------------------------------------------

struct ThetaMPMCStaticAdaptor {
  bool try_push(int* v) { return queue.try_push(v); }
  std::optional<int*> try_pop() { return queue.try_pop(); }
  void push(int* v) { queue.push(v); }
  int* pop() { return queue.pop(); }

  MPMCQueue<int*, kBenchQueueCapacity> queue{};
};

struct ThetaMPMCDynamicAdaptor {
  bool try_push(int* v) { return queue.try_push(v); }
  std::optional<int*> try_pop() { return queue.try_pop(); }
  void push(int* v) { queue.push(v); }
  int* pop() { return queue.pop(); }

  MPMCQueue<int*> queue{QueueOpts{}.set_max_size(kBenchQueueCapacity)};
};

struct ThetaSPSCStaticAdaptor {
  bool try_push(int* v) { return queue.try_push(v); }
  std::optional<int*> try_pop() { return queue.try_pop(); }
  void push(int* v) { queue.push(v); }
  int* pop() { return queue.pop(); }

  SPSCQueue<int*, kBenchQueueCapacity * 2> queue{};
};

struct ThetaMPSCAdaptor {
  bool try_push(int* v) { return queue.try_push(v); }
  std::optional<int*> try_pop() { return queue.try_pop(); }
  void push(int* v) { queue.push(v); }
  int* pop() { return queue.pop(); }

  MPSCQueue<int*> queue{QueueOpts{}.set_max_size(kBenchQueueCapacity)};
};

struct ThetaFastMPSCAdaptor {
  bool try_push(int* v) { return queue.try_push(v); }
  std::optional<int*> try_pop() { return queue.try_pop(); }
  void push(int* v) { queue.push(v); }
  int* pop() { return queue.pop(); }

  FastMPSCQueue<int*, kBenchQueueCapacity> queue{};
};

struct ThetaFastMPMCAdaptor {
  bool try_push(int* v) { return queue.try_push(v); }
  std::optional<int*> try_pop() { return queue.try_pop(); }
  void push(int* v) { queue.push(v); }
  int* pop() { return queue.pop(); }

  FastMPMCQueue<int*, kBenchQueueCapacity> queue{};
};

struct ThetaUSPSCAdaptor {
  bool try_push(int* v) { return queue.try_push(v); }
  std::optional<int*> try_pop() { return queue.try_pop(); }
  void push(int* v) { queue.push(v); }
  int* pop() { return queue.pop(); }

  USPSCQueue<int*, 9> queue{};
};

struct ThetaUMPSCAdaptor {
  bool try_push(int* v) { return queue.try_push(v); }
  std::optional<int*> try_pop() { return queue.try_pop(); }
  void push(int* v) { queue.push(v); }
  int* pop() { return queue.pop(); }

  UMPSCQueue<int*, 9> queue{};
};

struct ThetaUMPMCAdaptor {
  bool try_push(int* v) { return queue.try_push(v); }
  std::optional<int*> try_pop() { return queue.try_pop(); }
  void push(int* v) { queue.push(v); }
  int* pop() { return queue.pop(); }

  UMPMCQueue<int*, 9> queue{};
};

#if defined(THETA_QUEUE_HAS_ATOMIC_QUEUE)
struct AtomicQueueAdaptor {
  bool try_push(int* v) { return queue.try_push(v); }
  std::optional<int*> try_pop() {
    int* v = nullptr;
    if (queue.try_pop(v)) {
      return v;
    }
    return std::nullopt;
  }
  void push(int* v) { queue.push(v); }
  int* pop() { return queue.pop(); }

  atomic_queue::AtomicQueue</*T=*/int*,
                            /*SIZE=*/kBenchQueueCapacity,
                            /*NIL=*/nullptr,
                            /*MINIMIZE_CONTENTION=*/true,
                            /*MAXIMIZE_THROUGHPUT=*/true,
                            /*TOTAL_ORDER=*/false,
                            /*SPSC=*/false>
      queue{};
};

struct AtomicQueueSPSCAdaptor {
  bool try_push(int* v) { return queue.try_push(v); }
  std::optional<int*> try_pop() {
    int* v = nullptr;
    if (queue.try_pop(v)) {
      return v;
    }
    return std::nullopt;
  }

  atomic_queue::AtomicQueue</*T=*/int*,
                            /*SIZE=*/kBenchQueueCapacity,
                            /*NIL=*/nullptr,
                            /*MINIMIZE_CONTENTION=*/false,
                            /*MAXIMIZE_THROUGHPUT=*/true,
                            /*TOTAL_ORDER=*/false,
                            /*SPSC=*/true>
      queue{};
};
#endif

#if defined(THETA_QUEUE_HAS_FOLLY)
struct FollyMPMCAdaptor {
  bool try_push(int* v) { return queue.write(v); }
  std::optional<int*> try_pop() {
    int* v = nullptr;
    if (queue.read(v)) {
      return v;
    }
    return std::nullopt;
  }
  void push(int* v) { queue.blockingWrite(v); }
  int* pop() {
    int* v = nullptr;
    queue.blockingRead(v);
    return v;
  }

  folly::MPMCQueue<int*> queue{kBenchQueueCapacity};
};

struct FollySPSCAdaptor {
  bool try_push(int* v) { return queue.write(v); }
  std::optional<int*> try_pop() {
    int* v = nullptr;
    if (queue.read(v)) {
      return v;
    }
    return std::nullopt;
  }

  // ProducerConsumerQueue wastes 1 slot, so size + 1 gives 1024 usable slots.
  folly::ProducerConsumerQueue<int*> queue{kBenchQueueCapacity + 1};
};

struct FollyUSPSCAdaptor {
  bool try_push(int* v) {
    queue.enqueue(v);
    return true;
  }
  std::optional<int*> try_pop() {
    int* v = nullptr;
    if (queue.try_dequeue(v)) {
      return v;
    }
    return std::nullopt;
  }
  void push(int* v) { queue.enqueue(v); }
  int* pop() { return queue.dequeue(); }

  folly::USPSCQueue<int*, true, 9> queue{};
};

struct FollyUMPMCAdaptor {
  bool try_push(int* v) {
    queue.enqueue(v);
    return true;
  }
  std::optional<int*> try_pop() {
    int* v = nullptr;
    if (queue.try_dequeue(v)) {
      return v;
    }
    return std::nullopt;
  }
  void push(int* v) { queue.enqueue(v); }
  int* pop() { return queue.dequeue(); }

  folly::UMPMCQueue<int*, true, 9> queue{};
};
#endif

#if defined(THETA_QUEUE_HAS_BOOST)
struct BoostLockfreeMPMCAdaptor {
  bool try_push(int* v) { return queue.bounded_push(v); }
  std::optional<int*> try_pop() {
    int* v = nullptr;
    if (queue.pop(v)) {
      return v;
    }
    return std::nullopt;
  }

  boost::lockfree::queue<int*, boost::lockfree::capacity<kBenchQueueCapacity>>
      queue{};
};

struct BoostLockfreeSPSCAdaptor {
  bool try_push(int* v) { return queue.push(v); }
  std::optional<int*> try_pop() {
    int* v = nullptr;
    if (queue.pop(v)) {
      return v;
    }
    return std::nullopt;
  }

  boost::lockfree::spsc_queue<int*,
                              boost::lockfree::capacity<kBenchQueueCapacity>>
      queue{};
};
#endif

#if defined(THETA_QUEUE_HAS_ABSL)
struct AbslMutexRingAdaptor {
  bool try_push(int* v) {
    const absl::MutexLock lock(mu_);
    if (count_ == kBenchQueueCapacity) {
      return false;
    }
    buf_[tail_] = v;
    tail_ = (tail_ + 1) & (kBenchQueueCapacity - 1);
    ++count_;
    return true;
  }

  std::optional<int*> try_pop() {
    const absl::MutexLock lock(mu_);
    if (count_ == 0) {
      return std::nullopt;
    }
    int* v = buf_[head_];
    head_ = (head_ + 1) & (kBenchQueueCapacity - 1);
    --count_;
    return v;
  }

  [[nodiscard]] bool not_full() const { return count_ < kBenchQueueCapacity; }
  [[nodiscard]] bool not_empty() const { return count_ > 0; }

  void push(int* v) {
    const absl::MutexLock lock(
        mu_, absl::Condition(this, &AbslMutexRingAdaptor::not_full));
    buf_[tail_] = v;
    tail_ = (tail_ + 1) & (kBenchQueueCapacity - 1);
    ++count_;
  }

  int* pop() {
    const absl::MutexLock lock(
        mu_, absl::Condition(this, &AbslMutexRingAdaptor::not_empty));
    int* v = buf_[head_];
    head_ = (head_ + 1) & (kBenchQueueCapacity - 1);
    --count_;
    return v;
  }

  absl::Mutex mu_;
  std::size_t head_{0};
  std::size_t tail_{0};
  std::size_t count_{0};
  std::vector<int*> buf_ = std::vector<int*>(kBenchQueueCapacity, nullptr);
};
#endif

#if defined(THETA_QUEUE_HAS_THETA_MUTEX)
// Exact 1:1 drop-in replacement of absl::Mutex (+ absl::Condition) with
// theta::mutex (+ theta::condition_variable) around the identical ring buffer.
struct ThetaMutexRingAdaptor {
  bool try_push(int* v) {
    {
      const theta::mutex::Guard lock(mu_);
      if (count_ == kBenchQueueCapacity) {
        return false;
      }
      buf_[tail_] = v;
      tail_ = (tail_ + 1) & (kBenchQueueCapacity - 1);
      ++count_;
    }
#if defined(THETA_QUEUE_HAS_THETA_CV)
    not_empty_.notify_one();
#endif
    return true;
  }

  std::optional<int*> try_pop() {
    int* v = nullptr;
    {
      const theta::mutex::Guard lock(mu_);
      if (count_ == 0) {
        return std::nullopt;
      }
      v = buf_[head_];
      head_ = (head_ + 1) & (kBenchQueueCapacity - 1);
      --count_;
    }
#if defined(THETA_QUEUE_HAS_THETA_CV)
    not_full_.notify_one();
#endif
    return v;
  }

#if defined(THETA_QUEUE_HAS_THETA_CV)
  void push(int* v) {
    {
      theta::mutex::Guard lock(mu_);
      not_full_.wait(lock, [this]() { return count_ < kBenchQueueCapacity; });
      buf_[tail_] = v;
      tail_ = (tail_ + 1) & (kBenchQueueCapacity - 1);
      ++count_;
    }
    not_empty_.notify_one();
  }

  int* pop() {
    int* v = nullptr;
    {
      theta::mutex::Guard lock(mu_);
      not_empty_.wait(lock, [this]() { return count_ > 0; });
      v = buf_[head_];
      head_ = (head_ + 1) & (kBenchQueueCapacity - 1);
      --count_;
    }
    not_full_.notify_one();
    return v;
  }
#endif

  theta::mutex mu_{};
#if defined(THETA_QUEUE_HAS_THETA_CV)
  theta::condition_variable not_full_{};
  theta::condition_variable not_empty_{};
#endif
  std::size_t head_{0};
  std::size_t tail_{0};
  std::size_t count_{0};
  std::vector<int*> buf_ = std::vector<int*>(kBenchQueueCapacity, nullptr);
};
#endif

// ---------------------------------------------------------------------------
// 1. Single-Thread Push+Pop Roundtrip Latency
// ---------------------------------------------------------------------------

template <NonBlockingQueueAdaptor QType>
void BM_SingleThreadTryRoundtrip(  // NOLINT(readability-identifier-naming)
    benchmark::State& state) {
  pin_thread_to_core(0, 1);
  QType queue{};
  int payload = 42;

  for (const auto _ : state) {
    bool ok = queue.try_push(&payload);
    benchmark::DoNotOptimize(ok);
    auto out = queue.try_pop();
    benchmark::DoNotOptimize(out);
  }
  state.SetItemsProcessed(state.iterations());
}

// Escapes `&queue` on the heap before the benchmark loop so LLVM SROA cannot
// promote `std::atomic` indices into scalar CPU registers — measuring genuine
// memory atomics and acquire/release barriers on both header-only and
// compiled-slow-path queues.
template <NonBlockingQueueAdaptor QType>
void BM_SingleThreadEscapedRoundtrip(  // NOLINT(readability-identifier-naming)
    benchmark::State& state) {
  pin_thread_to_core(0, 1);
  const auto queue_ptr = std::make_unique<QType>();
  QType* q = queue_ptr.get();
  benchmark::DoNotOptimize(q);
  int payload = 42;

  for (const auto _ : state) {
    bool ok = q->try_push(&payload);
    benchmark::DoNotOptimize(ok);
    auto out = q->try_pop();
    benchmark::DoNotOptimize(out);
  }
  state.SetItemsProcessed(state.iterations());
}

// ---------------------------------------------------------------------------
// 2. Cross-Core Ping-Pong Handoff Latency (2 Physical Cores)
// ---------------------------------------------------------------------------

template <NonBlockingQueueAdaptor QType, bool kUseTry>
void BM_CrossCorePingPong(  // NOLINT(readability-identifier-naming)
    benchmark::State& state) {
  QType q_ping{};
  QType q_pong{};
  std::atomic<bool> stop{false};
  std::barrier ready(2);
  int payload = 7;
  int stop_sentinel = -1;

  std::thread echo([&]() {
    pin_thread_to_core(1, 2);
    ready.arrive_and_wait();
    while (true) {
      int* msg = nullptr;
      if constexpr (kUseTry) {
        while (true) {
          if (auto res = q_ping.try_pop(); res.has_value()) {
            msg = *res;
            break;
          }
          if (stop.load(std::memory_order::acquire)) {
            return;
          }
          cpu_relax();
        }
        if (msg == &stop_sentinel) {
          return;
        }
        while (!q_pong.try_push(msg)) {
          cpu_relax();
        }
      } else {
        msg = q_ping.pop();
        if (msg == &stop_sentinel) {
          return;
        }
        q_pong.push(msg);
      }
    }
  });

  pin_thread_to_core(0, 2);
  ready.arrive_and_wait();

  for (const auto _ : state) {
    if constexpr (kUseTry) {
      while (!q_ping.try_push(&payload)) {
        cpu_relax();
      }
      while (true) {
        if (auto res = q_pong.try_pop(); res.has_value()) {
          benchmark::DoNotOptimize(res);
          break;
        }
        cpu_relax();
      }
    } else {
      q_ping.push(&payload);
      int* res = q_pong.pop();
      benchmark::DoNotOptimize(res);
    }
  }

  stop.store(true, std::memory_order::release);
  if constexpr (kUseTry) {
    (void)q_ping.try_push(&stop_sentinel);
  } else {
    q_ping.push(&stop_sentinel);
  }
  echo.join();

  // Each iteration performs 2 one-way cross-core transfers (ping + pong).
  state.SetItemsProcessed(state.iterations() * 2);
}

// ---------------------------------------------------------------------------
// 3. Producer-Consumer Throughput Harness (SPSC, MPSC, MPMC)
// ---------------------------------------------------------------------------

template <typename QType, bool kUseTry>
void run_producer_consumer(benchmark::State& state, int num_producers,
                           int num_consumers) {
  QType queue{};
  const int total_threads = num_producers + num_consumers;
  const std::size_t total_items_per_batch =
      kBatchPerProducer * static_cast<std::size_t>(num_producers);
  const std::size_t items_per_consumer =
      total_items_per_batch / static_cast<std::size_t>(num_consumers);

  std::atomic<bool> stop{false};
  std::barrier ready(total_threads);
  std::barrier start_batch(total_threads);
  std::barrier end_batch(total_threads);
  int payload = 123;

  auto push_batch = [&]() {
    for (std::size_t i = 0; i < kBatchPerProducer; ++i) {
      if constexpr (kUseTry) {
        while (!queue.try_push(&payload)) {
          cpu_relax();
        }
      } else {
        queue.push(&payload);
      }
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(total_threads - 1));

  // Producers 1 .. num_producers - 1 (Producer 0 runs on the main thread so
  // total active threads == num_producers + num_consumers).
  for (int p = 1; p < num_producers; ++p) {
    workers.emplace_back([&, p]() {
      pin_thread_to_core(p, total_threads);
      ready.arrive_and_wait();
      while (true) {
        start_batch.arrive_and_wait();
        if (stop.load(std::memory_order::relaxed)) {
          return;
        }
        push_batch();
        end_batch.arrive_and_wait();
      }
    });
  }

  for (int c = 0; c < num_consumers; ++c) {
    workers.emplace_back([&, c]() {
      pin_thread_to_core(num_producers + c, total_threads);
      ready.arrive_and_wait();
      while (true) {
        start_batch.arrive_and_wait();
        if (stop.load(std::memory_order::relaxed)) {
          return;
        }
        for (std::size_t i = 0; i < items_per_consumer; ++i) {
          int* out = nullptr;
          if constexpr (kUseTry) {
            while (true) {
              if (auto res = queue.try_pop(); res.has_value()) {
                out = *res;
                break;
              }
              cpu_relax();
            }
          } else {
            out = queue.pop();
          }
          benchmark::DoNotOptimize(out);
        }
        end_batch.arrive_and_wait();
      }
    });
  }

  pin_thread_to_core(0, total_threads);
  ready.arrive_and_wait();
  for (auto state_token : state) {
    benchmark::DoNotOptimize(state_token);
    start_batch.arrive_and_wait();
    push_batch();
    end_batch.arrive_and_wait();
  }

  stop.store(true, std::memory_order::relaxed);
  start_batch.arrive_and_wait();

  for (auto& w : workers) {
    w.join();
  }

  const auto total_processed =
      state.iterations() * static_cast<int64_t>(total_items_per_batch);
  state.SetItemsProcessed(total_processed);
}

template <NonBlockingQueueAdaptor QType>
void BM_SPSC_Try(  // NOLINT(readability-identifier-naming)
    benchmark::State& state) {
  run_producer_consumer<QType, /*kUseTry=*/true>(state, 1, 1);
}

template <NonBlockingQueueAdaptor QType>
void BM_MPSC_Try(  // NOLINT(readability-identifier-naming)
    benchmark::State& state) {
  run_producer_consumer<QType, /*kUseTry=*/true>(
      state, static_cast<int>(state.range(0)), 1);
}

template <BlockingQueueAdaptor QType>
void BM_MPSC_Blocking(  // NOLINT(readability-identifier-naming)
    benchmark::State& state) {
  run_producer_consumer<QType, /*kUseTry=*/false>(
      state, static_cast<int>(state.range(0)), 1);
}

template <NonBlockingQueueAdaptor QType>
void BM_MPMC_Try(  // NOLINT(readability-identifier-naming)
    benchmark::State& state) {
  const auto n = static_cast<int>(state.range(0));
  run_producer_consumer<QType, /*kUseTry=*/true>(state, n, n);
}

template <BlockingQueueAdaptor QType>
void BM_MPMC_Blocking(  // NOLINT(readability-identifier-naming)
    benchmark::State& state) {
  const auto n = static_cast<int>(state.range(0));
  run_producer_consumer<QType, /*kUseTry=*/false>(state, n, n);
}

// ---------------------------------------------------------------------------
// Registrations
// ---------------------------------------------------------------------------

// Group 1: Single-thread roundtrip
BENCHMARK_TEMPLATE(BM_SingleThreadTryRoundtrip, ThetaSPSCStaticAdaptor);
BENCHMARK_TEMPLATE(BM_SingleThreadTryRoundtrip, ThetaMPMCStaticAdaptor);
BENCHMARK_TEMPLATE(BM_SingleThreadTryRoundtrip, ThetaMPMCDynamicAdaptor);
BENCHMARK_TEMPLATE(BM_SingleThreadTryRoundtrip, ThetaFastMPMCAdaptor);
BENCHMARK_TEMPLATE(BM_SingleThreadTryRoundtrip, ThetaMPSCAdaptor);
BENCHMARK_TEMPLATE(BM_SingleThreadTryRoundtrip, ThetaFastMPSCAdaptor);
BENCHMARK_TEMPLATE(BM_SingleThreadTryRoundtrip, ThetaUSPSCAdaptor);
BENCHMARK_TEMPLATE(BM_SingleThreadTryRoundtrip, ThetaUMPSCAdaptor);
BENCHMARK_TEMPLATE(BM_SingleThreadTryRoundtrip, ThetaUMPMCAdaptor);
#if defined(THETA_QUEUE_HAS_ATOMIC_QUEUE)
BENCHMARK_TEMPLATE(BM_SingleThreadTryRoundtrip, AtomicQueueAdaptor);
BENCHMARK_TEMPLATE(BM_SingleThreadTryRoundtrip, AtomicQueueSPSCAdaptor);
#endif
#if defined(THETA_QUEUE_HAS_FOLLY)
BENCHMARK_TEMPLATE(BM_SingleThreadTryRoundtrip, FollyMPMCAdaptor);
BENCHMARK_TEMPLATE(BM_SingleThreadTryRoundtrip, FollySPSCAdaptor);
BENCHMARK_TEMPLATE(BM_SingleThreadTryRoundtrip, FollyUSPSCAdaptor);
BENCHMARK_TEMPLATE(BM_SingleThreadTryRoundtrip, FollyUMPMCAdaptor);
#endif
#if defined(THETA_QUEUE_HAS_BOOST)
BENCHMARK_TEMPLATE(BM_SingleThreadTryRoundtrip, BoostLockfreeMPMCAdaptor);
BENCHMARK_TEMPLATE(BM_SingleThreadTryRoundtrip, BoostLockfreeSPSCAdaptor);
#endif
#if defined(THETA_QUEUE_HAS_ABSL)
BENCHMARK_TEMPLATE(BM_SingleThreadTryRoundtrip, AbslMutexRingAdaptor);
#endif
#if defined(THETA_QUEUE_HAS_THETA_MUTEX)
BENCHMARK_TEMPLATE(BM_SingleThreadTryRoundtrip, ThetaMutexRingAdaptor);
#endif

// Group 1b: Single-thread roundtrip with escaped queue (real memory atomics)
BENCHMARK_TEMPLATE(BM_SingleThreadEscapedRoundtrip, ThetaSPSCStaticAdaptor);
BENCHMARK_TEMPLATE(BM_SingleThreadEscapedRoundtrip, ThetaUSPSCAdaptor);
BENCHMARK_TEMPLATE(BM_SingleThreadEscapedRoundtrip, ThetaFastMPSCAdaptor);
BENCHMARK_TEMPLATE(BM_SingleThreadEscapedRoundtrip, ThetaFastMPMCAdaptor);
#if defined(THETA_QUEUE_HAS_ATOMIC_QUEUE)
BENCHMARK_TEMPLATE(BM_SingleThreadEscapedRoundtrip, AtomicQueueSPSCAdaptor);
#endif
#if defined(THETA_QUEUE_HAS_FOLLY)
BENCHMARK_TEMPLATE(BM_SingleThreadEscapedRoundtrip, FollySPSCAdaptor);
BENCHMARK_TEMPLATE(BM_SingleThreadEscapedRoundtrip, FollyUSPSCAdaptor);
#endif
#if defined(THETA_QUEUE_HAS_BOOST)
BENCHMARK_TEMPLATE(BM_SingleThreadEscapedRoundtrip, BoostLockfreeSPSCAdaptor);
#endif

// Group 2: Cross-core ping-pong latency
BENCHMARK_TEMPLATE(BM_CrossCorePingPong, ThetaSPSCStaticAdaptor, true)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_CrossCorePingPong, ThetaUSPSCAdaptor, true)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_CrossCorePingPong, ThetaMPMCStaticAdaptor, true)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_CrossCorePingPong, ThetaMPMCStaticAdaptor, false)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_CrossCorePingPong, ThetaFastMPMCAdaptor, true)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_CrossCorePingPong, ThetaMPSCAdaptor, true)->UseRealTime();
BENCHMARK_TEMPLATE(BM_CrossCorePingPong, ThetaFastMPSCAdaptor, true)
    ->UseRealTime();
#if defined(THETA_QUEUE_HAS_ATOMIC_QUEUE)
BENCHMARK_TEMPLATE(BM_CrossCorePingPong, AtomicQueueAdaptor, true)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_CrossCorePingPong, AtomicQueueSPSCAdaptor, true)
    ->UseRealTime();
#endif
#if defined(THETA_QUEUE_HAS_FOLLY)
BENCHMARK_TEMPLATE(BM_CrossCorePingPong, FollyMPMCAdaptor, true)->UseRealTime();
BENCHMARK_TEMPLATE(BM_CrossCorePingPong, FollyMPMCAdaptor, false)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_CrossCorePingPong, FollySPSCAdaptor, true)->UseRealTime();
#endif
#if defined(THETA_QUEUE_HAS_BOOST)
BENCHMARK_TEMPLATE(BM_CrossCorePingPong, BoostLockfreeMPMCAdaptor, true)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_CrossCorePingPong, BoostLockfreeSPSCAdaptor, true)
    ->UseRealTime();
#endif
#if defined(THETA_QUEUE_HAS_ABSL)
BENCHMARK_TEMPLATE(BM_CrossCorePingPong, AbslMutexRingAdaptor, true)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_CrossCorePingPong, AbslMutexRingAdaptor, false)
    ->UseRealTime();
#endif
#if defined(THETA_QUEUE_HAS_THETA_MUTEX)
BENCHMARK_TEMPLATE(BM_CrossCorePingPong, ThetaMutexRingAdaptor, true)
    ->UseRealTime();
#if defined(THETA_QUEUE_HAS_THETA_CV)
BENCHMARK_TEMPLATE(BM_CrossCorePingPong, ThetaMutexRingAdaptor, false)
    ->UseRealTime();
#endif
#endif

// Group 3: SPSC Throughput (1P + 1C)
BENCHMARK_TEMPLATE(BM_SPSC_Try, ThetaSPSCStaticAdaptor)->UseRealTime();
BENCHMARK_TEMPLATE(BM_SPSC_Try, ThetaUSPSCAdaptor)->UseRealTime();
BENCHMARK_TEMPLATE(BM_SPSC_Try, ThetaMPMCStaticAdaptor)->UseRealTime();
BENCHMARK_TEMPLATE(BM_SPSC_Try, ThetaFastMPMCAdaptor)->UseRealTime();
BENCHMARK_TEMPLATE(BM_SPSC_Try, ThetaMPSCAdaptor)->UseRealTime();
BENCHMARK_TEMPLATE(BM_SPSC_Try, ThetaFastMPSCAdaptor)->UseRealTime();
#if defined(THETA_QUEUE_HAS_ATOMIC_QUEUE)
BENCHMARK_TEMPLATE(BM_SPSC_Try, AtomicQueueAdaptor)->UseRealTime();
BENCHMARK_TEMPLATE(BM_SPSC_Try, AtomicQueueSPSCAdaptor)->UseRealTime();
#endif
#if defined(THETA_QUEUE_HAS_FOLLY)
BENCHMARK_TEMPLATE(BM_SPSC_Try, FollyMPMCAdaptor)->UseRealTime();
BENCHMARK_TEMPLATE(BM_SPSC_Try, FollySPSCAdaptor)->UseRealTime();
BENCHMARK_TEMPLATE(BM_SPSC_Try, FollyUSPSCAdaptor)->UseRealTime();
#endif
#if defined(THETA_QUEUE_HAS_BOOST)
BENCHMARK_TEMPLATE(BM_SPSC_Try, BoostLockfreeMPMCAdaptor)->UseRealTime();
BENCHMARK_TEMPLATE(BM_SPSC_Try, BoostLockfreeSPSCAdaptor)->UseRealTime();
#endif
#if defined(THETA_QUEUE_HAS_ABSL)
BENCHMARK_TEMPLATE(BM_SPSC_Try, AbslMutexRingAdaptor)->UseRealTime();
#endif
#if defined(THETA_QUEUE_HAS_THETA_MUTEX)
BENCHMARK_TEMPLATE(BM_SPSC_Try, ThetaMutexRingAdaptor)->UseRealTime();
#endif

// Group 4: MPSC Throughput (NP + 1C, Try)
#define THETA_MPSC_ARGS ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(12)->UseRealTime()

BENCHMARK_TEMPLATE(BM_MPSC_Try, ThetaMPMCStaticAdaptor) THETA_MPSC_ARGS;
BENCHMARK_TEMPLATE(BM_MPSC_Try, ThetaMPSCAdaptor) THETA_MPSC_ARGS;
BENCHMARK_TEMPLATE(BM_MPSC_Try, ThetaFastMPSCAdaptor) THETA_MPSC_ARGS;
BENCHMARK_TEMPLATE(BM_MPSC_Try, ThetaUMPSCAdaptor) THETA_MPSC_ARGS;
#if defined(THETA_QUEUE_HAS_ATOMIC_QUEUE)
BENCHMARK_TEMPLATE(BM_MPSC_Try, AtomicQueueAdaptor) THETA_MPSC_ARGS;
#endif
#if defined(THETA_QUEUE_HAS_FOLLY)
BENCHMARK_TEMPLATE(BM_MPSC_Try, FollyMPMCAdaptor) THETA_MPSC_ARGS;
#endif
#if defined(THETA_QUEUE_HAS_BOOST)
BENCHMARK_TEMPLATE(BM_MPSC_Try, BoostLockfreeMPMCAdaptor) THETA_MPSC_ARGS;
#endif
#if defined(THETA_QUEUE_HAS_ABSL)
BENCHMARK_TEMPLATE(BM_MPSC_Try, AbslMutexRingAdaptor) THETA_MPSC_ARGS;
#endif
#if defined(THETA_QUEUE_HAS_THETA_MUTEX)
BENCHMARK_TEMPLATE(BM_MPSC_Try, ThetaMutexRingAdaptor) THETA_MPSC_ARGS;
#endif

// Group 5: MPMC Throughput (NP + NC, Try & Blocking)
#define THETA_MPMC_ARGS \
  ->Arg(1)->Arg(2)->Arg(4)->Arg(6)->Arg(8)->Arg(12)->UseRealTime()

BENCHMARK_TEMPLATE(BM_MPMC_Try, ThetaMPMCStaticAdaptor) THETA_MPMC_ARGS;
BENCHMARK_TEMPLATE(BM_MPMC_Try, ThetaMPMCDynamicAdaptor) THETA_MPMC_ARGS;
BENCHMARK_TEMPLATE(BM_MPMC_Try, ThetaFastMPMCAdaptor) THETA_MPMC_ARGS;
BENCHMARK_TEMPLATE(BM_MPMC_Try, ThetaUMPMCAdaptor) THETA_MPMC_ARGS;
#if defined(THETA_QUEUE_HAS_ATOMIC_QUEUE)
BENCHMARK_TEMPLATE(BM_MPMC_Try, AtomicQueueAdaptor) THETA_MPMC_ARGS;
#endif
#if defined(THETA_QUEUE_HAS_FOLLY)
BENCHMARK_TEMPLATE(BM_MPMC_Try, FollyMPMCAdaptor) THETA_MPMC_ARGS;
BENCHMARK_TEMPLATE(BM_MPMC_Try, FollyUMPMCAdaptor) THETA_MPMC_ARGS;
#endif
#if defined(THETA_QUEUE_HAS_BOOST)
BENCHMARK_TEMPLATE(BM_MPMC_Try, BoostLockfreeMPMCAdaptor) THETA_MPMC_ARGS;
#endif
#if defined(THETA_QUEUE_HAS_ABSL)
BENCHMARK_TEMPLATE(BM_MPMC_Try, AbslMutexRingAdaptor) THETA_MPMC_ARGS;
#endif
#if defined(THETA_QUEUE_HAS_THETA_MUTEX)
BENCHMARK_TEMPLATE(BM_MPMC_Try, ThetaMutexRingAdaptor) THETA_MPMC_ARGS;
#endif

BENCHMARK_TEMPLATE(BM_MPMC_Blocking, ThetaMPMCStaticAdaptor) THETA_MPMC_ARGS;
BENCHMARK_TEMPLATE(BM_MPMC_Blocking, ThetaMPMCDynamicAdaptor) THETA_MPMC_ARGS;
BENCHMARK_TEMPLATE(BM_MPMC_Blocking, ThetaFastMPMCAdaptor) THETA_MPMC_ARGS;
#if defined(THETA_QUEUE_HAS_ATOMIC_QUEUE)
BENCHMARK_TEMPLATE(BM_MPMC_Blocking, AtomicQueueAdaptor) THETA_MPMC_ARGS;
#endif
#if defined(THETA_QUEUE_HAS_FOLLY)
BENCHMARK_TEMPLATE(BM_MPMC_Blocking, FollyMPMCAdaptor) THETA_MPMC_ARGS;
#endif
#if defined(THETA_QUEUE_HAS_ABSL)
BENCHMARK_TEMPLATE(BM_MPMC_Blocking, AbslMutexRingAdaptor) THETA_MPMC_ARGS;
#endif
#if defined(THETA_QUEUE_HAS_THETA_MUTEX) && defined(THETA_QUEUE_HAS_THETA_CV)
BENCHMARK_TEMPLATE(BM_MPMC_Blocking, ThetaMutexRingAdaptor) THETA_MPMC_ARGS;
#endif

// ---------------------------------------------------------------------------
// Group 6: Element Type & Size Scaling (64B, 256B TriviallyCopyable & MoveOnly)
//   Benchmarks hybrid fused vs. separated TurnSlotArray<T> and RawSlotStorage
//   on 64B structs, 256B structs, and 32B non-trivial move-only objects.
// ---------------------------------------------------------------------------

struct Payload64 {
  std::array<uint64_t, 8> words{};
};
static_assert(sizeof(Payload64) == 64);
static_assert(std::is_trivially_copyable_v<Payload64>);

struct Payload256 {
  std::array<uint64_t, 32> words{};
};
static_assert(sizeof(Payload256) == 256);
static_assert(std::is_trivially_copyable_v<Payload256>);

// Strictly move-only, non-trivially-copyable, non-trivially-destructible 32B
// object (models std::move_only_function / Task without heap alloc noise).
struct MoveOnly32 {
  uint64_t token_{0};
  std::array<uint64_t, 3> data_{};

  MoveOnly32() noexcept = default;
  explicit MoveOnly32(uint64_t tok) noexcept
      : token_(tok), data_{tok ^ 1U, tok ^ 2U, tok ^ 3U} {}
  MoveOnly32(const MoveOnly32&) = delete;
  MoveOnly32& operator=(const MoveOnly32&) = delete;

  MoveOnly32(MoveOnly32&& other) noexcept
      : token_(std::exchange(other.token_, 0)), data_(other.data_) {}

  MoveOnly32& operator=(MoveOnly32&& other) noexcept {
    if (this != &other) {
      token_ = std::exchange(other.token_, 0);
      data_ = other.data_;
    }
    return *this;
  }

  ~MoveOnly32() noexcept { token_ = 0; }
};
static_assert(sizeof(MoveOnly32) == 32);
static_assert(!std::is_trivially_copyable_v<MoveOnly32>);

template <typename T>
T make_sample_item(uint64_t seq) noexcept {
  if constexpr (std::is_same_v<T, MoveOnly32>) {
    return MoveOnly32{seq};
  } else {
    T item{};
    item.words[0] = seq;
    return item;
  }
}

template <typename T>
struct ThetaSPSCGenericAdaptor {
  using value_type = T;
  bool try_push(T&& v) { return queue.try_push(std::move(v)); }
  bool try_pop(T& out) { return queue.try_pop(out); }

  SPSCQueue<T, kBenchQueueCapacity * 2> queue{};
};

template <typename T>
struct ThetaMPMCGenericAdaptor {
  using value_type = T;
  bool try_push(T&& v) { return queue.try_push(std::move(v)); }
  bool try_pop(T& out) { return queue.try_pop(out); }

  MPMCQueue<T, kBenchQueueCapacity> queue{};
};

template <typename T>
struct ThetaFastMPSCGenericAdaptor {
  using value_type = T;
  bool try_push(T&& v) { return queue.try_push(std::move(v)); }
  bool try_pop(T& out) { return queue.try_pop(out); }

  FastMPSCQueue<T, kBenchQueueCapacity> queue{};
};

template <typename T>
struct ThetaFastMPMCGenericAdaptor {
  using value_type = T;
  bool try_push(T&& v) { return queue.try_push(std::move(v)); }
  bool try_pop(T& out) { return queue.try_pop(out); }

  FastMPMCQueue<T, kBenchQueueCapacity> queue{};
};

template <typename T>
struct ThetaUMPMCGenericAdaptor {
  using value_type = T;
  bool try_push(T&& v) { return queue.try_push(std::move(v)); }
  bool try_pop(T& out) { return queue.try_pop(out); }

  UMPMCQueue<T, 9> queue{};
};

#if defined(THETA_QUEUE_HAS_ATOMIC_QUEUE)
template <typename T>
struct AtomicQueue2GenericAdaptor {
  using value_type = T;
  bool try_push(T&& v) { return queue.try_push(std::move(v)); }
  bool try_pop(T& out) { return queue.try_pop(out); }

  atomic_queue::AtomicQueue2<T, kBenchQueueCapacity,
                             /*MINIMIZE_CONTENTION=*/true,
                             /*MAXIMIZE_THROUGHPUT=*/true,
                             /*TOTAL_ORDER=*/false,
                             /*SPSC=*/false>
      queue{};
};
#endif

#if defined(THETA_QUEUE_HAS_FOLLY)
template <typename T>
struct FollyMPMCGenericAdaptor {
  using value_type = T;
  bool try_push(T&& v) { return queue.write(std::move(v)); }
  bool try_pop(T& out) { return queue.read(out); }

  folly::MPMCQueue<T> queue{kBenchQueueCapacity};
};

template <typename T>
struct FollySPSCGenericAdaptor {
  using value_type = T;
  bool try_push(T&& v) { return queue.write(std::move(v)); }
  bool try_pop(T& out) { return queue.read(out); }

  folly::ProducerConsumerQueue<T> queue{kBenchQueueCapacity + 1};
};
#endif

#if defined(THETA_QUEUE_HAS_ABSL)
template <typename T>
struct AbslMutexGenericAdaptor {
  using value_type = T;
  bool try_push(T&& v) {
    const absl::MutexLock lock(mu_);
    if (count_ == kBenchQueueCapacity) {
      return false;
    }
    buf_[tail_] = std::move(v);
    tail_ = (tail_ + 1) & (kBenchQueueCapacity - 1);
    ++count_;
    return true;
  }

  bool try_pop(T& out) {
    const absl::MutexLock lock(mu_);
    if (count_ == 0) {
      return false;
    }
    out = std::move(buf_[head_]);
    head_ = (head_ + 1) & (kBenchQueueCapacity - 1);
    --count_;
    return true;
  }

  absl::Mutex mu_;
  std::size_t head_{0};
  std::size_t tail_{0};
  std::size_t count_{0};
  std::vector<T> buf_ = std::vector<T>(kBenchQueueCapacity);
};
#endif

#if defined(THETA_QUEUE_HAS_THETA_MUTEX)
template <typename T>
struct ThetaMutexGenericAdaptor {
  using value_type = T;
  bool try_push(T&& v) {
    const theta::mutex::Guard lock(mu_);
    if (count_ == kBenchQueueCapacity) {
      return false;
    }
    buf_[tail_] = std::move(v);
    tail_ = (tail_ + 1) & (kBenchQueueCapacity - 1);
    ++count_;
    return true;
  }

  bool try_pop(T& out) {
    const theta::mutex::Guard lock(mu_);
    if (count_ == 0) {
      return false;
    }
    out = std::move(buf_[head_]);
    head_ = (head_ + 1) & (kBenchQueueCapacity - 1);
    --count_;
    return true;
  }

  theta::mutex mu_{};
  std::size_t head_{0};
  std::size_t tail_{0};
  std::size_t count_{0};
  std::vector<T> buf_ = std::vector<T>(kBenchQueueCapacity);
};
#endif

template <typename QType>
void BM_Generic1TRoundtrip(  // NOLINT(readability-identifier-naming)
    benchmark::State& state) {
  using T = typename QType::value_type;
  pin_thread_to_core(0, 1);
  QType queue{};
  uint64_t seq = 1;
  T out{};

  for (const auto _ : state) {
    bool ok = queue.try_push(make_sample_item<T>(seq++));
    benchmark::DoNotOptimize(ok);
    bool got = queue.try_pop(out);
    benchmark::DoNotOptimize(got);
    benchmark::DoNotOptimize(out);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(sizeof(T)));
}

template <typename QType>
void BM_GenericMPMC4x4_Try(  // NOLINT(readability-identifier-naming)
    benchmark::State& state) {
  using T = typename QType::value_type;
  constexpr int kHalf = 4;
  constexpr int kTotalThreads = kHalf * 2;
  QType queue{};

  std::atomic<bool> stop{false};
  std::barrier ready(kTotalThreads);
  std::barrier start_batch(kTotalThreads);
  std::barrier end_batch(kTotalThreads);

  auto push_batch = [&](uint64_t base) {
    for (std::size_t i = 0; i < kBatchPerProducer; ++i) {
      T item = make_sample_item<T>(base + i);
      while (!queue.try_push(std::move(item))) {
        item = make_sample_item<T>(base + i);
        cpu_relax();
      }
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(kTotalThreads - 1);
  for (int p = 1; p < kHalf; ++p) {
    workers.emplace_back([&, p]() {
      pin_thread_to_core(p, kTotalThreads);
      ready.arrive_and_wait();
      while (true) {
        start_batch.arrive_and_wait();
        if (stop.load(std::memory_order::relaxed)) {
          return;
        }
        push_batch(static_cast<uint64_t>(p) << 32U);
        end_batch.arrive_and_wait();
      }
    });
  }
  for (int c = 0; c < kHalf; ++c) {
    workers.emplace_back([&, c]() {
      pin_thread_to_core(kHalf + c, kTotalThreads);
      ready.arrive_and_wait();
      T out{};
      while (true) {
        start_batch.arrive_and_wait();
        if (stop.load(std::memory_order::relaxed)) {
          return;
        }
        for (std::size_t i = 0; i < kBatchPerProducer; ++i) {
          while (!queue.try_pop(out)) {
            cpu_relax();
          }
          benchmark::DoNotOptimize(out);
        }
        end_batch.arrive_and_wait();
      }
    });
  }

  pin_thread_to_core(0, kTotalThreads);
  ready.arrive_and_wait();
  for (auto state_token : state) {
    benchmark::DoNotOptimize(state_token);
    start_batch.arrive_and_wait();
    push_batch(1ULL);
    end_batch.arrive_and_wait();
  }

  stop.store(true, std::memory_order::relaxed);
  start_batch.arrive_and_wait();
  for (auto& w : workers) {
    w.join();
  }

  const auto items =
      state.iterations() * static_cast<int64_t>(kBatchPerProducer * kHalf);
  state.SetItemsProcessed(items);
  state.SetBytesProcessed(items * static_cast<int64_t>(sizeof(T)));
}

template <typename QType>
void BM_GenericMPSC4x1_Try(  // NOLINT(readability-identifier-naming)
    benchmark::State& state) {
  using T = typename QType::value_type;
  constexpr int kProducers = 4;
  constexpr int kTotalThreads = kProducers + 1;
  QType queue{};

  std::atomic<bool> stop{false};
  std::barrier ready(kTotalThreads);
  std::barrier start_batch(kTotalThreads);
  std::barrier end_batch(kTotalThreads);

  auto push_batch = [&](uint64_t base) {
    for (std::size_t i = 0; i < kBatchPerProducer; ++i) {
      T item = make_sample_item<T>(base + i);
      while (!queue.try_push(std::move(item))) {
        item = make_sample_item<T>(base + i);
        cpu_relax();
      }
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(kTotalThreads - 1);
  for (int p = 1; p < kProducers; ++p) {
    workers.emplace_back([&, p]() {
      pin_thread_to_core(p, kTotalThreads);
      ready.arrive_and_wait();
      while (true) {
        start_batch.arrive_and_wait();
        if (stop.load(std::memory_order::relaxed)) {
          return;
        }
        push_batch(static_cast<uint64_t>(p) << 32U);
        end_batch.arrive_and_wait();
      }
    });
  }
  workers.emplace_back([&]() {
    pin_thread_to_core(kProducers, kTotalThreads);
    ready.arrive_and_wait();
    T out{};
    while (true) {
      start_batch.arrive_and_wait();
      if (stop.load(std::memory_order::relaxed)) {
        return;
      }
      for (std::size_t i = 0; i < kBatchPerProducer * kProducers; ++i) {
        while (!queue.try_pop(out)) {
          cpu_relax();
        }
        benchmark::DoNotOptimize(out);
      }
      end_batch.arrive_and_wait();
    }
  });

  pin_thread_to_core(0, kTotalThreads);
  ready.arrive_and_wait();
  for (auto state_token : state) {
    benchmark::DoNotOptimize(state_token);
    start_batch.arrive_and_wait();
    push_batch(1ULL);
    end_batch.arrive_and_wait();
  }

  stop.store(true, std::memory_order::relaxed);
  start_batch.arrive_and_wait();
  for (auto& w : workers) {
    w.join();
  }

  const auto items =
      state.iterations() * static_cast<int64_t>(kBatchPerProducer * kProducers);
  state.SetItemsProcessed(items);
  state.SetBytesProcessed(items * static_cast<int64_t>(sizeof(T)));
}

// 64B TriviallyCopyable
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, ThetaSPSCGenericAdaptor<Payload64>);
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, ThetaMPMCGenericAdaptor<Payload64>);
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip,
                   ThetaFastMPSCGenericAdaptor<Payload64>);
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip,
                   ThetaFastMPMCGenericAdaptor<Payload64>);
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, ThetaUMPMCGenericAdaptor<Payload64>);
BENCHMARK_TEMPLATE(BM_GenericMPSC4x1_Try, ThetaMPMCGenericAdaptor<Payload64>)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_GenericMPSC4x1_Try,
                   ThetaFastMPSCGenericAdaptor<Payload64>)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try, ThetaMPMCGenericAdaptor<Payload64>)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try,
                   ThetaFastMPMCGenericAdaptor<Payload64>)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try, ThetaUMPMCGenericAdaptor<Payload64>)
    ->UseRealTime();
#if defined(THETA_QUEUE_HAS_ATOMIC_QUEUE)
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip,
                   AtomicQueue2GenericAdaptor<Payload64>);
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try, AtomicQueue2GenericAdaptor<Payload64>)
    ->UseRealTime();
#endif
#if defined(THETA_QUEUE_HAS_FOLLY)
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, FollyMPMCGenericAdaptor<Payload64>);
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, FollySPSCGenericAdaptor<Payload64>);
BENCHMARK_TEMPLATE(BM_GenericMPSC4x1_Try, FollyMPMCGenericAdaptor<Payload64>)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try, FollyMPMCGenericAdaptor<Payload64>)
    ->UseRealTime();
#endif
#if defined(THETA_QUEUE_HAS_ABSL)
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, AbslMutexGenericAdaptor<Payload64>);
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try, AbslMutexGenericAdaptor<Payload64>)
    ->UseRealTime();
#endif
#if defined(THETA_QUEUE_HAS_THETA_MUTEX)
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, ThetaMutexGenericAdaptor<Payload64>);
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try, ThetaMutexGenericAdaptor<Payload64>)
    ->UseRealTime();
#endif

// 256B TriviallyCopyable
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, ThetaSPSCGenericAdaptor<Payload256>);
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, ThetaMPMCGenericAdaptor<Payload256>);
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip,
                   ThetaFastMPSCGenericAdaptor<Payload256>);
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip,
                   ThetaFastMPMCGenericAdaptor<Payload256>);
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, ThetaUMPMCGenericAdaptor<Payload256>);
BENCHMARK_TEMPLATE(BM_GenericMPSC4x1_Try, ThetaMPMCGenericAdaptor<Payload256>)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_GenericMPSC4x1_Try,
                   ThetaFastMPSCGenericAdaptor<Payload256>)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try, ThetaMPMCGenericAdaptor<Payload256>)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try,
                   ThetaFastMPMCGenericAdaptor<Payload256>)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try, ThetaUMPMCGenericAdaptor<Payload256>)
    ->UseRealTime();
#if defined(THETA_QUEUE_HAS_ATOMIC_QUEUE)
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip,
                   AtomicQueue2GenericAdaptor<Payload256>);
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try,
                   AtomicQueue2GenericAdaptor<Payload256>)
    ->UseRealTime();
#endif
#if defined(THETA_QUEUE_HAS_FOLLY)
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, FollyMPMCGenericAdaptor<Payload256>);
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, FollySPSCGenericAdaptor<Payload256>);
BENCHMARK_TEMPLATE(BM_GenericMPSC4x1_Try, FollyMPMCGenericAdaptor<Payload256>)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try, FollyMPMCGenericAdaptor<Payload256>)
    ->UseRealTime();
#endif
#if defined(THETA_QUEUE_HAS_ABSL)
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, AbslMutexGenericAdaptor<Payload256>);
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try, AbslMutexGenericAdaptor<Payload256>)
    ->UseRealTime();
#endif
#if defined(THETA_QUEUE_HAS_THETA_MUTEX)
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, ThetaMutexGenericAdaptor<Payload256>);
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try, ThetaMutexGenericAdaptor<Payload256>)
    ->UseRealTime();
#endif

// 32B Move-Only Non-Trivial (MoveOnly32)
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, ThetaSPSCGenericAdaptor<MoveOnly32>);
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, ThetaMPMCGenericAdaptor<MoveOnly32>);
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip,
                   ThetaFastMPSCGenericAdaptor<MoveOnly32>);
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip,
                   ThetaFastMPMCGenericAdaptor<MoveOnly32>);
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, ThetaUMPMCGenericAdaptor<MoveOnly32>);
BENCHMARK_TEMPLATE(BM_GenericMPSC4x1_Try, ThetaMPMCGenericAdaptor<MoveOnly32>)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_GenericMPSC4x1_Try,
                   ThetaFastMPSCGenericAdaptor<MoveOnly32>)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try, ThetaMPMCGenericAdaptor<MoveOnly32>)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try,
                   ThetaFastMPMCGenericAdaptor<MoveOnly32>)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try, ThetaUMPMCGenericAdaptor<MoveOnly32>)
    ->UseRealTime();
#if defined(THETA_QUEUE_HAS_FOLLY)
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, FollyMPMCGenericAdaptor<MoveOnly32>);
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, FollySPSCGenericAdaptor<MoveOnly32>);
BENCHMARK_TEMPLATE(BM_GenericMPSC4x1_Try, FollyMPMCGenericAdaptor<MoveOnly32>)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try, FollyMPMCGenericAdaptor<MoveOnly32>)
    ->UseRealTime();
#endif
#if defined(THETA_QUEUE_HAS_ABSL)
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, AbslMutexGenericAdaptor<MoveOnly32>);
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try, AbslMutexGenericAdaptor<MoveOnly32>)
    ->UseRealTime();
#endif
#if defined(THETA_QUEUE_HAS_THETA_MUTEX)
BENCHMARK_TEMPLATE(BM_Generic1TRoundtrip, ThetaMutexGenericAdaptor<MoveOnly32>);
BENCHMARK_TEMPLATE(BM_GenericMPMC4x4_Try, ThetaMutexGenericAdaptor<MoveOnly32>)
    ->UseRealTime();
#endif

// ---------------------------------------------------------------------------
// Group 7: First-Class Bulk (`try_push_bulk`) & Zero-Copy `drain` (`Flusher`)
// ---------------------------------------------------------------------------

void BM_SPSC_BulkDrain(  // NOLINT(readability-identifier-naming)
    benchmark::State& state) {
  constexpr std::size_t kChunk = 64;
  SPSCQueue<uint64_t, 2048> queue{};
  std::barrier ready(2);
  std::barrier start_batch(2);
  std::barrier end_batch(2);
  std::atomic<bool> stop{false};

  std::thread consumer([&]() {
    pin_thread_to_core(1, 2);
    ready.arrive_and_wait();
    while (true) {
      start_batch.arrive_and_wait();
      if (stop.load(std::memory_order_relaxed)) {
        return;
      }
      std::size_t popped = 0;
      uint64_t sink = 0;
      while (popped < kBatchPerProducer) {
        const std::size_t n =
            queue.drain([&](uint64_t v) noexcept { sink += v; });
        if (n == 0U) {
          cpu_relax();
        } else {
          popped += n;
        }
      }
      benchmark::DoNotOptimize(sink);
      end_batch.arrive_and_wait();
    }
  });

  std::array<uint64_t, kChunk> batch{};
  batch.fill(1U);
  pin_thread_to_core(0, 2);
  ready.arrive_and_wait();

  for (auto state_token : state) {
    benchmark::DoNotOptimize(state_token);
    start_batch.arrive_and_wait();
    std::size_t pushed = 0;
    while (pushed < kBatchPerProducer) {
      const std::size_t rem =
          std::min<std::size_t>(kChunk, kBatchPerProducer - pushed);
      const std::size_t n =
          queue.try_push_bulk(std::span<const uint64_t>{batch.data(), rem});
      if (n == 0U) {
        cpu_relax();
      } else {
        pushed += n;
      }
    }
    end_batch.arrive_and_wait();
  }

  stop.store(true, std::memory_order_relaxed);
  start_batch.arrive_and_wait();
  consumer.join();
  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(kBatchPerProducer));
}

BENCHMARK(BM_SPSC_BulkDrain)->UseRealTime();

// ---------------------------------------------------------------------------
// Group 8: Tail-Latency Percentile Distributions (Coordinated-Omission Safe)
//   Reports p50 / p99 / p99.9 / p99.99 / max in nanoseconds via UserCounters.
// ---------------------------------------------------------------------------

inline uint64_t steady_now_ns() noexcept {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

inline void record_percentiles(benchmark::State& state, const char* prefix,
                               std::vector<uint32_t>& samples) {
  if (samples.empty()) {
    return;
  }
  std::ranges::sort(samples);
  const auto pct = [&](double q) -> double {
    const auto idx = static_cast<std::size_t>(
        std::min<double>(static_cast<double>(samples.size() - 1U),
                         q * static_cast<double>(samples.size())));
    return static_cast<double>(samples[idx]);
  };
  std::array<char, 32> key{};
  const auto set_k = [&](const char* suffix, double val) {
    std::snprintf(key.data(), key.size(), "%s%s", prefix, suffix);
    state.counters[key.data()] = val;
  };
  set_k("p50_ns", pct(0.50));
  set_k("p99_ns", pct(0.99));
  set_k("p999_ns", pct(0.999));
  set_k("p9999_ns", pct(0.9999));
  set_k("max_ns", static_cast<double>(samples.back()));
}

template <NonBlockingQueueAdaptor QType>
void BM_TailLatency_SPSC_Paced(  // NOLINT(readability-identifier-naming)
    benchmark::State& state) {
  constexpr std::size_t kBatch = 2048;
  constexpr uint64_t kIntervalNs = 250;  // 4M msg/s open-loop target
  QType queue{};
  std::vector<uint64_t> sched_times(kBatch, 0U);
  std::vector<int> slots(kBatch, 0);
  std::vector<uint32_t> all_e2e;
  all_e2e.reserve(65536);

  std::barrier ready(2);
  std::barrier start_batch(2);
  std::barrier end_batch(2);
  std::atomic<bool> stop{false};

  std::thread consumer([&]() {
    pin_thread_to_core(1, 2);
    std::vector<uint32_t> batch_lat(kBatch, 0U);
    ready.arrive_and_wait();
    while (true) {
      start_batch.arrive_and_wait();
      if (stop.load(std::memory_order_relaxed)) {
        return;
      }
      for (std::size_t i = 0; i < kBatch; ++i) {
        const int* msg = nullptr;
        while (true) {
          if (auto res = queue.try_pop(); res.has_value()) {
            msg = *res;
            break;
          }
          cpu_relax();
        }
        const uint64_t rx = steady_now_ns();
        const auto idx = static_cast<std::size_t>(msg - slots.data());
        const uint64_t tx = sched_times[idx];
        batch_lat[i] = static_cast<uint32_t>(rx > tx ? (rx - tx) : 0U);
      }
      all_e2e.insert(all_e2e.end(), batch_lat.begin(), batch_lat.end());
      end_batch.arrive_and_wait();
    }
  });

  pin_thread_to_core(0, 2);
  ready.arrive_and_wait();

  for (auto tok : state) {
    benchmark::DoNotOptimize(tok);
    start_batch.arrive_and_wait();
    const uint64_t b0 = steady_now_ns() + 2'000U;
    for (std::size_t i = 0; i < kBatch; ++i) {
      const uint64_t sched = b0 + i * kIntervalNs;
      sched_times[i] = sched;
      while (steady_now_ns() < sched) {
        cpu_relax();
      }
      while (!queue.try_push(&slots[i])) {
        cpu_relax();
      }
    }
    end_batch.arrive_and_wait();
  }

  stop.store(true, std::memory_order_relaxed);
  start_batch.arrive_and_wait();
  consumer.join();

  record_percentiles(state, "e2e_", all_e2e);
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kBatch));
}

template <NonBlockingQueueAdaptor QType>
void BM_TailLatency_MPSC_NoisyNeighbor(  // NOLINT(readability-identifier-naming)
    benchmark::State& state) {
  constexpr std::size_t kProbesPerIter = 1024;
  constexpr uint64_t kProbeSpacingNs = 1000;  // 1M probes/s alongside 3 floods
  QType queue{};
  int flood_sentinel = -1;
  std::vector<int> probe_slots(kProbesPerIter, 0);
  std::vector<uint64_t> probe_sched(kProbesPerIter, 0U);
  std::vector<uint32_t> push_samples;
  std::vector<uint32_t> e2e_samples;
  push_samples.reserve(32768);
  e2e_samples.reserve(32768);

  std::barrier ready(5);
  std::barrier start_batch(5);
  std::barrier end_batch(5);
  std::atomic<bool> batch_done{false};
  std::atomic<bool> stop{false};

  std::vector<std::thread> workers;
  workers.reserve(4);
  for (int f = 0; f < 3; ++f) {
    workers.emplace_back([&, f]() {
      pin_thread_to_core(f + 1, 5);
      ready.arrive_and_wait();
      while (true) {
        start_batch.arrive_and_wait();
        if (stop.load(std::memory_order_relaxed)) {
          return;
        }
        while (!batch_done.load(std::memory_order_relaxed)) {
          for (int k = 0; k < 16; ++k) {
            (void)queue.try_push(&flood_sentinel);
          }
          cpu_relax();
        }
        end_batch.arrive_and_wait();
      }
    });
  }

  workers.emplace_back([&]() {
    pin_thread_to_core(4, 5);
    std::vector<uint32_t> batch_e2e;
    batch_e2e.reserve(kProbesPerIter);
    ready.arrive_and_wait();
    while (true) {
      start_batch.arrive_and_wait();
      if (stop.load(std::memory_order_relaxed)) {
        return;
      }
      batch_e2e.clear();
      while (batch_e2e.size() < kProbesPerIter) {
        if (auto res = queue.try_pop(); res.has_value()) {
          const int* ptr = *res;
          if (ptr != &flood_sentinel) {
            const uint64_t rx = steady_now_ns();
            const auto idx = static_cast<std::size_t>(ptr - probe_slots.data());
            const uint64_t tx = probe_sched[idx];
            batch_e2e.push_back(
                static_cast<uint32_t>(rx > tx ? (rx - tx) : 0U));
          }
        } else {
          cpu_relax();
        }
      }
      batch_done.store(true, std::memory_order_release);
      // Drain remaining flood items after batch ends
      end_batch.arrive_and_wait();
      while (queue.try_pop().has_value()) {
      }
      e2e_samples.insert(e2e_samples.end(), batch_e2e.begin(), batch_e2e.end());
    }
  });

  pin_thread_to_core(0, 5);
  ready.arrive_and_wait();

  for (auto tok : state) {
    benchmark::DoNotOptimize(tok);
    batch_done.store(false, std::memory_order_relaxed);
    start_batch.arrive_and_wait();
    const uint64_t b0 = steady_now_ns() + 2'000U;
    for (std::size_t i = 0; i < kProbesPerIter; ++i) {
      const uint64_t sched = b0 + i * kProbeSpacingNs;
      probe_sched[i] = sched;
      while (steady_now_ns() < sched) {
        cpu_relax();
      }
      const uint64_t t0 = steady_now_ns();
      while (!queue.try_push(&probe_slots[i])) {
        cpu_relax();
      }
      push_samples.push_back(static_cast<uint32_t>(steady_now_ns() - t0));
    }
    end_batch.arrive_and_wait();
  }

  stop.store(true, std::memory_order_relaxed);
  start_batch.arrive_and_wait();
  for (auto& w : workers) {
    w.join();
  }

  record_percentiles(state, "push_", push_samples);
  record_percentiles(state, "e2e_", e2e_samples);
  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(kProbesPerIter));
}

BENCHMARK_TEMPLATE(BM_TailLatency_SPSC_Paced, ThetaSPSCStaticAdaptor)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_TailLatency_SPSC_Paced, ThetaUSPSCAdaptor)->UseRealTime();
#if defined(THETA_QUEUE_HAS_FOLLY)
BENCHMARK_TEMPLATE(BM_TailLatency_SPSC_Paced, FollySPSCAdaptor)->UseRealTime();
BENCHMARK_TEMPLATE(BM_TailLatency_SPSC_Paced, FollyUSPSCAdaptor)->UseRealTime();
#endif

BENCHMARK_TEMPLATE(BM_TailLatency_MPSC_NoisyNeighbor, ThetaMPSCAdaptor)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_TailLatency_MPSC_NoisyNeighbor, ThetaFastMPSCAdaptor)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_TailLatency_MPSC_NoisyNeighbor, ThetaUMPSCAdaptor)
    ->UseRealTime();
#if defined(THETA_QUEUE_HAS_FOLLY)
BENCHMARK_TEMPLATE(BM_TailLatency_MPSC_NoisyNeighbor, FollyMPMCAdaptor)
    ->UseRealTime();
#endif
#if defined(THETA_QUEUE_HAS_ATOMIC_QUEUE)
BENCHMARK_TEMPLATE(BM_TailLatency_MPSC_NoisyNeighbor, AtomicQueueAdaptor)
    ->UseRealTime();
#endif

// ---------------------------------------------------------------------------
// Group 9: Open-Loop / Under-Saturated 6-Thread RPC Service Pipeline
//   Mirrors `theta-app-bench` `service_sla_bench`:
//   2 ingress producers -> StorageQueue (3 consumers, ~1.8 us simulated DB Get)
//                       +  AuditQueue   (1 consumer,  ~0.6 us simulated LZ4)
//   `state.range(0)` = target kRPS (e.g. 260, 365, or 0 for max saturation).
// ---------------------------------------------------------------------------
template <BlockingQueueAdaptor StorageQ, BlockingQueueAdaptor AuditQ>
void BM_TailLatency_OpenLoopRPC_6T(  // NOLINT(readability-identifier-naming)
    benchmark::State& state) {
  constexpr int kIngress = 2;
  constexpr int kStorage = 3;
  constexpr int kReqsPerIngress = 512;
  constexpr int kTotalReqs = kIngress * kReqsPerIngress;
  constexpr int kTotalCompletions = kTotalReqs * 2;

  const auto target_krps = static_cast<uint64_t>(state.range(0));
  const uint64_t interval_ns =
      target_krps > 0U
          ? ((1'000'000ULL * static_cast<uint64_t>(kIngress)) / target_krps)
          : 0ULL;

  StorageQ storage_q{};
  AuditQ audit_q{};

  std::vector<int> req_tokens(static_cast<std::size_t>(kTotalReqs), 0);
  std::vector<uint64_t> sched_ns_vec(static_cast<std::size_t>(kTotalReqs), 0U);
  std::vector<uint32_t> iter_lat_ns(static_cast<std::size_t>(kTotalReqs), 0U);
  std::vector<uint32_t> all_e2e_ns;
  all_e2e_ns.reserve(65536);
  int stop_token = -1;

  std::mutex epoch_mu;
  std::condition_variable start_cv;
  std::condition_variable done_cv;
  uint64_t epoch = 0;
  uint64_t burst_t0 = 0;
  std::atomic<int> completions{0};
  std::atomic<uint64_t> sink{0};
  bool shutdown = false;

  auto busy_work_ns = [](uint64_t ns, uint64_t seed) noexcept -> uint64_t {
    const uint64_t deadline = steady_now_ns() + ns;
    uint64_t x = seed ^ 0x9E3779B97F4A7C15ULL;
    do {
      for (int k = 0; k < 16; ++k) {
        x ^= x >> 12U;
        x ^= x << 25U;
        x ^= x >> 27U;
        x *= 0x2545F4914F6CDD1DULL;
      }
    } while (steady_now_ns() < deadline);
    return x;
  };

  std::vector<std::thread> ingress;
  ingress.reserve(kIngress);
  for (int ing = 0; ing < kIngress; ++ing) {
    ingress.emplace_back([&, ing]() {
      uint64_t seen = 0;
      for (;;) {
        uint64_t t0 = 0;
        {
          std::unique_lock<std::mutex> lk(epoch_mu);
          start_cv.wait(lk, [&] { return shutdown || epoch > seen; });
          if (shutdown) {
            return;
          }
          seen = epoch;
          t0 = burst_t0;
        }
        for (int i = 0; i < kReqsPerIngress; ++i) {
          const auto slot = static_cast<std::size_t>(ing * kReqsPerIngress + i);
          const uint64_t sched =
              interval_ns > 0U
                  ? (t0 + static_cast<uint64_t>(i) * interval_ns +
                     static_cast<uint64_t>(ing) * (interval_ns / 2U))
                  : steady_now_ns();
          if (interval_ns > 0U) {
            while (steady_now_ns() < sched) {
              cpu_relax();
            }
          } else {
            sink.fetch_add(busy_work_ns(1'600U, slot),
                           std::memory_order_relaxed);
          }
          sched_ns_vec[slot] = sched;
          req_tokens[slot] = static_cast<int>(slot);
          storage_q.push(&req_tokens[slot]);
          audit_q.push(&req_tokens[slot]);
        }
      }
    });
  }

  std::vector<std::thread> storage;
  storage.reserve(kStorage);
  for (int w = 0; w < kStorage; ++w) {
    storage.emplace_back([&]() {
      for (;;) {
        const int* tok = storage_q.pop();
        if (tok == &stop_token) {
          return;
        }
        const auto slot = static_cast<std::size_t>(*tok);
        sink.fetch_add(busy_work_ns(1'800U, slot), std::memory_order_relaxed);
        const uint64_t done = steady_now_ns();
        const uint64_t tx = sched_ns_vec[slot];
        iter_lat_ns[slot] = static_cast<uint32_t>(done > tx ? (done - tx) : 1U);
        if (completions.fetch_add(1, std::memory_order_acq_rel) + 1 ==
            kTotalCompletions) {
          const std::lock_guard<std::mutex> lk(epoch_mu);
          done_cv.notify_one();
        }
      }
    });
  }

  std::thread audit([&]() {
    for (;;) {
      const int* tok = audit_q.pop();
      if (tok == &stop_token) {
        return;
      }
      sink.fetch_add(busy_work_ns(600U, static_cast<uint64_t>(*tok)),
                     std::memory_order_relaxed);
      if (completions.fetch_add(1, std::memory_order_acq_rel) + 1 ==
          kTotalCompletions) {
        const std::lock_guard<std::mutex> lk(epoch_mu);
        done_cv.notify_one();
      }
    }
  });

  for (auto state_tok : state) {
    benchmark::DoNotOptimize(state_tok);
    completions.store(0, std::memory_order_relaxed);
    {
      const std::lock_guard<std::mutex> lk(epoch_mu);
      burst_t0 = steady_now_ns();
      ++epoch;
    }
    start_cv.notify_all();
    {
      std::unique_lock<std::mutex> lk(epoch_mu);
      done_cv.wait(lk, [&] {
        return completions.load(std::memory_order_acquire) == kTotalCompletions;
      });
    }
    if (all_e2e_ns.size() + kTotalReqs <= 131072U) {
      all_e2e_ns.insert(all_e2e_ns.end(), iter_lat_ns.begin(),
                        iter_lat_ns.end());
    }
  }

  {
    const std::lock_guard<std::mutex> lk(epoch_mu);
    shutdown = true;
  }
  start_cv.notify_all();
  for (auto& t : ingress) {
    t.join();
  }
  for (int w = 0; w < kStorage; ++w) {
    storage_q.push(&stop_token);
  }
  audit_q.push(&stop_token);
  for (auto& t : storage) {
    t.join();
  }
  audit.join();

  benchmark::DoNotOptimize(sink.load(std::memory_order_relaxed));
  record_percentiles(state, "e2e_", all_e2e_ns);
  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(kTotalReqs));
}

#define THETA_OPENLOOP_RPC_ARGS ->Arg(260)->Arg(365)->Arg(0)->UseRealTime()

BENCHMARK_TEMPLATE(BM_TailLatency_OpenLoopRPC_6T, ThetaMPMCStaticAdaptor,
                   ThetaMPSCAdaptor)
THETA_OPENLOOP_RPC_ARGS;
BENCHMARK_TEMPLATE(BM_TailLatency_OpenLoopRPC_6T, ThetaFastMPMCAdaptor,
                   ThetaFastMPSCAdaptor)
THETA_OPENLOOP_RPC_ARGS;
#if defined(THETA_QUEUE_HAS_FOLLY)
BENCHMARK_TEMPLATE(BM_TailLatency_OpenLoopRPC_6T, FollyMPMCAdaptor,
                   FollyMPMCAdaptor)
THETA_OPENLOOP_RPC_ARGS;
#endif
#if defined(THETA_QUEUE_HAS_ABSL)
BENCHMARK_TEMPLATE(BM_TailLatency_OpenLoopRPC_6T, AbslMutexRingAdaptor,
                   AbslMutexRingAdaptor)
THETA_OPENLOOP_RPC_ARGS;
#endif

}  // namespace
}  // namespace theta

BENCHMARK_MAIN();
