#pragma once

#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "concepts.h"
#include "queue_opts.h"

namespace theta {

template <AtomType T, size_t kBufferSize = 128>
class MPMCQueue {
  struct Tag {
    static_assert((kBufferSize & (kBufferSize - 1)) == 0, "");

    // An kIncrement value of
    //   kIncrement = 1 + hardware_destructive_interference_size / sizeof(T)
    // would make it so that adjacent values will always be on separate cache
    // lines. However, benchmarks don't show that this attempt to avoiding
    // false sharing helps.
    static constexpr uint64_t kIncrement = 1;
    // static constexpr uint64_t kIncrement =
    //     1 + hardware_destructive_interference_size / sizeof(T);
    static constexpr uint64_t kBufferWrapDelta = kBufferSize * kIncrement;
    static constexpr uint64_t kBufferSizeMask = kBufferSize - 1;
    static constexpr uint64_t kConsumerFlag = (1ULL << 63);
    static constexpr uint64_t kWaitingFlag = (1ULL << 62);

    uint64_t raw;

    Tag(uint64_t raw) : raw(raw) {}
    Tag() : Tag(0) {}

    Tag& operator++() {
      raw += kIncrement;
      return *this;
    }

    Tag& operator++(int) {
      raw += kIncrement;
      return *this;
    }

    Tag& operator--() {
      raw -= kIncrement;
      return *this;
    }

    Tag& operator--(int) {
      raw -= kIncrement;
      return *this;
    }

    auto operator<=>(const Tag&) const = default;

    std::string DebugString() const {
      return "Tag<" + std::string(is_producer() ? "P" : "C")
           + (is_waiting() ? std::string("|W") : "") + ">{"
           + std::to_string(value()) + "@" + std::to_string(to_index()) + "}";
    }

    uint64_t value() const { return (raw << 2) >> 2; }

    Tag prev_paired_tag() const {
      if (is_consumer()) {
        return Tag{(raw ^ kConsumerFlag) & ~kWaitingFlag};
      } else {
        return Tag{((raw - kBufferWrapDelta) ^ kConsumerFlag) & ~kWaitingFlag};
      }
    }

    bool is_paired(Tag observed_tag) const {
      return prev_paired_tag().raw == (observed_tag.raw & ~kWaitingFlag);
    }

    bool is_producer() const { return (raw & kConsumerFlag) == 0; }

    void mark_as_producer() { raw &= ~kConsumerFlag; }

    bool is_consumer() const { return (raw & kConsumerFlag) > 0; }

    void mark_as_consumer() { raw |= kConsumerFlag; }

    void mark_as_waiting() { raw |= kWaitingFlag; }

    bool is_waiting() const { return (raw & kWaitingFlag) > 0; }

    void clear_waiting_flag() { raw &= ~kWaitingFlag; }

    int to_index() const { return raw & kBufferSizeMask; }
  };
  static_assert(sizeof(Tag) == sizeof(uint64_t), "");

  union Index {
    struct {
      Tag tag;
    };
    struct {
      std::atomic<Tag> tag_atomic;
    };
    struct {
      std::atomic<uint64_t> tag_raw_atomic;
    };

    Index(Tag tag) : tag(tag) {}
    Index() : Index(/*tag=*/0) {}

    std::string DebugString() const {
      return "Index{tag=" + tag.DebugString() + "}";
    }
  };
  static_assert(sizeof(Index) == sizeof(Tag), "");

  union Data {
    struct {
      T value;
      Tag tag;
    };
    struct {
      std::atomic<T> value_atomic;
      std::atomic<Tag> tag_atomic;
    };
    std::atomic<__int128> line;

    Data(T value, Tag tag) : value(value), tag(tag) {}
    Data(__int128 line) : line(line) {}
    Data() : Data(/*line=*/0) {}
    Data(const Data& other)
        : Data(other.line.load(std::memory_order::relaxed)) {}
    Data& operator=(const Data& other) {
      line.store(other.line.load(std::memory_order::relaxed),
                 std::memory_order::relaxed);
      return *this;
    }

    std::string DebugString() const {
      return "Data{value=" + std::string(bool(value) ? "####" : "null") + ", "
           + tag.DebugString() + "}";
    }
  };
  static_assert(sizeof(Data) == sizeof(Data::line), "");
  static_assert(sizeof(Data) == 16, "");

 public:
  MPMCQueue()
      : head_(Tag::kBufferWrapDelta)
      , tail_(Tag::kBufferWrapDelta)
      , buffer_(kBufferSize) {
    Tag tag;
    tag.mark_as_consumer();
    for (size_t i = 0; i < buffer_.size(); i++) {
      buffer_[tag.to_index()].tag = tag;
      ++tag;
    }
    std::atomic_thread_fence(std::memory_order::release);
  }
  MPMCQueue(const QueueOpts&) : MPMCQueue() {}

  ~MPMCQueue() {
    while (true) {
      auto v = try_pop();
      if (!v) {
        break;
      }
    }
  }

  void push(T val) {
    Tag tail{tail_.tag_raw_atomic.fetch_add(Tag::kIncrement,
                                            std::memory_order::acq_rel)};
    do_push(std::move(val), tail);
  }

  bool try_push(T val) {
    const Tag head{head_.tag_atomic.load(std::memory_order::acquire)};

    Tag expected_tail{head};
    Tag desired_tail{expected_tail};
    desired_tail++;

    while (
        !tail_.tag_atomic.compare_exchange_weak(expected_tail,
                                                desired_tail,
                                                std::memory_order::release,
                                                std::memory_order::relaxed)) {
      desired_tail = expected_tail;
      desired_tail++;
      if (desired_tail.raw >= head.raw + Tag::kBufferWrapDelta) {
        return false;
      }
    }

    do_push(std::move(val), expected_tail);
    return true;
  }

  T pop() {
    Tag tag{/*raw=*/head_.tag_raw_atomic.fetch_add(Tag::kIncrement,
                                                   std::memory_order::acq_rel)};
    tag.mark_as_consumer();
    return do_pop(tag);
  }

  std::optional<T> try_pop() {
    const Tag tail{tail_.tag_atomic.load(std::memory_order::acquire)};

    Tag desired_head{tail.raw};
    Tag expected_head{desired_head.raw - Tag::kIncrement};

    while (
        !head_.tag_atomic.compare_exchange_weak(expected_head,
                                                desired_head,
                                                std::memory_order::release,
                                                std::memory_order::relaxed)) {
      desired_head = expected_head;
      desired_head++;
      if (desired_head > tail) {
        return {};
      }
    }

    expected_head.mark_as_consumer();
    return do_pop(expected_head);
  }

  size_t size() const {
    // Reading head before tail will make it possible to "see" more elements in
    // the queue than it can hold, but this makes it so that the size will
    // never be negative.
    auto head = head_.tag_atomic.load(std::memory_order::acquire);
    auto tail = tail_.tag_atomic.load(std::memory_order::acquire);

    return (tail.raw - head.raw) / Tag::kIncrement;
  }

  static constexpr size_t capacity() { return kBufferSize; }

 private:
  alignas(hardware_destructive_interference_size) Index head_;
  alignas(hardware_destructive_interference_size) Index tail_;
  alignas(hardware_destructive_interference_size) std::vector<Data> buffer_;

  void do_push(T val, const Tag& tag) {
    assert(tag.is_producer());
    assert(!tag.is_waiting());

    int idx = tag.to_index();

    // This is the strangest issue -- with Ubuntu clang version 15.0.7,
    // when observed_data is defined inside of the loop scope, benchmarks will
    // slow down by over 5x.
    Data observed_data;
    while (true) {
      __int128 observed_data_line
          = buffer_[idx].line.load(std::memory_order::acquire);
      observed_data = Data{/*line=*/observed_data_line};

      if (tag.is_paired(observed_data.tag)) {
        break;
      }

      wait_for_data(tag, observed_data.tag);
    }

    Data new_data{/*value_=*/val, /*tag_=*/tag};
    Data old_data{buffer_[idx].line.exchange(
        new_data.line.load(std::memory_order::relaxed),
        std::memory_order::acq_rel)};
    if (old_data.tag.is_waiting()) {
      buffer_[idx].tag_atomic.notify_all();
    }
  }

  T do_pop(const Tag& tag) {
    assert(tag.is_consumer());
    assert(!tag.is_waiting());

    int idx = tag.to_index();

    Data observed_data;
    while (true) {
      observed_data
          = Data{/*line=*/buffer_[idx].line.load(std::memory_order::acquire)};

      if (tag.is_paired(observed_data.tag)) {
        break;
      }

      wait_for_data(tag, observed_data.tag);
    }

    // This is another strange issue -- it is faster to exchange the __int128
    // value backing the Data object instead of just exchanging the 8 byte tag
    // value inside of that Data object.
    Data old_data{buffer_[idx].line.exchange(
        Data{/*value=*/T{}, /*tag=*/tag}.line.load(std::memory_order::relaxed),
        std::memory_order::acq_rel)};

    if (old_data.tag.is_waiting()) {
      buffer_[idx].tag_atomic.notify_all();
    }

    return observed_data.value;
  }

  void wait_for_data(const Tag& claimed_tag, Tag observed_tag) {
    int idx = claimed_tag.to_index();
    while (true) {
      Tag want_tag{observed_tag};
      want_tag.mark_as_waiting();
      if ((observed_tag == want_tag)
          || buffer_[idx].tag_atomic.compare_exchange_weak(
              observed_tag,
              want_tag,
              std::memory_order::release,
              std::memory_order::relaxed)) {
        buffer_[idx].tag_atomic.wait(want_tag, std::memory_order::acquire);
        break;
      }

      if (claimed_tag.is_paired(observed_tag)) {
        break;
      }
    }
  }
};
}  // namespace theta
