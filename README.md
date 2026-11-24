<!-- FILENAME: README.md -->

# theta-queue

High-performance C++23 bounded and unbounded lock-free concurrent queues in `namespace theta` (`include/theta/queue.hpp` + `include/theta/queue/*.hpp`, with cold OS `futex`/`membarrier` and `rseq` CAM paths compiled into `libtheta-queue.a`).

## Queue Variants

Include `<theta/queue.hpp>` (or individual headers under `<theta/queue/*.hpp>`):

### Bounded Queues
- **`theta::SPSCQueue<T, kStaticBufferSize = 0>`** (`<theta/queue/spsc_queue.hpp>`):
  Pre-wrapped `[0 .. N-1]` index ring (`capacity() == N - 1`), linear slot layout for L1D write-combining, 128B-isolated producer/consumer cache lines with local head/tail shadows, frameless 2-tier asymmetric wakeup, and bulk `try_push_bulk` / `try_pop_bulk` / zero-copy `drain`.
- **`theta::MPSCQueue<T, kStaticBufferSize = 0, kOrder = QueueOrder::StrictFIFO>`** (`<theta/queue/mpsc_queue.hpp>`):
  - `QueueOrder::StrictFIFO`: Wait-free `LOCK XADD` blocking `push()`, strictly lock-free Vyukov `CAS` `try_push()` (with up to 64-slot single-`CAS` batching in `try_push_bulk()`), 128B page-local slot bouncing, contention-gated `PREFETCHW`, and zero-RMW consumer `head_`.
  - **`theta::FastMPSCQueue<T, kStaticBufferSize = 0>`** (`QueueOrder::PerProducer`): 2-Tier Linux `rseq` + 7-slot CPU CAM lanes (SSE2/NEON-free x86-64 SIMD & AArch64 GPR SWAR CAM), inline word commits for `<= 8B` scalars and out-of-`rseq` turn-committed `RichLanePool<T>` for arbitrary/move-only payloads, deficit round-robin (`kRseqScQuantum = 64`) single-consumer draining, and automatic `MEMBARRIER_RSEQ` CPU-migration recovery.
- **`theta::MPMCQueue<T, kStaticBufferSize = 0, kOrder = QueueOrder::StrictFIFO>`** (`<theta/queue/mpmc_queue.hpp>`):
  - `QueueOrder::StrictFIFO`: Wait-free `LOCK XADD` blocking `push()`/`pop()`, strictly lock-free Vyukov `CAS` `try_push()`/`try_pop()` (plus up to 64-slot single-`CAS` batching in `try_push_bulk()`/`try_pop_bulk()`), hybrid fused/separated `{turn, T}` storage, 128B page-local slot bouncing, `stlf_micro_pause` SMT-friendly backoff, and 16-way ticket-striped `StripedParkLot` (`MEMBARRIER_CMD_PRIVATE_EXPEDITED`).
  - **`theta::FastMPMCQueue<T, kStaticBufferSize = 0>`** (`QueueOrder::PerProducer`): Dual 2-Tier Linux `rseq` + 7-slot SIMD/SWAR CPU CAM lanes on x86-64 and AArch64 across both scalar and arbitrary/move-only payloads (`RichLanePool<T>`), with one-shot consumer work-stealing promotion.

### Unbounded Queues (`ElasticSlab`)
- **`theta::UnboundedQueue<T, kSingleProducer = false, kSingleConsumer = false, kLgSlab = 9>`** (`<theta/queue/unbounded_queue.hpp>`) and aliases:
  - **`theta::USPSCQueue<T, kLgSlab = 9>`**: Circular in-place 128B-isolated single-slab fast path when `depth < 2^kLgSlab`, seamlessly chaining and recycling flat slabs lock-free on deep bursts.
  - **`theta::UMPSCQueue<T, kLgSlab = 9>`**: Multi-producer `fetch_add` tickets over bounced flat slabs with a zero-RMW single-consumer reader and type-stable slab pool.
  - **`theta::UMPMCQueue<T, kLgSlab = 9>`**: Multi-producer, multi-consumer elastic slabs with type-stable pooled slab recycling (zero per-operation Hazard Pointers).

All queues support first-class batch APIs (`try_push_bulk`, `try_pop_bulk`, and zero-copy `drain`) and accept any `theta::QueueElement` type—including pointers, scalars, large or over-aligned SIMD structs, and move-only objects (`std::unique_ptr`, `std::string`, non-default-constructible tasks).

## Build & Verification

```bash
nix develop
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```
