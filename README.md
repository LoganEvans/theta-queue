<!-- FILENAME: README.md -->

# theta-queue

Header-only C++23 bounded and unbounded lock-free concurrent queues in `namespace theta`.

## Queue Variants

Include `<theta/queue.hpp>` (or individual headers under `<theta/queue/...>`):

### Bounded Queues
- **`theta::SPSCQueue<T, kStaticBufferSize = 0>`** (`<theta/queue/spsc-queue.h>`):
  Pre-wrapped `[0 .. N-1]` index ring (`capacity() == N - 1`), linear slot layout for L1D write-combining, 128B-isolated producer/consumer cache lines with local head/tail shadows, frameless `asm volatile goto` asymmetric wakeup, and bulk `try_push_bulk` / `try_pop_bulk` / zero-copy `drain`.
- **`theta::MPSCQueue<T, kStaticBufferSize = 0, kOrder = QueueOrder::StrictFIFO>`** (`<theta/queue/mpsc-queue.h>`):
  - `QueueOrder::StrictFIFO`: `HybridFaaRing` wait-free `LOCK XADD` interior fast path + exact boundary `CAS`, 128B page-local slot bouncing, contention-gated `PREFETCHW`, and zero-RMW consumer `head_`.
  - **`theta::FastMPSCQueue<T, kStaticBufferSize = 0>`** (`QueueOrder::PerProducer`): 2-Tier Linux `rseq` + 7-slot SSE2 CPU CAM lanes with sticky zero-RMW consumer draining and automatic `MEMBARRIER_RSEQ` CPU-migration recovery.
- **`theta::MPMCQueue<T, kStaticBufferSize = 0, kOrder = QueueOrder::StrictFIFO>`** (`<theta/queue/mpmc-queue.h>`):
  - `QueueOrder::StrictFIFO`: `HybridFaaRing` bounded MPMC ring with hybrid fused/separated `{turn, T}` storage, 128B page-local slot bouncing, `stlf_micro_pause` SMT-friendly backoff, and 16-way ticket-striped `StripedParkLot` (`MEMBARRIER_CMD_PRIVATE_EXPEDITED`).
  - **`theta::FastMPMCQueue<T, kStaticBufferSize = 0>`** (`QueueOrder::PerProducer`): Dual 2-Tier Linux `rseq` + 7-slot SSE2 CPU CAM lanes with one-shot consumer work-stealing promotion.

### Unbounded Queues (`ElasticSlab`)
- **`theta::UnboundedQueue<T, kSingleProducer = false, kSingleConsumer = false, kLgSlab = 9>`** (`<theta/queue/unbounded-queue.h>`) and aliases:
  - **`theta::USPSCQueue<T, kLgSlab = 9>`**: Circular in-place 128B-isolated single-slab fast path when `depth < 2^kLgSlab`, seamlessly chaining and recycling flat slabs lock-free on deep bursts.
  - **`theta::UMPSCQueue<T, kLgSlab = 9>`**: Multi-producer `fetch_add` tickets over bounced flat slabs with a zero-RMW single-consumer reader and type-stable slab pool.
  - **`theta::UMPMCQueue<T, kLgSlab = 9>`**: Multi-producer, multi-consumer elastic slabs with type-stable pooled slab recycling (zero per-operation Hazard Pointers).

All queues support first-class batch APIs (`try_push_bulk`, `try_pop_bulk`, and zero-copy `drain`) and accept any `theta::QueueElement` type—including pointers, scalars, large or over-aligned SIMD structs, and move-only objects (`std::unique_ptr`, `std::string`, non-default-constructible tasks).

## Build & Verification

```bash
nix develop
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARK=ON
cmake --build build
ctest --test-dir build --output-on-failure
```
