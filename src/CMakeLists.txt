add_library(
  mpmc-queue INTERFACE
  mpmc_queue.h
)
target_include_directories(mpmc-queue INTERFACE .)
target_link_libraries(mpmc-queue INTERFACE atomic)

add_library(
  mpsc-queue INTERFACE
  mpsc_queue.h
)
target_include_directories(mpsc-queue INTERFACE .)
target_link_libraries(mpmc-queue INTERFACE atomic)
