add_executable(
  queue_bench
  queue_bench.cc
)
target_link_libraries(
  queue_bench
  mpmc-queue
  mpsc-queue
  benchmark::benchmark
)
target_include_directories(
  queue_bench PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
