find_package(benchmark REQUIRED)

add_executable(queue-benchmark queue-benchmark.cc)
target_link_libraries(queue-benchmark mpmc-queue mpsc-queue
                      benchmark::benchmark max0x7ba::atomic_queue)

install(
  TARGETS queue-benchmark
  EXPORT ${PROJECT_NAME}
  DESTINATION $ENV{out}/bin/benchmark)
