enable_testing()

add_executable(
  queue_test
  queue_test.cc
)
target_link_libraries(
  queue_test
  mpmc-queue
  mpsc-queue
  GTest::gmock
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(queue_test)
