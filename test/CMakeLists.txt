find_package(GTest REQUIRED)
find_package(debug-utils CONFIG REQUIRED)
include(GoogleTest)

add_executable(queue-test queue-test.cc)
target_link_libraries(
  queue-test PUBLIC GTest::gmock GTest::gtest_main theta::debug-utils
                    theta::stacktrace-signal-handlers mpmc-queue mpsc-queue)
gtest_discover_tests(queue-test)

install(
  TARGETS queue-test
  EXPORT ${PROJECT_NAME}
  DESTINATION $ENV{out}/bin/test)
