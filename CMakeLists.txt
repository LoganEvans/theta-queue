cmake_minimum_required(VERSION 3.14)
SET (CMAKE_C_COMPILER "/usr/bin/clang" CACHE STRING "C compiler" FORCE)
SET (CMAKE_CXX_COMPILER "/usr/bin/clang++" CACHE STRING "C++ compiler" FORCE)
project(Queue)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  googlebench
  URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
)
FetchContent_MakeAvailable(googlebench)

add_subdirectory(bench)
add_subdirectory(src)
add_subdirectory(test)
