cmake_minimum_required(VERSION 3.31)
project(queue VERSION 0.1)
include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-fdiagnostics-color=always
                    -Wno-unused-command-line-argument)
add_link_options(-Wno-unused-command-line-argument)

find_package(fmt CONFIG REQUIRED)

add_library(mpmc-queue INTERFACE
            ${CMAKE_CURRENT_LIST_DIR}/include/theta/queue/mpmc-queue.h)
target_include_directories(
  mpmc-queue INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include/>)
target_link_libraries(mpmc-queue INTERFACE atomic)

add_library(mpsc-queue INTERFACE
            ${CMAKE_CURRENT_LIST_DIR}/include/theta/queue/mpsc-queue.h)
target_include_directories(
  mpsc-queue INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include/>)
target_link_libraries(mpmc-queue INTERFACE atomic)

if(BUILD_TESTING)
  add_subdirectory(test)
endif(BUILD_TESTING)

if($ENV{BUILD_BENCHMARK})
  add_subdirectory(third_party/atomic_queue)
  add_subdirectory(benchmark)
endif($ENV{BUILD_BENCHMARK})

install(
  TARGETS mpmc-queue mpsc-queue
  EXPORT ${PROJECT_NAME}
  DESTINATION $ENV{out}/lib)

install(
  EXPORT ${PROJECT_NAME}
  DESTINATION $ENV{out}/lib/cmake
  FILE ${PROJECT_NAME}-config.cmake
  NAMESPACE theta::)
