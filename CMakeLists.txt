# FILENAME: CMakeLists.txt

cmake_minimum_required(VERSION 3.25)
project(
  theta-queue
  VERSION 0.1.0
  LANGUAGES CXX)

find_package(theta-cmake CONFIG QUIET)
if(NOT theta-cmake_FOUND
   AND EXISTS
       "${CMAKE_CURRENT_SOURCE_DIR}/../theta-utils/cmake/ThetaBuild.cmake")
  include("${CMAKE_CURRENT_SOURCE_DIR}/../theta-utils/cmake/ThetaBuild.cmake")
endif()

theta_enable_strict_warnings()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  add_compile_options($<$<CONFIG:Release,RelWithDebInfo>:-O3>
                      -march=armv8.2-a+lse+rcpc)
  if(NOT CMAKE_CXX_FLAGS MATCHES "-fsanitize")
    add_compile_options(-fomit-frame-pointer)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  add_compile_options($<$<CONFIG:Release,RelWithDebInfo>:-O3> -mbmi -mbmi2
                      -mlzcnt -mbranches-within-32B-boundaries)
  if(NOT CMAKE_CXX_FLAGS MATCHES "-fsanitize")
    add_compile_options(-fomit-frame-pointer)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang"
     AND THETA_ENABLE_LTO
     AND CMAKE_BUILD_TYPE MATCHES "^(Release|RelWithDebInfo)$")
    add_link_options(-Wl,-mllvm,--x86-branches-within-32B-boundaries)
  endif()
else()
  add_compile_options($<$<CONFIG:Release,RelWithDebInfo>:-O3>)
  if(NOT CMAKE_CXX_FLAGS MATCHES "-fsanitize")
    add_compile_options(-fomit-frame-pointer)
  endif()
endif()

theta_add_header_check()

find_package(Threads REQUIRED)

add_library(theta-queue STATIC src/queue.cpp)
add_library(theta::theta-queue ALIAS theta-queue)
add_library(theta::queue ALIAS theta-queue)
set_target_properties(theta-queue PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(
  theta-queue PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                     $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(theta-queue PUBLIC Threads::Threads)

set(THETA_QUEUE_HAS_THETA_CMAKE "OFF")
set(THETA_CMAKE_CONFIG_DIR "")
if(theta-cmake_FOUND)
  set(THETA_QUEUE_HAS_THETA_CMAKE "ON")
  get_filename_component(THETA_CMAKE_CONFIG_DIR "${theta-cmake_DIR}" ABSOLUTE)
  target_link_libraries(theta-queue PUBLIC theta::theta-cmake)
elseif(EXISTS
       "${CMAKE_CURRENT_SOURCE_DIR}/../theta-utils/include/theta/arch.hpp")
  get_filename_component(
    _theta_utils_inc "${CMAKE_CURRENT_SOURCE_DIR}/../theta-utils/include"
    REALPATH)
  target_include_directories(theta-queue
                             PUBLIC $<BUILD_INTERFACE:${_theta_utils_inc}>)
endif()

add_library(spsc-queue INTERFACE)
add_library(theta::spsc-queue ALIAS spsc-queue)
target_link_libraries(spsc-queue INTERFACE theta-queue)

add_library(mpmc-queue INTERFACE)
add_library(theta::mpmc-queue ALIAS mpmc-queue)
target_link_libraries(mpmc-queue INTERFACE theta-queue)

add_library(mpsc-queue INTERFACE)
add_library(theta::mpsc-queue ALIAS mpsc-queue)
target_link_libraries(mpsc-queue INTERFACE theta-queue)

add_library(unbounded-queue INTERFACE)
add_library(theta::unbounded-queue ALIAS unbounded-queue)
target_link_libraries(unbounded-queue INTERFACE theta-queue)

if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  include(GoogleTest)

  add_executable(queue_test tests/queue_test.cpp tests/queue_odr_helper.cpp)
  target_link_libraries(queue_test PRIVATE theta::theta-queue GTest::gtest_main)
  gtest_discover_tests(queue_test)
endif()

if(BUILD_BENCHMARK
   OR "$ENV{BUILD_BENCHMARK}" STREQUAL "ON"
   OR "$ENV{BUILD_BENCHMARK}" STREQUAL "true")
  find_package(benchmark REQUIRED)
  find_package(absl QUIET)
  find_package(theta-mutex CONFIG QUIET)
  set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH ON)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(COMP_FOLLY QUIET IMPORTED_TARGET libfolly fmt libglog)
  endif()
  set(THETA_MUTEX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../theta-mutex")
  find_path(BOOST_LOCKFREE_INCLUDE_DIR boost/lockfree/queue.hpp)
  find_path(ATOMIC_QUEUE_INCLUDE_DIR atomic_queue/atomic_queue.h
            HINTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/third_party")
  find_path(CONCURRENTQUEUE_INCLUDE_DIR moodycamel/concurrentqueue.h
            concurrentqueue.h)

  add_executable(queue_benchmark benchmarks/queue_benchmark.cpp)
  target_link_libraries(
    queue_benchmark PRIVATE theta::theta-queue benchmark::benchmark
                            Threads::Threads atomic)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(queue_benchmark PRIVATE -mcx16)
  endif()

  set(_use_pkg_theta_mutex OFF)
  if(theta-mutex_FOUND AND TARGET theta::theta-mutex)
    get_target_property(_tm_inc theta::theta-mutex
                        INTERFACE_INCLUDE_DIRECTORIES)
    if(_tm_inc AND EXISTS "${_tm_inc}/theta/mutex.hpp")
      set(_use_pkg_theta_mutex ON)
    endif()
  endif()

  if(_use_pkg_theta_mutex)
    target_link_libraries(queue_benchmark PRIVATE theta::theta-mutex)
    target_compile_definitions(queue_benchmark
                               PRIVATE THETA_QUEUE_HAS_THETA_MUTEX=1)
  elseif(EXISTS "${THETA_MUTEX_ROOT}/src/mutex.cpp" AND absl_FOUND)
    add_library(
      theta_mutex_sibling STATIC
      "${THETA_MUTEX_ROOT}/src/topology.cpp"
      "${THETA_MUTEX_ROOT}/src/mutex.cpp"
      "${THETA_MUTEX_ROOT}/src/condition_variable.cpp")
    target_include_directories(theta_mutex_sibling
                               PUBLIC "${THETA_MUTEX_ROOT}/include")
    target_link_libraries(theta_mutex_sibling PUBLIC absl::base
                                                     Threads::Threads)
    target_link_libraries(queue_benchmark PRIVATE theta_mutex_sibling)
    target_compile_definitions(queue_benchmark
                               PRIVATE THETA_QUEUE_HAS_THETA_MUTEX=1)
  endif()

  if(ATOMIC_QUEUE_INCLUDE_DIR)
    target_include_directories(queue_benchmark SYSTEM
                               PRIVATE "${ATOMIC_QUEUE_INCLUDE_DIR}")
    target_compile_definitions(queue_benchmark
                               PRIVATE THETA_QUEUE_HAS_ATOMIC_QUEUE=1)
  endif()

  if(absl_FOUND)
    target_link_libraries(queue_benchmark PRIVATE absl::synchronization
                                                  absl::base)
    target_compile_definitions(queue_benchmark PRIVATE THETA_QUEUE_HAS_ABSL=1)
  endif()

  if(BOOST_LOCKFREE_INCLUDE_DIR)
    target_include_directories(queue_benchmark SYSTEM
                               PRIVATE "${BOOST_LOCKFREE_INCLUDE_DIR}")
    target_compile_definitions(queue_benchmark PRIVATE THETA_QUEUE_HAS_BOOST=1)
  endif()

  if(COMP_FOLLY_FOUND)
    target_link_libraries(queue_benchmark PRIVATE PkgConfig::COMP_FOLLY)
    target_compile_options(
      queue_benchmark PRIVATE $<$<CXX_COMPILER_ID:Clang>:-fno-sanitize=vptr>)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
      target_compile_options(queue_benchmark
                             PRIVATE -march=armv8.2-a+lse+rcpc+nocrc)
    endif()
    target_compile_definitions(queue_benchmark PRIVATE THETA_QUEUE_HAS_FOLLY=1)
    if(NOT BOOST_LOCKFREE_INCLUDE_DIR)
      target_compile_definitions(queue_benchmark
                                 PRIVATE THETA_QUEUE_HAS_BOOST=1)
    endif()
  endif()

  if(CONCURRENTQUEUE_INCLUDE_DIR)
    target_include_directories(queue_benchmark SYSTEM
                               PRIVATE "${CONCURRENTQUEUE_INCLUDE_DIR}")
    target_compile_definitions(queue_benchmark
                               PRIVATE THETA_QUEUE_HAS_MOODYCAMEL=1)
  endif()

  install(TARGETS queue_benchmark RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()

if(NOT "${THETA_PGO_PROFILE}" STREQUAL "")
  file(
    GLOB_RECURSE
    THETA_ALL_CXX_SRCS
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
  set_property(
    SOURCE ${THETA_ALL_CXX_SRCS}
    APPEND
    PROPERTY OBJECT_DEPENDS "${THETA_PGO_PROFILE}")
endif()

set(THETA_QUEUE_DEPS_CMAKE "${CMAKE_CURRENT_BINARY_DIR}/theta-queue-deps.cmake")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/theta-queue-deps.cmake.in"
               "${THETA_QUEUE_DEPS_CMAKE}" @ONLY)

theta_install_package(
  TARGETS
  theta-queue
  spsc-queue
  mpmc-queue
  mpsc-queue
  unbounded-queue
  EXTRA_CMAKE_INCLUDE
  "theta-queue-deps.cmake"
  EXTRA_CMAKE_FILES
  "${THETA_QUEUE_DEPS_CMAKE}"
  DESCRIPTION
  "High-performance bounded and unbounded lock-free concurrent queues")
