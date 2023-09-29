#pragma once

#ifdef __cpp_lib_hardware_interference_size
using std::hardware_constructive_interference_size;
using std::hardware_destructive_interference_size;
#else
constexpr std::size_t hardware_constructive_interference_size = 64;
constexpr std::size_t hardware_destructive_interference_size = 128;
#endif

template <typename T>
auto constexpr is_atomic = false;

template <typename T>
auto constexpr is_atomic<std::atomic<T>> = std::atomic<T>::is_always_lock_free;

template <typename T>
auto constexpr can_be_atomic = is_atomic<std::atomic<T>>;

template <typename T>
concept AtomType = requires(T t) {
  std::is_trivially_constructible<T>::value;
  can_be_atomic<T>;
  sizeof(T) <= 8;
};

template <typename T>
static constexpr bool memset0_to_bool() {
  T t;
  memset(&t, 0, sizeof(T));
  return static_cast<bool>(t);
}

template <typename T>
concept ZeroableAtomType = requires(T t) {
  std::is_trivially_constructible<T>::value;
  can_be_atomic<T>;
  static_cast<bool>(T{}) == false;
  memset0_to_bool<T>() == false;
};
