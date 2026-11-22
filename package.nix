# FILENAME: package.nix
{
  llvmPackages_23,
  cmake,
  pkg-config,
  abseil-cpp,
  boost,
  fmt,
  folly,
  gbenchmark,
  glog,
  gtest,
  linuxHeaders,
  theta-cmake ? null,
  theta-mutex ? null,
  src,
}:
llvmPackages_23.stdenv.mkDerivation {
  inherit src;
  pname = "theta-queue";
  version = "0.1.0";

  outputs = [ "out" ];

  buildInputs = [
    abseil-cpp
    boost
    fmt
    folly
    gbenchmark
    glog
    gtest
    linuxHeaders
  ]
  ++ builtins.filter (x: x != null) [ theta-mutex ];

  nativeBuildInputs = [
    cmake
    pkg-config
    llvmPackages_23.clang-tools
  ]
  ++ builtins.filter (x: x != null) [ theta-cmake ];

  dontStrip = true;
  cmakeBuildType = "Release";
  separateDebugInfo = true;

  doCheck = false;
  env = {
    BUILD_BENCHMARK = "OFF";
  };
}
