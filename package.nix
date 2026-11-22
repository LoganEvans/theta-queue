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
  ];

  propagatedBuildInputs = builtins.filter (x: x != null) [ theta-cmake ];

  nativeBuildInputs = [
    cmake
    pkg-config
    llvmPackages_23.clang-tools
  ];

  dontStrip = true;
  cmakeBuildType = "Release";
  separateDebugInfo = true;

  doCheck = false;
  env = {
    BUILD_BENCHMARK = "OFF";
  };
}
