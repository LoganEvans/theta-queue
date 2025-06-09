{
  clangStdenv,
  cmake,
  fmt,
  gbenchmark,
  glog,
  gtest,
  librseq,
  pkgconf,
  src,
  theta-debug-utils,
  readerwriterqueue,
  linuxPackages,
  flamegraph,
  fetchFromGitHub,
}:
let
  concurrentqueue = fetchFromGitHub {
    owner = "cameron314";
    repo = "concurrentqueue";
    tag = "v1.0.4";
    hash = "sha256-MkhlDme6ZwKPuRINhfpv7cxliI2GU3RmTfC6O0ke/IQ=";
  };
in
clangStdenv.mkDerivation rec {
  inherit src;
  name = "queue";

  outputs = [
    "out"
  ];

  cq = concurrentqueue;
  buildInputs = [
    fmt
    gbenchmark
    glog
    gtest
    theta-debug-utils
    concurrentqueue
  ];

  nativeBuildInputs = [
    cmake
    pkgconf
    linuxPackages.perf
    flamegraph
  ];

  dontStrip = true;
  cmakeBuildType = "Release";
  separateDebugInfo = true;

  doCheck = false;
  env = {
    BUILD_BENCHMARK = false;
  };
}
