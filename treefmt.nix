# FILENAME: treefmt.nix
{ pkgs, ... }:
{
  projectRootFile = "flake.nix";

  settings.global.excludes = [ "benchmarks/third_party/**" ];

  programs = {
    clang-format = {
      enable = true;
      package = pkgs.llvmPackages_23.clang-tools;
    };
    cmake-format = {
      enable = true;
      includes = [
        "*CMakeLists.txt"
        "cmake/*.cmake"
        "cmake/*.cmake.in"
      ];
    };
    nixfmt.enable = true;
    statix.enable = true;
    deadnix.enable = true;
  };
}
