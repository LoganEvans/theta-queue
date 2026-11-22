# FILENAME: treefmt.nix
{ pkgs, ... }:
{
  projectRootFile = "flake.nix";

  settings.global.excludes = [ "benchmarks/third_party/**" ];

  programs.clang-format.enable = true;
  programs.clang-format.package = pkgs.llvmPackages_23.clang-tools;
  programs.cmake-format.enable = true;
  programs.cmake-format.includes = [
    "*CMakeLists.txt"
    "cmake/*.cmake"
  ];
  programs.nixfmt.enable = true;
}
