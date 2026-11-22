# FILENAME: flake.nix
{
  description = "theta-queue: Lock-free MPMC, MPSC, SPSC, and unbounded concurrent queues";

  inputs = {
    theta-utils.url = "github:LoganEvans/theta-utils";
    nixpkgs.follows = "theta-utils/nixpkgs";
    flake-utils.follows = "theta-utils/flake-utils";
    treefmt-nix.follows = "theta-utils/treefmt-nix";
  };

  outputs =
    {
      self,
      nixpkgs,
      theta-utils,
      ...
    }:
    let
      base = theta-utils.lib.mkThetaFlake {
        inherit self;
        pname = "theta-queue";
        packageFn = ./package.nix;
        treefmtModule = ./treefmt.nix;
      };
    in
    base
    // {
      devShells = builtins.mapAttrs (
        system: shells:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = shells.default.overrideAttrs (oldAttrs: {
            nativeBuildInputs = (oldAttrs.nativeBuildInputs or [ ]) ++ [
              pkgs.statix
              pkgs.deadnix
            ];
          });
        }
      ) base.devShells;
    };
}
