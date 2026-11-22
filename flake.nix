# FILENAME: flake.nix
{
  description = "theta-queue: Lock-free MPMC and MPSC concurrent queues";

  inputs = {
    theta-utils.url = "github:LoganEvans/theta-utils";
    nixpkgs.follows = "theta-utils/nixpkgs";
    flake-utils.follows = "theta-utils/flake-utils";
    treefmt-nix.follows = "theta-utils/treefmt-nix";

    theta-mutex = {
      url = "github:LoganEvans/theta-mutex";
      inputs.theta-utils.follows = "theta-utils";
    };
  };

  outputs =
    {
      self,
      theta-utils,
      theta-mutex,
      ...
    }:
    let
      base = theta-utils.lib.mkThetaFlake {
        inherit self;
        pname = "theta-queue";
        packageFn = ./package.nix;
        treefmtModule = ./treefmt.nix;
      };

      injectThetaMutex =
        system: drv:
        drv.overrideAttrs (oldAttrs: {
          buildInputs = (oldAttrs.buildInputs or [ ]) ++ [
            theta-mutex.packages.${system}.default
          ];
        });
    in
    base
    // {
      packages = builtins.mapAttrs (
        system: pkgsSet:
        pkgsSet
        // {
          theta-queue = injectThetaMutex system pkgsSet.theta-queue;
          default = injectThetaMutex system pkgsSet.default;
          test = injectThetaMutex system pkgsSet.test;
          benchmark = injectThetaMutex system pkgsSet.benchmark;
          benchmark-lto = injectThetaMutex system pkgsSet.benchmark-lto;
        }
      ) base.packages;

      devShells = builtins.mapAttrs (system: shells: {
        default = injectThetaMutex system shells.default;
      }) base.devShells;

      checks = builtins.mapAttrs (
        system: chkSet:
        chkSet
        // {
          test = injectThetaMutex system chkSet.test;
        }
      ) base.checks;
    };
}
