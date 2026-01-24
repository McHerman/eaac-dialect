{
  description = "Local LLVM / MLIR / CIRCT / Chisel Dev Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        python = pkgs.python313;

        cmakePrefix = pkgs.lib.makeSearchPath "lib/cmake" [
          pkgs.zlib.dev
          pkgs.libedit.dev
          pkgs.libxml2.dev
        ];
      in with pkgs; {
        devShells.default = pkgs.mkShell.override { stdenv = overrideCC llvmPackages.stdenv (llvmPackages.stdenv.cc.override { inherit (llvmPackages) bintools; }); } {

          buildInputs = [
            python313Packages.pyutil
            python313Packages.docutils
            # Toolchain + build
            clang
            lld
            cmake
            ninja

            # LLVM runtime / link deps
            zlib
            zlib.dev
            libedit
            libedit.dev
            libxml2
            libxml2.dev
            ncurses
            libffi

            pkg-config

            # MLIR tooling
            python
          ];

          CMAKE_PREFIX_PATH = cmakePrefix;

          shellHook = ''
            export CC=clang
            export CXX=clang++

            echo "Local LLVM / MLIR dev shell ready"
          '';
        };
      });
}
