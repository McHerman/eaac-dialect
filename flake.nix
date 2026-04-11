{
  description = "Local LLVM / MLIR / CIRCT / Chisel Dev Environment";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in with pkgs; {
        devShell = mkShell.override { 
          stdenv = overrideCC llvmPackages.stdenv 
            (llvmPackages.stdenv.cc.override { inherit (llvmPackages) bintools; }); 
        } {
          name = "llvm-mlir-env";
          
          buildInputs = [
            python3
            python3Packages.pyutil
            python3Packages.docutils
            
            # Toolchain + build
            clang
            lld
            cmake
            ninja
            ccache
            
            # LLVM runtime / link deps
            zlib
            libedit
            libxml2
            ncurses
            libffi
            pkg-config

            # Debugging
            lldb
            graphviz

            # Other stuff
            just
          ];
          
          shellHook = ''
            export CC=clang
            export CXX=clang++
            echo "Local LLVM / MLIR dev shell ready"
          '';
        };
      });
}
