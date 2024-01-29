let
  pkgs = import <nixpkgs> { config = {}; overlays = []; };
in

pkgs.mkShell {
  packages = with pkgs; [
    python3
    python3Packages.numpy
    python3Packages.mnist
    python3Packages.uvicorn
    python3Packages.fastapi
  ];
}
