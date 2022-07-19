{ pkgs ? import <nixpkgs>
    {
        config =
        {
            allowUnfree = true;
            cudaSupport = true;
        };
    }
}:

pkgs.mkShell {
  buildInputs =
  [
    pkgs.python310
    pkgs.python310Packages.pip
    pkgs.python310Packages.virtualenv
    #pkgs.python310Packages.pytorch-bin
    #pkgs.python38Packages.pytorch
    #pkgs.python310Packages.torchvision
    #pkgs.python310Packages.pytorch-lightning
    #pkgs.python310Packages.pytorch-metric-learning
  ];
}
