#!/bin/bash
set -e

echo "Creating the Conda environment 'delphos-env'..."
conda env create -f environment.yml

echo "Installing Apollo 0.3.7 into the isolated R environment..."
conda run -n delphos-env Rscript install_apollo.R

echo ""
echo "========================================================="
echo "Installation Complete!"
echo "Please activate the environment by running:"
echo "    conda activate delphos-env"
echo "========================================================="
