#!/bin/bash

#SBATCH --job-name=spinchain
#SBATCH --output=output/example_sbatch.out
#SBATCH --error=error/example_sbatch.err
#SBATCH --time=00:30:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000

module load python
source activate ~/.local/env 
python3.10 ./Halfchain.py
# mv outputs/example_sbatch.out outputs/R-$(date +%Y%m%d%H%M%S).out
# mv errors/example_sbatch.err errors/R-$(date +%Y%m%d%H%M%S).err
