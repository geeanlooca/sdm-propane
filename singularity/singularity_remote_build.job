#!/bin/bash
#SBATCH -J singularity_build
#SBATCH -o singularity_build_output_%j.txt
#SBATCH -e singularity_build_error_%j.txt
#SBATCH -n 1
#SBATCH -p allgroups
#SBATCH --mem 10G

srun singularity build --remote singularity_image.sif singularity/singularity_image.def

