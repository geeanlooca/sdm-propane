#!/bin/bash
#SBATCH -c 32
#SBATCH -p allgroups
#SBATCH --mem 16G
#SBATCH --array=0-39%4
#SBATCH -o orthogonal_linear_pols_%A.%a_out.txt
#SBATCH -e orthogonal_linear_pols_%A.%a_err.txt


MAX_FIBERS=5000
LK_MIN=-2
LK_MAX=4.5
FIBER_LENGTH=10
LC=10
DZ=1

singularity exec ../../../ubuntu_mkl_1.0.sif python -u -W  ignore ../elliptical_polarizations_uniform_pumping_coupling_sweep.py \
        --correlation-length $LC --dz $DZ \
        --fiber-length $FIBER_LENGTH --total-pump-power 1000 \
        --signal-power-per-mode 1e-4 \
        --sampling 500 \
        --runs-per-batch 32 \
        --numpy-seed 0 \
        --polarization parallel \
        --min-beat-length ${LK_MIN} \
        --max-beat-length ${LK_MAX} \
        --max-fibers ${MAX_FIBERS}