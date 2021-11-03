#!/bin/bash
#SBATCH -c 32
#SBATCH -p allgroups
#SBATCH --mem 16G
#SBATCH --array=0-19%4


MAX_FIBERS=5000
LK_MIN=-1
LK_MAX=4
FIBER_LENGTH=10
LC=10
DZ=1
OUTPUT_FILE=parallel_polarization_uniform_pumping_coupling_sweep_L_${FIBER_LENGTH}km_dz_${DZ}m_%a_out.txt
ERR_PUT_FILE=parallel_polarization_uniform_pumping_coupling_sweep_L_${FIBER_LENGTH}km_dz_${DZ}m_%a_err.txt

srun -o $OUTPUT_FILE -e $ERR_FILE singularity exec ../../../ubuntu_mkl_1.0.sif python -u -W  ignore ../linear_polarizations_uniform_pumping_coupling_sweep.py \
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