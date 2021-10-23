#!/bin/bash
#SBATCH -c 32
#SBATCH -p allgroups
#SBATCH --mem 16G
#SBATCH --array=0-19%2


MAX_FIBERS=5000
LK_MIN=1e-1
LK_MAX=5e2
OUTPUT_FILE=random_polarization_coupling_sweep_%a_out.txt
ERR_FILE=random_polarization_coupling_sweep_%a_err.txt
FIBER_LENGTH=10
LC=10
DZ=0.1

srun -o $OUTPUT_FILE -e $ERR_FILE singularity exec ../../../ubuntu_mkl_1.0.sif python -u -W  ignore ../random_signal_random_pump_sweep.py \
        --correlation-length $LC --dz $DZ \
        --fiber-length $FIBER_LENGTH --total-pump-power 1000 \
        --signal-power-per-mode 1e-4 \
        --sampling 100 \
        --runs-per-batch 32 \
        --numpy-seed 0 \
        --min-beat-length ${LK_MIN} \
        --max-beat-length ${LK_MAX} \
        --max-fibers ${MAX_FIBERS}