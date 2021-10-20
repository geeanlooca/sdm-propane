#!/bin/bash
#SBATCH -c 32
#SBATCH -p allgroups
#SBATCH --mem 16G
#SBATCH --array=0-19%2


MAX_FIBERS=5000
LK_MIN=5e-5
LK_MAX=3e1
OUTPUT_FILE=random_polarization_coupling_sweep_%a_out.txt
ERR_FILE=random_polarization_coupling_sweep_%a_err.txt

srun -o $OUTPUT_FILE -e $ERR_FILE singularity exec ../../../ubuntu_mkl_1.0.sif python -u -W  ignore ../random_signal_random_pump_sweep.py \
        --correlation-length 10 --dz 1 \
        --fiber-length 50 --total-pump-power 1000 \
        --signal-power-per-mode 1e-4 \
        --sampling 1000 \
        --runs-per-batch 32 \
        --min-beat-length ${LK_MIN} \
        --max-beat-length ${LK_MAX} \
        --max-fibers ${MAX_FIBERS}