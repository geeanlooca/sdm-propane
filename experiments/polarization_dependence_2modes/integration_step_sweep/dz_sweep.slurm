#!/bin/bash
#SBATCH -c 32
#SBATCH -p allgroups
#SBATCH --mem 16G
#SBATCH --array=0-5%3

Lk=0.1
DZS=(5 1 0.2 0.05 0.1)
DZ=${DZS[$SLURM_ARRAY_TASK_ID]}

EXP_NAME=integration_step_sweep_Lk_${Lk}m_dz_${DZ}m
OUTPUT_FILE=${EXP_NAME}_dz__out.txt
ERR_FILE=${EXP_NAME}_err.txt
FIBER_LENGTH=5
MAX_FIBERS=5000


srun -o $OUTPUT_FILE -e $ERR_FILE singularity exec ../../../ubuntu_mkl_1.0.sif python -u -W  ignore ../random_signal_random_pump_pol.py \
        --experiment-name $EXP_NAME \
        --perturbation-beat-length ${Lk} \
        --correlation-length 10 --dz $DZ \
        --fiber-length $FIBER_LENGTH --total-pump-power 1000 \
        --signal-power-per-mode 1e-4 \
        --sampling 100 \
        --runs-per-batch $SLURM_CPUS_PER_TASK \
        --max-fibers $MAX_FIBERS