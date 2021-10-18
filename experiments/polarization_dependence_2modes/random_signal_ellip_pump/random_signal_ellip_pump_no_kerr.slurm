#!/bin/bash
#SBATCH -c 32
#SBATCH -p allgroups
#SBATCH --mem 16G

Lk=1

EXP_NAME=random_signal_ellip_pump_no_kerr_Lk=${Lk}m
OUTPUT_FILE=${EXP_NAME}_out.txt
ERR_FILE=${EXP_NAME}_err.txt

srun -o $OUTPUT_FILE -e $ERR_FILE singularity exec ../../../ubuntu_mkl_1.0.sif python -u -W  ignore ../random_signal_pol_fixed_pump_pol_ellip.py \
        --experiment-name $EXP_NAME \
        --perturbation-beat-length ${Lk} \
        --correlation-length 10 --dz 1 \
        --fiber-length 50 --total-pump-power 1000 \
        --signal-power-per-mode 1e-4 \
        --sampling 250 \
        --sigma 0 \
        --runs-per-batch 32 --forever