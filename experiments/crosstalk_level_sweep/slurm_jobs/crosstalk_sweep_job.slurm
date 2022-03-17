#!/bin/bash
#SBATCH -c 32
#SBATCH -p allgroups
#SBATCH --mem 16G
#SBATCH -o %A.%a_out.txt
#SBATCH -e %A.%a_err.txt


MAX_FIBERS=5000
LK_MIN=-3
LK_MAX=5
FIBER_LENGTH=1

SINGULARITY_IMG=../../../ubuntu_mkl_1.0.sif
CMD=python -u -W  ignore ../crosstalk_level_coupling_sweep.py
BIND=--bind ~/peg/results:/results

singularity exec $BIND $SINGULARITY_IMG  $CMD \
        --correlation-length 100 \
        --dz 10 \
        --fiber-length $FIBER_LENGTH \
        --min-beat-length ${LK_MIN} \
        --max-beat-length ${LK_MAX} \
        --beat-length-points 50 \
        --fibers ${MAX_FIBERS} \
        --output-dir /results
