#!/bin/bash
#SBATCH -c 32
#SBATCH -p allgroups
#SBATCH --mem 16G
#SBATCH --array=0-49%4
#SBATCH -o %A.%a_out.txt
#SBATCH -e %A.%a_err.txt


MAX_FIBERS=5000
LK_MIN=-2
LK_MAX=4.5
FIBER_LENGTH=10
LC=10
DZ=1
POLARIZATION=elliptical
ORIENTATION=parallel
BIREF=0.5
MODES=2
PERCENT=0.01

SINGULARITY_IMG=../../../ubuntu_mkl_1.0.sif
CMD=python -u -W  ignore ../uniform_pumping_coupling_sweep.py
BIND=--bind ~/peg/results:/results

singularity exec $BIND $SINGULARITY_IMG  $CMD\
        --correlation-length $LC \
        --dz $DZ \
        --fiber-length $FIBER_LENGTH \
        --total-pump-power 1000 \
        --signal-power-per-mode 1e-4 \
        --sampling 500 \
        --runs-per-batch 32 \
        --numpy-seed 0 \
        --polarization $POLARIZATION \
        --polarization-orientation $ORIENTATION \
        --birefringence-weight $BIREF \
        --min-beat-length ${LK_MIN} \
        --max-beat-length ${LK_MAX} \
        --max-fibers ${MAX_FIBERS} \
        --modes $MODES \
        --percent $PERCENT \
        --output-dir /results
