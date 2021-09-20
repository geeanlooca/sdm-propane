CC = c++
CCOPT = -O3 -Wall -shared -std=c++17 -fPIC -ffast-math
PYBIND = $(shell python3 -m pybind11 --includes)
INC = -Ipropagators/common -I/usr/include/mkl

raman_linear_coupling:
	$(CC) $(CCOPT) $(PYBIND) $(INC) propagators/raman_linear_coupling/raman_linear_coupling.cpp propagators/common/matrix_exponential.cpp -o raman_linear_coupling$(shell python3-config --extension-suffix) -lmkl_rt -m64 -fopenmp

common:
	$(CC) $(CCOPT) $(PYBIND) $(INC) propagators/common/matrix_exponential_binding.cpp propagators/common/matrix_exponential.cpp -o matrix_exponential$(shell python3-config --extension-suffix) -lmkl_rt -m64

blas_multicore$(shell python3-config --extension-suffix): propagators/tests/blas_multicore.cpp
	$(CC) $(CCOPT) $(PYBIND) $(INC) propagators/tests/blas_multicore.cpp  -o blas_multicore$(shell python3-config --extension-suffix) -lmkl_rt -m64 -fopenmp

blas: blas_multicore$(shell python3-config --extension-suffix)
	python -c "import blas_multicore; import numpy as np; a=np.random.random((2000, 2000)); blas_multicore.test_eigenvals(a);"

omp: blas_multicore$(shell python3-config --extension-suffix)
	python -c "import blas_multicore; blas_multicore.test_omp()"

test:
	python -m pytest -s tests/