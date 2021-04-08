raman_linear_coupling:
	c++ -O3 -Wall -shared -std=c++14 -fPIC $(shell python3 -m pybind11 --includes) -Ipropagators/common -I/usr/include/mkl propagators/raman_linear_coupling/raman_linear_coupling.cpp propagators/common/matrix_exponential.cpp -o raman_linear_coupling$(shell python3-config --extension-suffix) -lmkl_rt -m64

common:
	c++ -O3 -Wall -shared -std=c++14 -fPIC $(shell python3 -m pybind11 --includes) -Ipropagators/common -I/usr/include/mkl propagators/common/matrix_exponential_binding.cpp propagators/common/matrix_exponential.cpp -o matrix_exponential$(shell python3-config --extension-suffix) -lmkl_rt -m64

test:
	python -m pytest -s tests/