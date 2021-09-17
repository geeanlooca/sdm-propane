import numpy as np
from raman_linear_coupling import indexing

def test_indexing():
    a = np.random.random((5,6,7,8))
    data = indexing(a, 1, 2, 3, 4)

    data_py = a[1,2,3,4]

    assert data == data_py

