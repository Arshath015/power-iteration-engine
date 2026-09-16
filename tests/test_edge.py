import numpy as np
import pytest
from engine.iteration import PowerIteration

def test_singular_matrix_raises():
    A = np.zeros((3, 3))
    pi = PowerIteration(A, max_iter=10)
    with pytest.raises(RuntimeError, match="zero vector"):
        pi.run()

def test_invalid_init_vec_length():
    A = np.eye(2)
    with pytest.raises(ValueError, match="init_vec length"):
        PowerIteration(A, init_vec=[1.0, 2.0, 3.0])
