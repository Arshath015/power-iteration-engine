import numpy as np
from engine.iteration import PowerIteration

def test_converges_to_known_eigenpair():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    pi = PowerIteration(A, max_iter=200, tol=1e-12)
    eigenvalue, eigenvector, iters = pi.run()
    # Dominant eigenvalue of this matrix is 3.0
    assert abs(eigenvalue - 3.0) < 1e-9
    # Eigenvector should be proportional to [1, 1]
    assert np.allclose(eigenvector, np.array([1.0, 1.0]) / np.sqrt(2), atol=1e-6)
    assert iters < 200
