"""Power iteration implementation.

The algorithm iteratively multiplies a vector by the matrix and normalises it.
Convergence is declared when the change in the Rayleigh quotient falls below
`tol` for `max_iter` steps. All linear algebra is performed with NumPy to keep
the implementation clear and fast.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

class PowerIteration:
    """Estimate the dominant eigenpair of a square matrix.

    Parameters
    ----------
    matrix: np.ndarray
        Real‑valued square matrix (n x n).
    max_iter: int, default 1000
        Upper bound on the number of iterations.
    tol: float, default 1e-8
        Convergence tolerance on the relative change of the Rayleigh quotient.
    init_vec: np.ndarray | None, default None
        Optional initial vector; if omitted a random unit vector is used.
    """

    def __init__(self, matrix: np.ndarray, max_iter: int = 1000, tol: float = 1e-8, init_vec: np.ndarray | None = None) -> None:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square (2‑D)")
        self.matrix = matrix.astype(float)
        self.n = matrix.shape[0]
        self.max_iter = max_iter
        self.tol = tol
        if init_vec is None:
            vec = np.random.rand(self.n)
        else:
            vec = np.asarray(init_vec, dtype=float)
            if vec.shape != (self.n,):
                raise ValueError("init_vec must have shape (n,)")
        self.vec = vec / np.linalg.norm(vec)

    def _rayleigh(self, v: np.ndarray) -> float:
        """Return the Rayleigh quotient vᵀAv / (vᵀv)."""
        return float(v @ self.matrix @ v) / float(v @ v)

    def run(self) -> Tuple[float, np.ndarray, int]:
        """Execute the iteration.

        Returns
        -------
        eigenvalue: float
            Approximated dominant eigenvalue.
        eigenvector: np.ndarray
            Normalised eigenvector associated with ``eigenvalue``.
        iterations: int
            Number of iterations performed.
        """
        prev_lambda = self._rayleigh(self.vec)
        for i in range(1, self.max_iter + 1):
            # Multiply and re‑normalise
            w = self.matrix @ self.vec
            norm = np.linalg.norm(w)
            if norm == 0:
                raise RuntimeError("Encountered zero vector during iteration; matrix may be singular.")
            self.vec = w / norm
            # Rayleigh quotient for convergence check
            cur_lambda = self._rayleigh(self.vec)
            if abs(cur_lambda - prev_lambda) <= self.tol * max(abs(cur_lambda), 1.0):
                return cur_lambda, self.vec.copy(), i
            prev_lambda = cur_lambda
        # Max iterations reached – return best estimate
        return prev_lambda, self.vec.copy(), self.max_iter
