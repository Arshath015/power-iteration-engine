"""Configuration handling for the engine.

The engine is driven entirely by a JSON/YAML configuration file that maps
directly onto the ``PowerIteration`` constructor. Validation is performed with
Pydantic to give clear error messages while keeping the schema declarative.
"""

from __future__ import annotations

import json
import yaml
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field, validator
import numpy as np

class EngineConfig(BaseModel):
    """Pydantic schema mirroring the arguments of :class:`PowerIteration`."""

    matrix: list[list[float]] = Field(..., description="Square matrix as a nested list.")
    max_iter: int = Field(1000, ge=1, description="Maximum number of power‑iteration steps.")
    tol: float = Field(1e-8, gt=0.0, description="Convergence tolerance.")
    init_vec: list[float] | None = Field(None, description="Optional initial vector.")

    @validator("matrix")
    def square_matrix(cls, v: list[list[float]]) -> list[list[float]]:
        if not v:
            raise ValueError("matrix must not be empty")
        n = len(v)
        if any(len(row) != n for row in v):
            raise ValueError("matrix must be square (same number of rows and columns)")
        return v

    @validator("init_vec")
    def vec_length(cls, v: list[float] | None, values: Dict[str, Any]) -> list[float] | None:
        if v is None:
            return v
        n = len(values.get("matrix", []))
        if len(v) != n:
            raise ValueError("init_vec length must match matrix dimension")
        return v

    @classmethod
    def from_file(cls, path: str | Path) -> "EngineConfig":
        """Load a configuration from a JSON or YAML file.

        Parameters
        ----------
        path: str or Path
            Path to the configuration file.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        if p.suffix.lower() in {".yaml", ".yml"}:
            raw = yaml.safe_load(p.read_text())
        elif p.suffix.lower() == ".json":
            raw = json.loads(p.read_text())
        else:
            raise ValueError("Unsupported config format; use .yaml, .yml or .json")
        return cls(**raw)

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Convert stored lists to NumPy arrays for the engine.

        Returns
        -------
        matrix: np.ndarray
        init_vec: np.ndarray | None
        """
        mat = np.array(self.matrix, dtype=float)
        vec = np.array(self.init_vec, dtype=float) if self.init_vec is not None else None
        return mat, vec
