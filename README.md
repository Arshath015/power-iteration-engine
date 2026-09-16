# Power Iteration Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python >=3.9](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

Estimate the dominant eigenvalue of a matrix via a fully config‑driven engine.

## Table of Contents
- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Theoretical Background](#theoretical-background)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Configuration Analysis](#configuration-analysis)
- [Testing](#testing)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [License](#license)

## Overview
The package implements the classic power iteration method from first
principles. All runtime behaviour—matrix, tolerance, maximum iterations, and
optional initial vector—is supplied via a declarative YAML/JSON configuration
file, enabling reproducible experiments without code changes.

## Tech Stack
- Python >=3.9
- NumPy for dense linear algebra
- PyYAML for configuration parsing
- Pydantic for schema validation
- pytest for testing

## Architecture
```text
power-iteration-engine/
├─ engine/
│  ├─ __init__.py        # public API
│  ├─ iteration.py      # core algorithm
│  └─ config.py         # config parsing & validation
├─ config/
│  └─ example_config.yaml
├─ tests/
│  ├─ test_iteration.py
│  └─ test_edge.py
├─ examples/
│  └─ run_power_iteration.py
├─ docs/
│  └─ analysis.md
└─ README.md
```

## Theoretical Background
Power iteration is an eigenvalue algorithm that repeatedly applies a matrix
``A`` to a vector ``v`` and normalises the result. Under the assumption that ``A``
has a unique dominant eigenvalue ``λ₁`` (|λ₁| > |λ₂| ≥ …), the sequence converges to
the corresponding eigenvector ``x₁``. The Rayleigh quotient ``vᵀAv / vᵀv`` provides
an increasingly accurate estimate of ``λ₁`` as the iteration proceeds.

Convergence speed depends on the ratio ``|λ₂/λ₁|`` and on the alignment of the
initial vector with ``x₁``. By exposing ``init_vec`` through the configuration we
allow users to control this alignment explicitly, turning a stochastic process
into a deterministic one when required.

## Installation
```bash
git clone https://github.com/yourorg/power-iteration-engine.git
cd power-iteration-engine
pip install -r requirements.txt
```

## Usage
Run the bundled example which reads ``config/example_config.yaml`` and writes
its result to ``examples/results/output.txt``:
```bash
python examples/run_power_iteration.py
```

## API Reference
- **EngineConfig.from_file(path: str|Path) → EngineConfig** – Load and validate a YAML/JSON config.
- **EngineConfig.to_numpy() → Tuple[np.ndarray, np.ndarray|None]** – Convert stored lists to NumPy arrays.
- **PowerIteration(matrix: np.ndarray, max_iter: int = 1000, tol: float = 1e-8, init_vec: np.ndarray|None = None)** – Constructor.
- **PowerIteration.run() → Tuple[float, np.ndarray, int]** – Execute the iteration and return eigenvalue, eigenvector, and iteration count.

## Configuration Analysis
See the detailed study in [docs/analysis.md](docs/analysis.md).

## Testing
```bash
pytest -q
```
The test suite includes convergence verification against a known analytical
solution and edge‑case checks for singular matrices and mismatched initial
vectors.

## Limitations
- Works only for real‑valued dense matrices.
- Convergence is not guaranteed for matrices with multiple dominant eigenvalues.
- No support for sparse structures or complex numbers.

## Roadmap
- Add optional support for sparse matrices via SciPy.
- Extend schema to allow multiple runs in a single config (batch mode).
- Provide a CLI wrapper that forwards config paths to the engine.

## License
MIT License
