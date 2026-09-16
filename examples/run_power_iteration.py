import pathlib
import numpy as np
from engine.config import EngineConfig
from engine.iteration import PowerIteration

def main():
    cfg_path = pathlib.Path(__file__).parents[1] / "config" / "example_config.yaml"
    cfg = EngineConfig.from_file(cfg_path)
    matrix, init_vec = cfg.to_numpy()
    engine = PowerIteration(matrix, max_iter=cfg.max_iter, tol=cfg.tol, init_vec=init_vec)
    eigenvalue, eigenvector, iters = engine.run()
    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "output.txt"
    with out_file.open("w") as f:
        f.write(f"Dominant eigenvalue: {eigenvalue:.12f}\n")
        f.write(f"Eigenvector: {eigenvector.tolist()}\n")
        f.write(f"Iterations: {iters}\n")
    print(f"Result written to {out_file}")

if __name__ == "__main__":
    main()
