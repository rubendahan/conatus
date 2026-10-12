"""Worker functions for the concurrency test (imported by spawned processes)."""
import numpy as np


def write_runs(root: str, worker: int, n: int) -> list[str]:
    """``n`` runs with parameters, results, a series and a field, all into ``root``."""
    import conatus.runs as R
    ids = []
    for i in range(n):
        with R.track(f"w{worker}", tags=["conc", f"w{worker}"], params={"worker": worker, "i": i},
                     root=root) as run:
            for k in range(20):
                run.log(step=k, value=worker * 100 + i + k)
            run.save_field("a", np.full((8, 8), worker * 1000 + i, dtype=float), compression="gzip")
            run.result(total=worker * 100 + i)
        assert not run.errors, run.errors
        ids.append(run.id)
    return ids


def certify_many(root: str, run_id: str, worker: int) -> None:
    """Status change of one shared run from several processes (read-modify-write under a lock)."""
    import conatus.runs as R
    R.get(run_id, root=root).certify(f"check by worker {worker}")
