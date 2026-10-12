"""Several processes write to one registry at the same time: nothing lost, every record readable."""
import multiprocessing as mp
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor

import pytest
from pathlib import Path

import numpy as np

import conatus.runs as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _workers  # noqa: E402

WORKERS, PER_WORKER = 6, 4


@pytest.mark.slow                   # 4.2 s in the fast suite (qual_durations0)
def test_concurrent_writers_lose_nothing(private_registry):
    root = str(private_registry)
    ctx = mp.get_context("spawn")                     # no fork of a process that has threads
    with ProcessPoolExecutor(WORKERS, mp_context=ctx) as ex:
        futs = [ex.submit(_workers.write_runs, root, w, PER_WORKER) for w in range(WORKERS)]
        ids = [i for f in futs for i in f.result()]
    assert len(set(ids)) == WORKERS * PER_WORKER      # unique ids, even in the same second
    # every writer indexed its own run despite the contention (before any reconciliation)
    con = sqlite3.connect(str(private_registry / "index.sqlite"))
    assert con.execute("SELECT COUNT(*) FROM runs WHERE state='completed'").fetchone()[0] == len(ids)
    con.close()
    assert R.registry().sync()["indexed"] == 0
    for w in range(WORKERS):
        runs = R.find(tags=f"w{w}", order="created")
        assert [r.params["i"] for r in runs] == list(range(PER_WORKER))
        for r in runs:
            assert r.results["total"] == w * 100 + r.params["i"]
            assert np.all(r.load_field("a") == w * 1000 + r.params["i"])
            assert np.array_equal(r.series("value")["value"], w * 100 + r.params["i"] + np.arange(20))
    assert len(R.find(total__ge=0)) == len(ids)


def test_concurrent_status_changes_keep_every_history_entry(private_registry):
    with R.track("shared") as run:
        pass
    root = str(private_registry)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(4, mp_context=ctx) as ex:
        list(ex.map(_workers.certify_many, [root] * 8, [run.id] * 8, range(8)))
    hist = R.get(run.id).rec["status_history"]
    assert len(hist) == 9 and {h["reason"] for h in hist[1:]} == {f"check by worker {w}" for w in range(8)}
