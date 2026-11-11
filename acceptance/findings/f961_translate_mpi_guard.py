"""F961 (review of 0.2.3, G8): a condition is an MPI guard by its structure, not its text.

Oracle: Python on the serial run (rank 0): `if rank_threshold > 1:` with rank_threshold = 0
skips its body; `if comm.rank == 0:` runs it. A guard whose body changes field data, a test of
another rank, an unknown condition and `if __name__ != "__main__":` are refused (complete=False).
"""
import warnings

from _acceptance import check, passed, start
from _value_model import classify

start(961, "MPI guards recognised structurally, other conditions refused")
import conatus as cn  # noqa: E402
from conatus import errors  # noqa: E402

warnings.simplefilter("ignore")
EXACT = {
    "rank-0 guard, output": ["u['g'] = 1", "if comm.rank == 0:\n    print(u['g'].max())"],
    "Get_rank name": ["r = comm.Get_rank()", "if r == 0:\n    print('x')", "u['g'] = 2"],
    "dist.comm": ["if 0 == dist.comm.rank:\n    print('x')", "u['g'] = 3"],
    "main guard": ["if __name__ == '__main__':\n    u['g'] = 4"],
}
HONEST = {
    "rank_threshold > 1": ["u['g'] = 1", "if rank_threshold > 1:\n    u['g'] = 2"],
    "rank_threshold < 1": ["u['g'] = 1", "if rank_threshold < 1:\n    u['g'] = 2"],
    "rank-0 guard writes data": ["if comm.rank == 0:\n    u['g'] = 2"],
    "other ranks": ["if comm.rank != 0:\n    print('x')"],
    "not main": ["if __name__ != '__main__':\n    u['g'] = 5"],
}
for name, st in EXACT.items():
    cat, why = classify(st, cn, errors)
    check(cat == "equal", f"{name}: {cat} {why}")
for name, st in HONEST.items():
    cat, why = classify(st, cn, errors)
    check(cat in ("incomplete", "refused"), f"{name}: {cat} {why}")
passed(f"{len(EXACT)} guards translated, {len(HONEST)} conditions refused")
