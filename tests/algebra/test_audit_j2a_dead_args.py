"""Audit R8: no dead argument in the nested helpers of the fused-RHS compiler.
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]

def test_fused_integ1_has_no_dead_argument():
    """R8: integ1(d, i, length) never read `length` (the length is applied when the products are
    post-processed). Every argument of the nested helpers of _compile_* must be used."""
    src = (ROOT / "python/conatus/timestep/fused.py").read_text()
    dead = []
    for fn in ast.walk(ast.parse(src)):
        if isinstance(fn, ast.FunctionDef) and fn.name in ("integ1", "diff1"):
            used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
            dead += [f"{fn.name}({a.arg})" for a in fn.args.args if a.arg not in used]
    assert not dead, dead
