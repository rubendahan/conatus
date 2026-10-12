"""What a checkpoint is a state *of*: model, equations, parameters, resolution, geometry.

A restart used to compare only the numerics (scheme, dt, start-up, size of the state vector,
ordering): a checkpoint of another model restarted silently, a heat run at kappa = 1 continued
at kappa = 5, a run on a box of length 2 pi continued on a box of length 22, as long as the state
vector had the same length. Every checkpoint now records its **identity** (``meta["identity"]``,
both formats) and a restart compares it with the solver it is loaded into:

* ``model``: the name of the problem (the native model's name);
* ``equations``: a hash of the equations as compiled (kind, unknowns, left and right sides,
  symmetry subspace), independent of the parameter values; compared only when the resolution
  and the geometry agree (the tau lifts of the compiled equations carry the basis size and
  bounds, so a resolution change would also show as an equation change);
* ``parameters``: the value of every declared parameter;
* ``resolution``: the size of the basis along each coordinate;
* ``geometry``: the basis family and the bounds (or radius) along each coordinate.

Any difference is refused with the list of what differs, unless ``allow=`` names it: a
parameter name (``allow=("kappa",)``: continue this state at a new kappa, the legitimate use) or
one of the categories ``"parameters"``, ``"model"``, ``"equations"``, ``"geometry"``. The
resolution can never be allowed (the state would be read in the wrong layout). An allowed
difference is recorded: ``explain()`` prints it and every later checkpoint carries it
(``meta["restart_changes"]``). A checkpoint written before this record (no ``identity``) is
restarted with a warning, after comparing what it holds (the problem name).
"""
from __future__ import annotations

import hashlib
import warnings
from collections.abc import Iterable
from typing import Any

from ..errors import OptionError, did_you_mean

#: The parts of a checkpoint's identity, in the order a mismatch lists them.
IDENTITY_KEYS = ("model", "resolution", "geometry", "equations", "parameters")
#: Categories ``allow=`` accepts besides parameter names ("resolution" is never allowed).
ALLOWABLE = ("parameters", "model", "equations", "geometry")
_DOC = "docs/site/guide/io.md"


class CheckpointWarning(UserWarning):
    """A restart from a checkpoint that cannot be fully checked (written before the identity
    record): what could not be compared is named."""


def equations_hash(problem: Any) -> str:
    """Hash of the equations as compiled: kind, unknown names and tensor signatures (in order:
    they lay out the state), both sides of every equation (in any order) and the symmetry
    subspace. Stable across processes, platforms and Python versions (sha256 of canonical
    structural keys; whitespace and the order of terms or equations do not change it). Parameter
    values are not part of it (``Parameter``'s structural key has the name only); the domains of
    the fields enter through the structural keys (size, bounds, dealiasing factor), which is why
    it is compared only at equal resolution and geometry, and why a change of the dealiasing
    factor shows as an equations change."""
    h = hashlib.sha256()
    h.update(str(getattr(problem, "kind", "")).encode())
    for v in getattr(problem, "variables", ()):
        h.update(f"var:{v.name}:{len(v.tensorsig)}".encode())
    # the equations as a set: declaring them in another order is the same system (the state is
    # laid out by the unknowns, hashed in order above); each side's structural hash is a sha256 of
    # the canonical expression (sums and products sorted), never Python's salted hash()
    for e in sorted(f"{eq.lhs.structural_hash}={eq.rhs.structural_hash}"
                    for eq in getattr(problem, "equations", ())):
        h.update(e.encode())
    sym = getattr(problem, "symmetry", None)
    if sym is not None:
        h.update(repr(sym).encode())
    return h.hexdigest()[:16]


def _bases(plan: Any) -> list:
    dom = getattr(plan, "domain", None)
    return list(getattr(dom, "bases", ()) or ())


def checkpoint_identity(solver: Any) -> dict:
    """The identity record of ``solver``'s current problem (module docstring)."""
    from ._describe import parameters_of
    plan = solver.plan
    problem = plan.problem
    res, geo = {}, {}
    for b in _bases(plan):
        c = b.coord.name
        res[c] = int(b.size) if not isinstance(b.size, tuple) else [int(s) for s in b.size]
        g = [type(b).__name__]
        bounds = getattr(b, "bounds", None)
        if bounds is not None:
            g.append([float(x) for x in bounds])
        radius = getattr(b, "radius", None)
        if radius is not None and not callable(radius):
            g.append(float(radius))
        geo[c] = g
    return {"model": str(getattr(problem, "name", "") or ""),
            "equations": equations_hash(problem),
            "parameters": parameters_of(solver),
            "resolution": res,
            "geometry": geo}


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, list) and len(v) == 2 and all(isinstance(x, float) for x in v):
        return f"{complex(v[0], v[1]):g}"
    return repr(v)


def _differences(old: dict, new: dict) -> list[tuple[str, str, str]]:
    """(allow key, category, text) of every difference between two identity records."""
    out = []
    same_layout = old.get("resolution") == new.get("resolution") and \
        old.get("geometry") == new.get("geometry")
    for key in IDENTITY_KEYS:
        if key not in old or key not in new:
            continue
        a, b = old[key], new[key]
        if key == "parameters":
            for p in sorted(set(a) | set(b)):
                if a.get(p) != b.get(p):
                    out.append((p, "parameters", f"parameter {p}: {_fmt(a.get(p)) if p in a else '(absent)'}"
                                f" in the checkpoint, {_fmt(b.get(p)) if p in b else '(absent)'} now"))
        elif key == "equations":
            if same_layout and a != b:
                out.append(("equations", "equations", f"equations: hash {a} in the checkpoint, "
                            f"{b} now (another equation, term or sign, or another dealiasing "
                            "factor)"))
        elif key in ("resolution", "geometry"):
            for c in sorted(set(a) | set(b)):
                if a.get(c) != b.get(c):
                    out.append((key, key, f"{key} along {c}: {a.get(c)!r} in the checkpoint, "
                                f"{b.get(c)!r} now"))
        elif a != b:
            out.append((key, key, f"{key}: {a!r} in the checkpoint, {b!r} now"))
    return out


def check_allow(allow: Any, params: Iterable[str], where: str) -> tuple[str, ...]:
    """``allow=`` as a tuple of names: parameter names or the categories of ALLOWABLE."""
    if allow is None or allow == ():
        return ()
    if isinstance(allow, str):
        allow = (allow,)
    try:
        names = tuple(str(a) for a in allow)
    except TypeError:
        raise OptionError(f"{where}(allow={allow!r})",
                          why="allow= lists what may differ between the checkpoint and this run",
                          fix="allow=('kappa',) (a parameter) or allow=('parameters',)",
                          doc=_DOC) from None
    params = list(params)
    for n in names:
        if n == "resolution":
            raise OptionError(f"{where}(allow=('resolution',))",
                              why="the state is a vector of spectral coefficients laid out for one "
                                  "resolution: read at another one it would be another state",
                              fix="restart at the resolution of the checkpoint, then interpolate "
                                  "explicitly (sim.grid(...) / initial(...) at the new N)", doc=_DOC)
        if n not in params and n not in ALLOWABLE:
            raise OptionError(f"{where}(allow=({n!r},)): {n!r} is neither a parameter of this "
                              "run nor one of " + ", ".join(repr(a) for a in ALLOWABLE)
                              + "." + did_you_mean(n, [*params, *ALLOWABLE]),
                              why="allow= names what may differ between the checkpoint and this "
                                  "run; an unknown name would allow nothing and hide the typo",
                              fix="parameters: " + (", ".join(params) or "(none)"), doc=_DOC)
    return names


def compare_identity(meta: dict, solver: Any, allow: tuple[str, ...], where: str,
                     path: Any = None) -> list[str]:
    """Compare a checkpoint's identity with ``solver``; refuse what ``allow`` does not name.
    Returns the allowed differences (text, recorded by the caller)."""
    mine = checkpoint_identity(solver)
    old = meta.get("identity")
    src = f" {path}" if path is not None else ""
    if old is None:
        prob = meta.get("problem")
        if prob is not None and prob != mine["model"] and "model" not in allow:
            raise OptionError(
                f"{where}: the checkpoint{src} is a state of '{prob}', this run solves "
                f"'{mine['model']}'",
                why="a restart continues the state of the same problem; another model would "
                    "start from a state that is not its own",
                fix="restart from a checkpoint of this model, or allow=('model',) to continue this "
                    "state deliberately (recorded by explain())", doc=_DOC)
        warnings.warn(
            f"{where}: the checkpoint{src} was written before checkpoints recorded their model, "
            "equations, parameters, resolution and geometry: only the problem name and the "
            "numerics (scheme, dt, size of the state) were compared. Re-save it (sim.save(...)) "
            "after the restart to record them.", CheckpointWarning, stacklevel=4)
        return []
    diffs = _differences(old, mine)
    refused = [t for key, cat, t in diffs if key not in allow and cat not in allow]
    if refused:
        raise OptionError(
            f"{where}: the checkpoint{src} is not a state of this run: " + "; ".join(refused),
            why="a restart continues the same problem; a checkpoint of another model, other "
                "equations, other parameters, another resolution or another box would continue "
                "silently as if it were this run's own state",
            fix=_fix(diffs, allow), doc=_DOC)
    return [t for _k, _c, t in diffs]


def _fix(diffs: list, allow: tuple[str, ...]) -> str:
    """The fix line of a refused restart: what to rebuild, or what ``allow=`` would name."""
    cats = [c for k, c, _ in diffs if k not in allow and c not in allow]
    if "resolution" in cats:
        return ("build the run at the resolution of the checkpoint (a state cannot be read at "
                "another N; interpolate explicitly with sim.grid(...) and initial(...))")
    keys = list(dict.fromkeys(k for k, c, _ in diffs if k not in allow and c not in allow))
    return ("rebuild the run with the model, equations, parameters and box of the checkpoint, "
            "or name what may change deliberately: allow=(" + ", ".join(repr(k) for k in keys)
            + ("," if len(keys) == 1 else "") + ") (recorded by explain() and in every later "
            "checkpoint)")
