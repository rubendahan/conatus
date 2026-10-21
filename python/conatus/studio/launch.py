"""Studio S2: prepare and submit Slurm jobs from the Studio (new run, fork, resume, Newton seed).

Stdlib only (no tornado): used by the local server (``studio/server.py``), by the relay (which runs
next to ``slurm/submit.sh``) and, for the structural checks only, by the hub (which has no file).

Rules:

* A job is **only** submitted through ``slurm/submit.sh <template> <name> -- args`` and only for a
  template of the :data:`CATALOG` (known script, declared parameters with bounds). There is no free
  command, no free script, no free path.
* Every parameter is **declared and required** (no hidden default): a missing one is refused, an
  unknown one is refused, a parameter that does not apply (e.g. a period for an equilibrium) is
  refused. Example values are shown in the page as explicit presets, never filled in silently.
* Two phases. ``prepare`` validates, renders the exact argument list, runs ``submit.sh --dry-run``
  and returns the **exact script** (template text) and command with a SHA-256 digest of
  (script path, script text, name, args). ``submit`` requires ``confirm = true`` **and** that digest:
  it re-renders and refuses if anything differs from what was shown. Submission is off unless the
  relay or the ``hub serve --local`` was started with ``--allow-submit`` (the local
  ``studio/server.py`` refuses that flag); submissions per hour are bounded.
* Every prepare / submit / refusal is appended to ``<tmp>/studio/audit.jsonl`` and every submitted
  job to ``<tmp>/studio/jobs.jsonl`` (the Studio's job list), besides submit.sh's own registries.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from conatus._private import open_private, private_dir

__all__ = ["Param", "Template", "CATALOG", "LaunchError", "Launcher", "validate_request",
           "render_args", "params_diff", "JOB_OPS", "NAME_RE", "MAX_PARAMS_BYTES"]

NAME_RE = re.compile(r"^[a-z0-9_]{1,40}$")               # same rule as submit.sh (+ a length bound)
RUN_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
CKPT_RE = re.compile(r"^[A-Za-z0-9_./-]{1,512}\.npz$")
JOB_RUN_ID_RE = re.compile(r"^\d{8}_\d{6}_[a-z0-9_]{1,40}$")
MAX_PARAMS_BYTES = 4096
JOB_OPS = ("catalog", "prepare", "submit", "list", "log")    # the whole job whitelist
LOG_TAIL_BYTES = 16384


class LaunchError(ValueError):
    """A refused job request; ``code`` is the HTTP status the servers answer with."""

    def __init__(self, message: str, code: int = 400) -> None:
        super().__init__(message)
        self.code = int(code)


# ============================================================================ catalogue
@dataclass(frozen=True)
class Param:
    """One declared job parameter. ``when=(other, values)``: required iff ``params[other]`` is in
    ``values``, refused otherwise. ``required=False``: may be absent (its absence has a documented
    meaning, e.g. no ``restart`` = start from the declared initial condition)."""

    name: str
    kind: str                       # int | float | choice | checkpoint | run
    doc: str
    lo: float | None = None
    hi: float | None = None
    choices: tuple = ()
    required: bool = True
    when: tuple | None = None

    def spec(self) -> dict:
        """JSON description of the parameter (empty fields omitted), for the catalogue."""
        return {k: v for k, v in dict(name=self.name, kind=self.kind, doc=self.doc, lo=self.lo,
                                      hi=self.hi, choices=list(self.choices), required=self.required,
                                      when=list(self.when) if self.when else None).items()
                if v not in (None, [])}


@dataclass(frozen=True)
class Template:
    """One whitelisted job template: a submit script and its declared parameters."""
    key: str
    script: str                     # relative to the repository root
    title: str
    doc: str
    params: tuple
    examples: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def param(self, name: str) -> Param | None:
        """The declared parameter named ``name``, else None."""
        return next((p for p in self.params if p.name == name), None)

    def spec(self) -> dict:
        """JSON description of the template and its parameters, for the catalogue."""
        return dict(key=self.key, script=self.script, title=self.title, doc=self.doc,
                    params=[p.spec() for p in self.params],
                    examples={k: dict(v) for k, v in self.examples.items()})


_RESTART = Param("restart", "checkpoint", "checkpoint to resume from (absent: fresh start from the "
                 "declared initial condition); numerics must match or the job refuses it",
                 required=False)
_PARENT = Param("parent", "run", "run this job derives from (recorded in the telemetry header)",
                required=False)
_ORBIT = ("solution", ("rpo", "ppo"))

CATALOG: dict[str, Template] = {t.key: t for t in (
    Template(
        "studio_ks", "slurm/templates/studio_ks.sh", "Kuramoto–Sivashinsky DNS (live)",
        "u_t = −u u_x − u_xx − u_xxxx on (0, L), RealFourier(N), on a CPU node. Keeps a bounded history "
        "of states to guess near-recurrences (T, ℓ) when the Studio asks for a checkpoint.",
        (Param("L", "float", "domain length", 1.0, 1000.0),
         Param("N", "int", "Fourier modes (even)", 8, 1024),
         Param("dealias", "float", "dealiasing factor", 1.0, 3.0),
         Param("scheme", "choice", "time scheme (Newton seeds need an RK scheme)",
               choices=("RK111", "RK222", "RK443", "RKSMR", "RKGFY", "SBDF1", "SBDF2", "SBDF3",
                        "SBDF4", "CNAB1", "CNAB2", "MCNAB2", "CNLF2")),
         Param("dt", "float", "time step", 1e-8, 1.0),
         Param("steps", "int", "number of steps", 1, 10**8),
         Param("every", "int", "telemetry call period (steps)", 1, 10**6),
         Param("seed", "int", "random seed of the initial condition", 0, 2**31 - 1),
         Param("amplitude", "float", "amplitude of the random initial modes", 0.0, 100.0),
         Param("rec_every", "float", "time between two stored states (recurrence history)", 1e-6, 1e3),
         Param("rec_Tmax", "float", "longest period searched for near-recurrences", 1e-3, 1e5),
         _RESTART, _PARENT),
        {"ks_L22_turbulent": dict(L=22.0, N=32, dealias=1.5, scheme="RK443", dt=0.02, steps=200000,
                                  every=10, seed=1, amplitude=0.5, rec_every=0.1, rec_Tmax=60.0)}),
    Template(
        "studio_rb2d", "slurm/templates/studio_rb2d.sh", "Rayleigh–Bénard 2D (live, GPU)",
        "studio/demo_rb2d.py on one GPU (the template's partition), RK443, fused F; Pr = 1, Lx = 4.",
        (Param("nx", "int", "Fourier modes in x", 16, 4096),
         Param("nz", "int", "Chebyshev modes in z", 8, 1024),
         Param("ra", "float", "Rayleigh number", 1.0, 1e12),
         Param("dt", "float", "time step", 1e-9, 1.0),
         Param("steps", "int", "number of steps", 1, 10**8),
         Param("every", "int", "telemetry call period (steps)", 1, 10**6),
         _RESTART, _PARENT),
        {"rb2d_ra1e6": dict(nx=256, nz=64, ra=1e6, dt=1e-5, steps=20000, every=20)}),
    Template(
        "studio_newton", "slurm/templates/studio_newton.sh", "Newton–Krylov–hookstep from a checkpoint",
        "conatus.ecs search seeded by a live state (KS). Every tolerance, radius and budget is "
        "declared; the model (L, N, dealias, scheme) must match the checkpoint or the job refuses it.",
        (Param("checkpoint", "checkpoint", "seed state (written by the run on a Studio command)"),
         Param("solution", "choice", "invariant solution sought",
               choices=("equilibrium", "travelling_wave", "rpo", "ppo")),
         Param("subspace", "choice", "equilibria are sought in R u = u: the seed is projected "
               "(distance reported)", choices=("odd",), when=("solution", ("equilibrium",))),
         Param("L", "float", "domain length (must match the run)", 1.0, 1000.0),
         Param("N", "int", "Fourier modes (must match the run)", 8, 1024),
         Param("dealias", "float", "dealiasing factor (must match the run)", 1.0, 3.0),
         Param("scheme", "choice", "RK scheme of the flow map (must match the run)",
               choices=("RK111", "RK222", "RK443", "RKSMR", "RKGFY")),
         Param("c0", "float", "initial drift velocity", -100.0, 100.0,
               when=("solution", ("travelling_wave",))),
         Param("T0", "float", "initial period (near-recurrence guess)", 1e-6, 1e5, when=_ORBIT),
         Param("ell0", "float", "initial shift ℓ", -1000.0, 1000.0, when=_ORBIT),
         Param("nsteps", "int", "time steps per period of the flow map (dt = T / nsteps)", 1, 10**6,
               when=_ORBIT),
         Param("tol", "float", "Newton tolerance on ‖R‖/scale", 1e-16, 1e-2),
         Param("max_iterations", "int", "Newton iteration budget", 1, 1000),
         Param("gmres_rtol", "float", "GMRES relative tolerance", 1e-14, 0.5),
         Param("gmres_maxiter", "int", "GMRES Krylov dimension", 1, 2000),
         Param("delta0", "float", "initial trust radius", 1e-12, 1e6),
         Param("delta_min", "float", "minimal trust radius", 1e-16, 1e6),
         Param("delta_max", "float", "maximal trust radius", 1e-12, 1e9),
         _PARENT),
        # values (benchmarks/gates/j1_ks.py), offered as an explicit preset in the page
        {"j1_gate_tolerances": dict(tol=1e-11, max_iterations=60, gmres_rtol=1e-4, gmres_maxiter=80,
                                    delta0=0.5, delta_min=1e-10, delta_max=5.0)}),
)}


# ============================================================================ validation
def _number(p: Param, v: Any) -> float | int:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise LaunchError(f"parameter {p.name!r} must be a number, got {v!r}")
    if p.kind == "int" and not isinstance(v, int):
        if isinstance(v, float) and v.is_integer():
            v = int(v)                                  # JSON 32.0 -> 32 (same value, no change)
        else:
            raise LaunchError(f"parameter {p.name!r} must be an integer, got {v!r}")
    try:                        # a JSON integer of 400 digits overflows float(): refused like inf
        finite = math.isfinite(v)
    except OverflowError:
        finite = False
    if not finite:
        raise LaunchError(f"parameter {p.name!r} must be finite, got {str(v)[:40]}")
    if (p.lo is not None and v < p.lo) or (p.hi is not None and v > p.hi):
        raise LaunchError(f"parameter {p.name!r} = {v!r} outside [{p.lo:g}, {p.hi:g}]")
    return v if p.kind == "int" else float(v)


def _checkpoint(p: Param, v: Any, ckpt_dir: Path | None) -> str:
    if not isinstance(v, str) or not CKPT_RE.match(v) or ".." in v.split("/"):
        raise LaunchError(f"parameter {p.name!r}: a checkpoint path *.npz (letters, digits, _./-) "
                          f"is required, got {str(v)[:80]!r}")
    if ckpt_dir is not None:                            # only where the files are (relay, server)
        if not v.startswith("/"):
            # a relative path was checked against the cwd of the relay / server, but the job
            # opens it from the repository root (submit.sh's cwd): two different files
            raise LaunchError(f"parameter {p.name!r}: give the absolute path of the checkpoint "
                              f"(as in its checkpoint_written event), got {v!r}")
        real, root = Path(v).resolve(), ckpt_dir.resolve()
        if root not in real.parents:
            raise LaunchError(f"parameter {p.name!r}: {v} is not inside the checkpoint directory {root}")
        if not real.is_file():
            raise LaunchError(f"parameter {p.name!r}: checkpoint {v} does not exist")
    return v


def validate_request(template: Any, name: Any, params: Any, *,
                     ckpt_dir: Path | None = None) -> tuple[Template, str, dict]:
    """Check a job request against the catalogue. Structural checks everywhere (hub included);
    file checks (checkpoint inside ``ckpt_dir`` and existing) only when ``ckpt_dir`` is given."""
    if not isinstance(template, str) or template not in CATALOG:
        raise LaunchError(f"unknown template {str(template)[:60]!r}; known: {sorted(CATALOG)}")
    tpl = CATALOG[template]
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise LaunchError(f"job name must match [a-z0-9_]{{1,40}}, got {str(name)[:60]!r}")
    if not isinstance(params, dict):
        raise LaunchError("params must be a JSON object")
    if len(json.dumps(params)) > MAX_PARAMS_BYTES:
        raise LaunchError(f"params larger than {MAX_PARAMS_BYTES} bytes")
    unknown = sorted(set(params) - {p.name for p in tpl.params})
    if unknown:
        raise LaunchError(f"unknown parameter(s) {unknown} for {template}")
    out: dict[str, Any] = {}
    for p in tpl.params:
        applies = True
        if p.when is not None:
            applies = params.get(p.when[0]) in p.when[1]
        present = p.name in params and params[p.name] is not None
        if not applies:
            if present:
                raise LaunchError(f"parameter {p.name!r} does not apply when "
                                  f"{p.when[0]} = {params.get(p.when[0])!r}")
            continue
        if not present:
            if p.required:
                raise LaunchError(f"parameter {p.name!r} is required (no default): {p.doc}")
            continue
        v = params[p.name]
        if p.kind in ("int", "float"):
            out[p.name] = _number(p, v)
        elif p.kind == "choice":
            if v not in p.choices:
                raise LaunchError(f"parameter {p.name!r} must be one of {list(p.choices)}, got {v!r}")
            out[p.name] = v
        elif p.kind == "checkpoint":
            out[p.name] = _checkpoint(p, v, ckpt_dir)
        elif p.kind == "run":
            if not isinstance(v, str) or not RUN_REF_RE.match(v):
                raise LaunchError(f"parameter {p.name!r} must be a run reference, got {str(v)[:60]!r}")
            out[p.name] = v
    if template == "studio_newton" and not (out["delta_min"] <= out["delta0"] <= out["delta_max"]):
        raise LaunchError("trust radii must satisfy delta_min <= delta0 <= delta_max")
    if template == "studio_ks" and out["N"] % 2:
        raise LaunchError("N must be even (RealFourier pairs)")
    return tpl, name, out


def render_args(tpl: Template, params: Mapping[str, Any]) -> list[str]:
    """``--name value`` in catalogue order; floats with repr (exact round trip)."""
    args: list[str] = []
    for p in tpl.params:
        if p.name in params:
            v = params[p.name]
            args += [f"--{p.name}", repr(v) if isinstance(v, float) else str(v)]
    return args


def params_diff(parent: Mapping[str, Any] | None, child: Mapping[str, Any]) -> dict:
    """{name: [parent value, child value]} for every parameter that differs (None = absent)."""
    if not parent:
        return {}
    keys = sorted(set(parent) | set(child))
    return {k: [parent.get(k), child.get(k)] for k in keys if parent.get(k) != child.get(k)}


# ============================================================================ launcher
class Launcher:
    """Runs on the machine that has ``slurm/submit.sh`` (local server, relay)."""

    def __init__(self, root: str | Path, tmp: str | Path | None = None, *, allow_submit: bool,
                 submit_script: str | Path | None = None, max_per_hour: int = 20,
                 clock=time.time) -> None:
        self.root = Path(root)
        self.tmp = Path(tmp) if tmp is not None else Path(os.environ.get("CN_TMP", self.root / "scratch"))
        self.allow_submit = bool(allow_submit)
        self.submit_script = Path(submit_script) if submit_script else self.root / "slurm" / "submit.sh"
        self.max_per_hour = int(max_per_hour)
        self.ckpt_dir = self.tmp / "checkpoints"
        self.registry = self.tmp / "studio" / "jobs.jsonl"
        self.audit_path = self.tmp / "studio" / "audit.jsonl"
        self.clock = clock
        self._submitted: list[float] = []

    def explain(self) -> str:
        """One-line summary of every launcher knob (submission, rate, paths, templates)."""
        return (f"Launcher(root={self.root}, submit via {self.submit_script}, submission "
                f"{'ENABLED' if self.allow_submit else 'disabled (start with --allow-submit)'}, "
                f"<= {self.max_per_hour}/hour, templates {sorted(CATALOG)}, checkpoints under "
                f"{self.ckpt_dir}, registry {self.registry}, audit {self.audit_path})")

    # -------------------------------------------------------------- dispatcher (server, relay)
    def handle(self, op: str, args: Mapping[str, Any], *, user: str,
               slurm: Mapping[str, Any] | None = None) -> dict:
        """One whitelisted job operation. Raises :class:`LaunchError` on refusal."""
        if op not in JOB_OPS:
            raise LaunchError(f"unknown job operation {op!r}; known: {list(JOB_OPS)}")
        if not isinstance(args, Mapping):
            raise LaunchError("job arguments must be a JSON object")
        if op == "catalog":
            return {"templates": {k: t.spec() for k, t in CATALOG.items()},
                    "allow_submit": self.allow_submit}
        if op == "prepare":
            return self.prepare(args.get("template"), args.get("name"), args.get("params"),
                                parent_params=args.get("parent_params"), user=user)
        if op == "submit":
            return self.submit(args.get("template"), args.get("name"), args.get("params"),
                               digest=args.get("digest"), confirm=args.get("confirm"),
                               parent_params=args.get("parent_params"), user=user)
        if op == "list":
            return {"jobs": self.jobs(slurm)}
        return self.log(args.get("run_id"), args.get("stream", "out"))

    # -------------------------------------------------------------- prepare / submit
    def _render(self, template, name, params, parent_params) -> dict:
        tpl, name, clean = validate_request(template, name, params, ckpt_dir=self.ckpt_dir)
        if parent_params is not None and (not isinstance(parent_params, dict)
                                          or len(json.dumps(parent_params)) > MAX_PARAMS_BYTES):
            raise LaunchError(f"parent_params must be a JSON object of at most {MAX_PARAMS_BYTES} bytes")
        script_path = self.root / tpl.script
        try:
            script = script_path.read_text()
        except OSError as err:
            raise LaunchError(f"template script {tpl.script} unreadable: {err}", 500) from err
        args = render_args(tpl, clean)
        digest = hashlib.sha256(json.dumps(
            {"script_path": tpl.script, "script": script, "name": name, "args": args},
            sort_keys=True).encode()).hexdigest()
        command = ["slurm/submit.sh", tpl.script, name, "--", *args]
        return dict(template=tpl.key, script_path=tpl.script, script=script, name=name,
                    params=clean, args=args, command=command, digest=digest,
                    diff=params_diff(parent_params, clean))

    def prepare(self, template, name, params, *, parent_params=None, user: str) -> dict:
        """Validate and render; ``submit.sh --dry-run`` shows the exact sbatch line. Writes nothing
        but the audit line (submit.sh's dry run writes no directory, script copy or registry)."""
        try:
            return self._prepare(template, name, params, parent_params, user)
        except LaunchError as err:                      # every refusal is audited
            self._audit(op="prepare", user=user, result=f"refused: {err}", template=template,
                        name=name, params=params)
            raise

    def _prepare(self, template, name, params, parent_params, user: str) -> dict:
        out = self._render(template, name, params, parent_params)
        try:
            r = subprocess.run(["bash", str(self.submit_script), out["script_path"], out["name"],
                                "--dry-run", "--", *out["args"]], cwd=self.root,
                               capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as err:
            raise LaunchError(f"submit.sh --dry-run failed to run: {err}", 500) from err
        if r.returncode != 0:
            raise LaunchError(f"submit.sh --dry-run refused: {(r.stdout + r.stderr).strip()[-400:]}")
        out["dry_run"] = r.stdout.strip()
        out["allow_submit"] = self.allow_submit
        self._audit(op="prepare", user=user, result="ok", template=out["template"], name=out["name"],
                    params=out["params"], digest=out["digest"])
        return out

    def submit(self, template, name, params, *, digest, confirm, parent_params=None,
               user: str) -> dict:
        """Submit what ``prepare`` showed, after an explicit confirmation. Never retried."""
        try:
            return self._submit(template, name, params, digest, confirm, parent_params, user)
        except LaunchError as err:                      # every refusal is audited
            if err.code not in (500, 504):              # (failures are audited where they happen)
                self._audit(op="submit", user=user, result=f"refused: {err}", template=template,
                            name=name, params=params, confirm=confirm)
            raise

    def _submit(self, template, name, params, digest, confirm, parent_params, user: str) -> dict:
        if not self.allow_submit:
            raise LaunchError("job submission is disabled here (start the server / relay with "
                              "--allow-submit)", 403)
        if confirm is not True:
            raise LaunchError("submission requires an explicit confirmation (confirm: true)", 409)
        out = self._render(template, name, params, parent_params)
        if not isinstance(digest, str) or digest != out["digest"]:
            raise LaunchError("the job differs from the prepared one (digest mismatch): prepare it "
                              "again and check the script before submitting", 409)
        now = self.clock()
        self._submitted = [t for t in self._submitted if now - t < 3600]
        if len(self._submitted) >= self.max_per_hour:
            raise LaunchError(f"more than {self.max_per_hour} submissions in the last hour", 429)
        self._submitted.append(now)
        try:
            r = subprocess.run(["bash", str(self.submit_script), out["script_path"], out["name"],
                                "--", *out["args"]], cwd=self.root, capture_output=True, text=True,
                               timeout=120)
        except (OSError, subprocess.TimeoutExpired) as err:
            self._audit(op="submit", user=user, result=f"submit.sh did not complete: {err}",
                        template=out["template"], name=out["name"], params=out["params"])
            raise LaunchError(f"submit.sh did not complete ({err}); state unknown, not retried", 504) from err
        text = r.stdout + r.stderr
        m_run = re.search(r"^run_id\s*:\s*(\S+)", r.stdout, re.M)
        m_job = re.search(r"^jobid\s*:\s*(\S+)", r.stdout, re.M)
        if r.returncode != 0 or not m_run:
            self._audit(op="submit", user=user, result=f"submit.sh failed: {text.strip()[-400:]}",
                        template=out["template"], name=out["name"], params=out["params"])
            raise LaunchError(f"submit.sh failed: {text.strip()[-400:]}", 500)
        entry = dict(wall=time.time(), user=user, template=out["template"], name=out["name"],
                     params=out["params"], parent=out["params"].get("parent"), diff=out["diff"],
                     run_id=m_run.group(1), jobid=m_job.group(1) if m_job else None,
                     digest=out["digest"], command=out["command"])
        private_dir(self.registry.parent)
        with open_private(self.registry, "a") as fh:     # 0600: user names, commands (F12)
            fh.write(json.dumps(entry) + "\n")
        self._audit(op="submit", user=user, result="submitted", template=out["template"],
                    name=out["name"], params=out["params"], run_id=entry["run_id"], jobid=entry["jobid"])
        return {"ok": True, **entry, "stdout": r.stdout.strip()}

    # -------------------------------------------------------------- jobs and logs
    def _entries(self) -> list[dict]:
        try:
            lines = self.registry.read_text().splitlines()[-200:]
        except OSError:
            return []
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def jobs(self, slurm: Mapping[str, Any] | None = None) -> list[dict]:
        """Studio-submitted jobs, newest first, with a state from STATUS.json (finished) or the
        last squeue snapshot (queued / running); otherwise the state is said to be unknown."""
        queue = {str(j.get("id")): j for j in (slurm or {}).get("jobs", [])}
        out = []
        for e in reversed(self._entries()):
            status = self.tmp / "output" / e["run_id"] / "STATUS.json"
            state, exit_code = "not in the queue snapshot (state unknown)", None
            try:
                st = json.loads(status.read_text())
                exit_code = st.get("exit_code")
                state = "COMPLETED" if exit_code == 0 else f"FAILED (exit {exit_code})"
            except (OSError, ValueError):
                first = str(e.get("jobid") or "").split(",")[0]
                if first in queue:
                    state = queue[first].get("state", "?")
            logs = sorted(p.name for p in (self.tmp / "logs").glob(f"{e['run_id']}_*.out"))
            out.append({**{k: e.get(k) for k in ("wall", "user", "template", "name", "params",
                                                 "parent", "diff", "run_id", "jobid")},
                        "state": state, "exit_code": exit_code, "log": bool(logs)})
        return out

    def log(self, run_id: Any, stream: Any = "out") -> dict:
        """Tail (<= 16 KiB) of the Slurm log of a Studio-submitted job; nothing else is readable."""
        if not isinstance(run_id, str) or not JOB_RUN_ID_RE.match(run_id):
            raise LaunchError(f"bad job run_id {str(run_id)[:60]!r}")
        if stream not in ("out", "err"):
            raise LaunchError("stream must be 'out' or 'err'")
        if run_id not in {e.get("run_id") for e in self._entries()}:
            raise LaunchError(f"{run_id} was not submitted from the Studio", 404)
        files = sorted((self.tmp / "logs").glob(f"{run_id}_*.{stream}"))
        if not files:
            return {"run_id": run_id, "stream": stream, "text": "", "note": "no log yet (queued?)"}
        p = files[-1]
        size = p.stat().st_size
        with p.open("rb") as fh:
            fh.seek(max(0, size - LOG_TAIL_BYTES))
            text = fh.read().decode("utf-8", "replace")
        return {"run_id": run_id, "stream": stream, "file": p.name, "size": size,
                "truncated": size > LOG_TAIL_BYTES, "text": text}

    def _audit(self, **entry: Any) -> None:
        entry = {"wall": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry}
        try:
            private_dir(self.audit_path.parent)
            with open_private(self.audit_path, "a") as fh:   # 0600 (F12)
                fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
        except OSError as err:                          # never lost: echoed
            print(f"STUDIO AUDIT (file unwritable: {err}) {json.dumps(entry, default=str)}", flush=True)

