"""Planned adjoint of the fused spherical-shell right-hand side: F'(X)† W along the fused path.

The planned tangent of ``f_backend='fused'`` (``timestep/sphere_fused``) evaluates each linearized
right-hand side ``le`` (linear in the direction fields d) as one chain of planned stages:

    d (coefficients) --radial GEMM ('val' | 'over_r', 'dr')--> mixed space --[spin gradient]-->
    --angular GEMMs + one FFT per spin pair--> canonical grid components ("parts", phi last)
    --product tree (one straight-line program per canonical output component)--> output parts
    --FFT + angular projection--> mixed --radial projection--> coefficients --R (gather_rhs)--> rows

This module writes the exact adjoint of that chain for the real pairing <a, b> = Re(a^H b), stage
by stage, in the same representation (canonical parts, real GEMMs on interleaved (re, im) data,
m-chunks on the existing degrees, batched FFTs on the contiguous phi axis):

* rows -> coefficients: R^H; restriction of the equation layout (adjoint of ``embed_coeffs``);
* radial projection x @ F -> z @ F^T; radial syntheses c @ [B_val | B_over_r | B_dr] -> one GEMM
  by the transposed concatenation (all the leaves of one direction field at once);
* angular projection / synthesis (real stacks per m, chunked) -> their transposes, chunked alike;
* phi FFTs (``norm='forward'``): fft -> ifft with backward normalisation, ifft -> unnormalised fft;
  the spectrum assembly of a spin pair (m = 0 symmetrised (V + conj W) / 2, the modes -m from the
  partner by conjugation) and of a real component (irfft of the half spectrum) -> their adjoints,
  so the m = 0 projection of the planned F is part of the transposed chain;
* the product tree: reverse mode over its SSA program (``_ProductPlan``): passive nodes (state,
  base flow, time factors a_g(t), r^k) evaluated forward, cotangents accumulated backward
  (mul by a passive K -> conj(K), conj -> conj, Re and real nodes -> real part); on the GPU this is
  one generated elementwise kernel per product tree (inputs: passive loads and output
  cotangents; outputs: the cotangents of the active loads);
* the spin-basis gradient (G_-+ in l, d/dr, frame terms) -> its transpose (one generated kernel on
  the GPU, the array code on the CPU).

Scope (else the equation keeps the explicit walk of ``sphere_rhs``, declared by ``explain()``):
shell, RealFourier in phi, a product tree at the top of ``le``, active leaves that are direction
fields d or grad(d) of a direction field on the full domain. The passive values come from the
planned evaluator itself (same leaves, same constants, same time factors as the planned tangent).
Same adjoint as the explicit walk up to rounding (the summation order changes), checked by
dot-product tests stage by stage (the development test suite) and end to end.
"""
from __future__ import annotations

import numpy as np

from ..discretize.sphere_eval import _flip, comps
from ..discretize.sphere_ops import spin_of
from ..discretize.spin import canon as _canon
from ..discretize.spin import real_component as _real_comp
from ..ir import expr as E

__all__ = ["PlanRefused", "PlannedSphereTranspose", "check_scope"]


class PlanRefused(Exception):
    """An equation outside the planned scope (it keeps the explicit walk; the reason is kept)."""


def _unbroadcast(a, shape: tuple):
    from .sphere_rhs import unbroadcast
    return unbroadcast(a, shape)


# ----------------------------------------------------------------------------------------------
# reverse mode over a product plan
# ----------------------------------------------------------------------------------------------
_RKERNELS: dict = {}


class _ReversePlan:
    """Transpose of a ``_ProductPlan`` that is linear in its active leaves (indices ``active``):
    output-part cotangents -> active-load cotangents, for the real pairing."""

    def __init__(self, plan, active: set) -> None:
        self.p = plan
        nodes = plan.nodes
        self.live = plan._live()
        act: dict = {}
        for i in self.live:                              # SSA: arguments have smaller indices
            op, args, _ = nodes[i]
            if op == "ld":
                a = args[0] in active
            elif op == "mul":
                a0, a1 = act[args[0]], act[args[1]]
                if a0 and a1:
                    raise PlanRefused("product of two direction-dependent factors")
                a = a0 or a1
            elif op == "add":
                a0, a1 = act[args[0]], act[args[1]]
                if a0 != a1:
                    raise PlanRefused("sum of a direction-dependent and a constant term (affine)")
                a = a0
            elif op in ("conj", "re"):
                a = act[args[0]]
            elif op == "pow":
                a = act[args[0]]
                if a:
                    raise PlanRefused("power of a direction-dependent factor")
            elif op == "fn":
                a = act[args[1]]
                if a:
                    raise PlanRefused("pointwise function of a direction-dependent factor")
            else:                                        # num, sc
                a = False
            act[i] = a
        for c, i in plan.out:
            if i is not None and not act[i]:
                raise PlanRefused(f"output component {c} independent of the direction (affine)")
        self.act = act
        self.passive = [i for i in self.live if not act[i]]
        self.active_nodes = [i for i in self.live if act[i]]
        self.loads = sorted({nodes[i][1] for i in self.active_nodes if nodes[i][0] == "ld"})
        self.ploads = sorted({nodes[i][1] for i in self.passive if nodes[i][0] == "ld"})
        self._kernel = None

    # ---------------------------------------------------------------- array execution (CPU, oracle)
    def run_arrays(self, leaf_parts: list, scalars: list, obar: dict, rk, xp) -> dict:
        nodes, real = self.p.nodes, self.p._real
        v: dict = {}
        for i in self.passive:
            op, args, _ = nodes[i]
            if op == "ld":
                v[i] = leaf_parts[args[0]][args[1]]
            elif op == "sc":
                v[i] = scalars[args[0]]
            elif op == "num":
                v[i] = args[0] if args[1] == 0 else complex(*args)
            elif op == "mul":
                v[i] = v[args[0]] * v[args[1]]
            elif op == "add":
                v[i] = v[args[0]] + v[args[1]]
            elif op == "conj":
                v[i] = v[args[0]] if real(args[0]) else xp.conj(v[args[0]])
            elif op == "pow":
                v[i] = v[args[0]] ** args[1]
            elif op == "re":
                v[i] = v[args[0]].real
            else:
                v[i] = getattr(xp, args[0])(v[args[1]])
        bar: dict = {}

        def add(i, b):
            bar[i] = b if i not in bar else bar[i] + b
        for c, i in self.p.out:
            if i is None:
                continue
            ob = obar[c]
            add(i, ob if rk is None else ob * rk)
        out: dict = {}
        for i in reversed(self.active_nodes):
            b = bar.pop(i, None)
            if b is None:
                continue
            op, args, isreal = nodes[i]
            if isreal and xp.iscomplexobj(b):
                b = b.real                               # adjoint of the inclusion R -> C
            if op == "ld":
                out[args] = b
            elif op == "add":
                add(args[0], b)
                add(args[1], b)
            elif op == "mul":
                pa, aa = (args[0], args[1]) if self.act[args[1]] else (args[1], args[0])
                K = v[pa]
                add(aa, (xp.conj(K) if np.iscomplexobj(K) else K) * b)
            elif op == "conj":
                add(args[0], b if real(args[0]) else xp.conj(b))
            elif op == "re":
                add(args[0], b)
        return out

    # ---------------------------------------------------------------- generated kernel (GPU)
    def _src(self, obar_real: list) -> tuple:
        p = self.p
        nodes = p.nodes
        lname = {ld: f"L{ld[0]}_{''.join(map(str, ld[1])) or 's'}" for ld in self.ploads}
        ins = [f"{'float64' if _real_comp(ld[1]) else 'complex128'} {lname[ld]}" for ld in self.ploads]
        ins += [f"{'float64' if r else 'complex128'} ob{j}" for j, r in enumerate(obar_real)]
        ins += [f"float64 s{k}" for k in range(len(p.scalars))]
        if p.rk:
            ins.append("float64 rk")
        outs = [f"{'float64' if _real_comp(c) else 'complex128'} g{li}_{''.join(map(str, c)) or 's'}"
                for li, c in self.loads]
        T = {True: "double", False: "complex<double>"}
        body = []
        for i in self.passive:
            op, args, real = nodes[i]
            if op == "ld":
                e = lname[args]
            elif op == "sc":
                e = f"s{args[0]}"
            elif op == "num":
                e = repr(float(args[0])) if args[1] == 0 else f"complex<double>({args[0]!r}, {args[1]!r})"
            elif op == "mul":
                e = f"v{args[0]} * v{args[1]}"
            elif op == "add":
                e = f"v{args[0]} + v{args[1]}"
            elif op == "conj":
                e = f"v{args[0]}" if p._real(args[0]) else f"conj(v{args[0]})"
            elif op == "pow":
                e = f"pow(v{args[0]}, {args[1]!r})"
            elif op == "re":
                e = f"v{args[0]}.real()"
            else:
                fn, a = args
                e = {"abs": f"fabs(v{a})", "sign": f"(double)((v{a} > 0) - (v{a} < 0))"}.get(fn, f"{fn}(v{a})")
            body.append(f"const {T[real]} v{i} = {e};")
        for i in self.active_nodes:
            body.append(f"complex<double> b{i} = complex<double>(0.0, 0.0);")
        for j, (_c, i) in enumerate(p.out):
            if i is None:
                continue
            e = f"ob{j}" + (" * rk" if p.rk else "")
            body.append(f"b{i} += complex<double>({e});")
        for i in reversed(self.active_nodes):
            op, args, isreal = nodes[i]
            if isreal:
                body.append(f"b{i} = complex<double>(b{i}.real(), 0.0);")
            if op == "add":
                body.append(f"b{args[0]} += b{i};")
                body.append(f"b{args[1]} += b{i};")
            elif op == "mul":
                pa, aa = (args[0], args[1]) if self.act[args[1]] else (args[1], args[0])
                K = f"v{pa}" if p._real(pa) else f"conj(v{pa})"
                body.append(f"b{aa} += {K} * b{i};")
            elif op == "conj":
                body.append(f"b{args[0]} += " + (f"b{i};" if p._real(args[0]) else f"conj(b{i});"))
            elif op == "re":
                body.append(f"b{args[0]} += b{i};")
        ldnode = {nodes[i][1]: i for i in self.active_nodes if nodes[i][0] == "ld"}
        for li, c in self.loads:
            nm = f"g{li}_{''.join(map(str, c)) or 's'}"
            i = ldnode[(li, c)]
            body.append(f"{nm} = b{i}.real();" if _real_comp(c) else f"{nm} = b{i};")
        return ", ".join(ins), ", ".join(outs), "\n".join(body)

    def run_kernel(self, leaf_parts: list, scalars: list, obar: dict, rk, xp, shapes: dict) -> dict:
        p = self.p
        oreal = [_real_comp(c) for c, _ in p.out]
        if self._kernel is None:
            ins, outs, body = self._src(oreal)
            key = (ins, outs, body)
            k = _RKERNELS.get(key)
            if k is None:
                import hashlib
                name = "cn_sphere_prodT_" + hashlib.sha1(repr(key).encode()).hexdigest()[:12]
                k = _RKERNELS[key] = xp.ElementwiseKernel(ins, outs, body, name)
            self._kernel = k
        args = [leaf_parts[li][c] for li, c in self.ploads]
        for c, _ in p.out:
            args.append(obar[c])
        args += [float(s) for s in scalars]
        if p.rk:
            args.append(rk)
        outs = [xp.empty(shapes[ld], dtype=float if _real_comp(ld[1]) else complex) for ld in self.loads]
        self._kernel(*args, *outs)
        return dict(zip(self.loads, outs))


# ----------------------------------------------------------------------------------------------
# transposed planned stages (canonical parts, phi last)
# ----------------------------------------------------------------------------------------------
class _PlannedStagesT:
    """Adjoints of the ``_FastSpace`` stages used by the planned tangent (shell, RealFourier,
    full domain, dealiased grid), on the conventions of ``sphere_fused``."""

    def __init__(self, fast, T) -> None:
        self.F, self.T = fast, T                       # _FastSpace, SphereTransposes
        S = fast._S
        self.S = S
        self.Mp, self.Mt, self.Mr = S.sizes(True)
        self.nm = len(S.ms)
        if 2 * (self.nm - 1) >= self.Mp:
            raise PlanRefused(f"Mphi = {self.Mp} too small for nm = {self.nm} stored modes")
        self._cache: dict = {}

    def _ang_T(self, which: str, y, s: int, xp):
        """Transpose of the forward angular stage ``which`` ('bwd' synthesis / 'fwd' projection)
        of spin s on complex y (nm, ., Mr), chunked on the existing degrees like the forward."""
        T, ms = self.T, self.S.ms
        if which == "bwd":            # transpose of synthesis: projection-like
            return T._ang_apply("fwd", T._ang_H("bwd", ms, s, self.Mt, xp), y, ms, s, xp)
        return T._ang_apply("bwd", T._ang_H("fwd", ms, s, self.Mt, xp), y, ms, s, xp)

    def rad_T(self, ybars: list, whats: tuple, xp):
        """Adjoint of ``coeff_to_mixed_many(c, whats)``: one GEMM by the transposed
        concatenation [B_w1 | B_w2 | ...]^T of the radial syntheses."""
        from ..timestep.sphere_fused import _right_real
        key = ("radT", whats, xp is np)
        M = self._cache.get(key)
        if M is None:
            M = self._cache[key] = xp.ascontiguousarray(self.F._rad(whats, self.Mr, xp).T)
        y = ybars[0] if len(ybars) == 1 else xp.concatenate(ybars, axis=-1)
        return _right_real(xp.asarray(y, dtype=complex), M, xp)

    def m2c_T(self, cbar, xp):
        """Adjoint of the radial projection ``mixed_to_coeff`` (z = x @ F -> x̄ = z̄ @ F^T)."""
        from ..timestep.sphere_fused import _right_real
        key = ("fwdT", xp is np)
        M = self._cache.get(key)
        if M is None:
            M = self._cache[key] = xp.ascontiguousarray(
                xp.asarray(self.S._shell_mat("fwd", "val", self.Mr)).T)
        return _right_real(xp.asarray(cbar, dtype=complex), M, xp)

    def parts_to_mixed_T(self, xbar, rank: int, xp) -> dict:
        """Adjoint of ``parts_to_mixed`` (canonical parts (Mt, Mr, Mp) -> mixed (nc,)*rank +
        (nm, L+1, Mr)): transposed angular projections, then per canonical component the adjoint
        of the mode extraction and of fft(norm='forward') = ifft with backward normalisation."""
        nm, Mp = self.nm, self.Mp
        az = {}
        for c in comps(rank, self.S.nc):
            y = xbar[c] if c else xbar
            az[c] = xp.moveaxis(self._ang_T("fwd", y, spin_of(c), xp), 0, -1)   # (Mt, Mr, nm)
        out = {}
        for c in comps(rank, self.S.nc):
            if _canon(c) != c:
                continue
            f = _flip(c)
            if f == c:                 # rfft(y)[ms] of a real y: ȳ = irfft of the halved spectrum
                H = xp.zeros(az[c].shape[:-1] + (Mp // 2 + 1,), dtype=complex)
                H[..., 0] = az[c][..., 0]
                H[..., 1:nm] = 0.5 * az[c][..., 1:]
                out[c] = xp.fft.irfft(H, n=Mp, axis=-1, norm="backward")
            else:                      # az_c = Fy[m], az_f = conj(Fy[-m])
                Sb = xp.zeros(az[c].shape[:-1] + (Mp,), dtype=complex)
                Sb[..., :nm] = az[c]
                Sb[..., 0] += xp.conj(az[f][..., 0])
                Sb[..., Mp - 1:Mp - nm:-1] = xp.conj(az[f][..., 1:])
                out[c] = xp.fft.ifft(Sb, axis=-1, norm="backward")
        return out

    def mixed_to_parts_T(self, gbar: dict, rank: int, xp):
        """Adjoint of ``mixed_to_parts`` (mixed -> canonical parts): unnormalised fft of each
        part cotangent, the adjoint of the spectrum assembly (m = 0 symmetrised pair, modes -m by
        conjugation; half spectrum of a real component), transposed angular syntheses."""
        nm, Mp = self.nm, self.Mp
        S = self.S
        out = xp.zeros((S.nc,) * rank + (nm, S.L + 1, self.Mr), dtype=complex)
        for c in comps(rank, S.nc):
            if _canon(c) != c:
                continue
            f = _flip(c)
            gb = gbar.get(c)
            if gb is None:
                continue
            if f == c:                 # g = irfft(H, norm='forward'), H[0] = Re V0, H[1:nm] = V
                Hb = xp.fft.rfft(xp.asarray(gb, dtype=float), axis=-1, norm="backward")
                Vb = xp.empty(Hb.shape[:-1] + (nm,), dtype=complex)
                Vb[..., 0] = Hb[..., 0].real
                Vb[..., 1:] = 2.0 * Hb[..., 1:nm]
                pairs = ((c, Vb),)
            else:                      # g = ifft(Sx, norm='forward') = sum_k Sx_k e^{ik phi}
                Sb = xp.fft.fft(gb, axis=-1, norm="backward")
                Vb = xp.empty(Sb.shape[:-1] + (nm,), dtype=complex)
                Wb = xp.empty_like(Vb)
                Vb[..., 0] = 0.5 * Sb[..., 0]
                Vb[..., 1:] = Sb[..., 1:nm]
                Wb[..., 0] = 0.5 * xp.conj(Sb[..., 0])
                Wb[..., 1:] = xp.conj(Sb[..., Mp - 1:Mp - nm:-1])
                pairs = ((c, Vb), (f, Wb))
            for cc, yb in pairs:
                v = self._ang_T("bwd", xp.ascontiguousarray(xp.moveaxis(yb, -1, 0)), spin_of(cc), xp)
                if cc:
                    out[cc] = v
                else:
                    out[...] = v
        return out

    def grad_T(self, gbar, rank: int, xp, kernel: bool):
        """Adjoint of the spin-basis gradient of ``_FusedEvaluator._grad`` (mixed space): cotangent
        (nc,) + (nc,)*rank + (nm, L+1, Mr) -> (Ā, D̄), the cotangents of the 'over_r' and 'dr'
        radial syntheses of the operand."""
        from ..timestep.sphere_fused import _grad_terms
        S = self.S
        nc, ms = S.nc, S.ms
        shape = gbar.shape[1:]
        L1 = shape[rank + 1]
        terms = _grad_terms(rank, nc)
        cs = comps(rank, nc)
        if kernel:
            return self._grad_T_kernel(gbar, rank, xp, terms, cs, L1)
        Ab = xp.zeros(shape, dtype=complex)
        Db = xp.zeros(shape, dtype=complex)
        for oc, ts in terms.items():
            gb = gbar[oc]
            for sign, kind, c in ts:
                tgt = Ab[c] if c else Ab
                if kind == "D":
                    if c:
                        Db[c] += gb
                    else:
                        Db[...] += gb
                    continue
                if kind == "A":
                    tgt += gb if sign > 0 else -gb
                    continue
                G = S.Gstack(-1 if kind == "Gm" else 1, ms, spin_of(c), xp)[:, :L1, None]
                tgt += G * gb if sign > 0 else -(G * gb)
        return Ab, Db

    def _grad_T_kernel(self, gbar, rank, xp, terms, cs, L1):
        from ..timestep.sphere_fused import _cn
        S = self.S
        ms = S.ms
        key = ("gradT", rank, S.nc)
        k = _RKERNELS.get(key)
        if k is None:
            inv: dict = {}
            for oc, ts in terms.items():
                for sign, kind, c in ts:
                    tgt = ("D" if kind == "D" else "A", c)
                    inv.setdefault(tgt, []).append((sign, kind, oc))
            ins = [f"complex128 g{_cn(oc)}" for oc in terms]
            ins += [f"float64 Gm{_cn(c)}, float64 Gp{_cn(c)}" for c in cs]
            outs = [f"complex128 A{_cn(c)}" for c in cs] + [f"complex128 D{_cn(c)}" for c in cs]
            body = []
            for which in ("A", "D"):
                for c in cs:
                    e = None
                    for sign, kind, oc in inv.get((which, c), []):
                        t = {"Gm": f"Gm{_cn(c)} * g{_cn(oc)}", "Gp": f"Gp{_cn(c)} * g{_cn(oc)}",
                             "D": f"g{_cn(oc)}", "A": f"g{_cn(oc)}"}[kind]
                        e = (t if sign > 0 else f"-({t})") if e is None else \
                            f"({e}) {'+' if sign > 0 else '-'} {t}"
                    body.append(f"{which}{_cn(c)} = {e if e is not None else 'complex<double>(0.0, 0.0)'};")
            k = _RKERNELS[key] = xp.ElementwiseKernel(", ".join(ins), ", ".join(outs), "\n".join(body),
                                                      f"cn_sphere_gradT_r{rank}_c{S.nc}")
        shape = gbar.shape[1:]
        Ab = xp.empty(shape, dtype=complex)
        Db = xp.empty(shape, dtype=complex)
        args = [gbar[oc] for oc in terms]
        for c in cs:
            s = spin_of(c)
            args += [S.Gstack(-1, ms, s, xp)[:, :L1, None], S.Gstack(1, ms, s, xp)[:, :L1, None]]
        k(*args, *[Ab[c] if c else Ab for c in cs], *[Db[c] if c else Db for c in cs])
        return Ab, Db


# ----------------------------------------------------------------------------------------------
# the planned transpose of all the linearized right-hand sides
# ----------------------------------------------------------------------------------------------
class _EqPlan:
    __slots__ = ("k", "le", "plan", "rev", "active", "dom", "rank", "shape", "RH")


class PlannedSphereTranspose:
    """F'(X, t)† W along the planned path for the equations in scope (module doc); the others are
    left to the caller's explicit walk (``fallback``: equation index -> reason)."""

    def __init__(self, lin) -> None:
        from ..timestep.sphere_fused import _PRODUCT, _ProductPlan
        self.lin = lin
        fev, lin2 = lin._fused
        self.fev = fev
        self.pev = lin._fused_passive
        S = lin.S
        self.S = S
        self.xp = lin.xp
        self.kernel = self.xp is not np and fev.products == "kernel"
        self.eqs: list = []
        self.fallback: dict = {}
        try:
            self.stages = _PlannedStagesT(fev.S, lin.T)
            stage_refusal = None
        except PlanRefused as why:                     # every equation keeps the walk
            self.stages, stage_refusal = None, str(why)
        full = None
        for i, (le, k, RH, shape) in enumerate(zip(lin2, lin.rpow, lin._RH, lin.host.eq_shapes)):
            if le is None or RH is None:
                continue
            try:
                if stage_refusal is not None:
                    raise PlanRefused(stage_refusal)
                if not isinstance(le, _PRODUCT):
                    raise PlanRefused(f"top node {type(le).__name__} is not a product tree")
                dom = lin.ev.rpow_domain(le.domain) if k else le.domain
                if not all(S.has(dom)):
                    raise PlanRefused("output not on the full (phi, theta, r) domain")
                plan = _ProductPlan(le, S.nc, S.G, bool(k))
                active = set()
                for li, leaf in enumerate(plan.leaves):
                    if not (set(leaf.free_fields()) & lin._dset):
                        continue
                    d = leaf.args[0] if isinstance(leaf, E.Gradient) else leaf
                    if not (d in lin._dset and (leaf is d or isinstance(leaf, E.Gradient))):
                        raise PlanRefused(f"active leaf '{leaf}' is not d or grad(d)")
                    if not all(S.has(d.domain)):
                        raise PlanRefused(f"direction {d} not on the full domain")
                    full = d.domain
                    active.add(li)
                rev = _ReversePlan(plan, active)
            except PlanRefused as why:
                self.fallback[i] = str(why)
                continue
            q = _EqPlan()
            q.k, q.le, q.plan, q.rev, q.active = k, le, plan, rev, active
            q.dom, q.rank, q.shape, q.RH = dom, le.rank, tuple(shape), RH
            self.eqs.append((i, q))
        self._full = full
        self.planned = frozenset(i for i, _ in self.eqs)

    def _rk(self, k: int):
        fS = self.fev.S
        key = ("rpow_plast", self.stages.Mr, k)
        if key not in fS._cache:
            fS._cache[key] = self.xp.asarray(np.ascontiguousarray((fS.ops.grid(self.stages.Mr) ** k)[:, None]))
        return fS._cache[key]

    def transpose(self, data: dict, params: dict, W) -> dict:
        """Cotangents (direction field -> coefficient array) of the planned equations for the
        cotangent W of the rows; ``data``/``params``: the state and time of the linearization."""
        xp, st, S = self.xp, self.stages, self.S
        pev = self.pev
        # per direction field: mixed-space cotangents of its 'val', 'over_r', 'dr' syntheses
        mix: dict = {}

        def addm(d, what, a):
            slot = mix.setdefault(d, {})
            slot[what] = a if what not in slot else slot[what] + a
        part_shape = (st.Mt, st.Mr, st.Mp)
        for _i, q in self.eqs:
            cbar = (q.RH @ W).reshape(q.shape)
            cs = S.coeff_shape(q.dom, q.rank)
            if cs != q.shape:                          # adjoint of embed_coeffs: restriction
                cbar = cbar[tuple(slice(0, n_) for n_ in cs)]
            xbar = st.m2c_T(cbar, xp)
            obar = st.parts_to_mixed_T(xbar, q.rank, xp)
            leaves = [None if li in q.active else pev._parts(pev._ev(leaf, data, params))
                      for li, leaf in enumerate(q.plan.leaves)]
            pev._in_scalars = True
            try:
                scal = [pev._ev(sn, data, params) for sn in q.plan.scalars]
            finally:
                pev._in_scalars = False
            rk = self._rk(q.k) if q.k else None
            shapes = {ld: part_shape for ld in q.rev.loads}
            if self.kernel:
                lbar = q.rev.run_kernel(leaves, scal, obar, rk, xp, shapes)
            else:
                lbar = q.rev.run_arrays(leaves, scal, obar, rk, xp)
            # gather the load cotangents per active leaf, then to mixed space
            per_leaf: dict = {}
            for (li, c), b in lbar.items():
                per_leaf.setdefault(li, {})[c] = _unbroadcast(b, part_shape) if b.shape != part_shape else b
            for li, gb in per_leaf.items():
                leaf = q.plan.leaves[li]
                if isinstance(leaf, E.Gradient):
                    d = leaf.args[0]
                    gx = st.mixed_to_parts_T(gb, d.rank + 1, xp)
                    Ab, Db = st.grad_T(gx, d.rank, xp, self.kernel)
                    addm(d, "over_r", Ab)
                    addm(d, "dr", Db)
                else:
                    addm(leaf, "val", st.mixed_to_parts_T(gb, leaf.rank, xp))
        out = {}
        for d, slot in mix.items():
            whats = tuple(w for w in ("val", "over_r", "dr") if w in slot)
            out[d] = st.rad_T([slot[w] for w in whats], whats, xp)
        return out

    def explain(self) -> str:
        """Which equations take the planned adjoint, and why the others keep the explicit walk."""
        planned = ", ".join(str(i) for i in sorted(self.planned)) or "none"
        txt = (f"planned adjoint of the fused F for equation(s) {planned} (transposed planned stages: "
               "radial GEMMs, chunked angular GEMMs, phi FFTs with the m = 0 symmetrisation, reverse "
               "mode of the product program" + (" as one generated kernel per product tree" if self.kernel else "")
               + ", spin-gradient transpose)")
        if self.fallback:
            txt += "; explicit walk for " + "; ".join(f"eq {i} ({why})" for i, why in sorted(self.fallback.items()))
        return txt


def check_scope(lin) -> str | None:
    """None if the planned transpose can be built for this linearized RHS, else the reason."""
    if lin._fused is None:
        return "f_backend='evaluator' (no planned tangent)"
    if not lin.S.shell or not lin.S.real:
        return "shell with RealFourier in phi only"
    return None

