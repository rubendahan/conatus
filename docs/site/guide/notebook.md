# Notebooks

In Jupyter, Conatus objects display as formatted tables and cards.

Conatus objects display as formatted HTML in Jupyter. In a terminal, `print(obj.explain())` gives
the same content as text. Nothing about the notebook is required: neither IPython nor ipywidgets
is a dependency, and `__repr__` / `explain()` are unchanged.

| Object | What the cell shows |
|---|---|
| `Problem` (IVP, LBVP, EVP, NLBVP) | variables with tensor type and domain, parameters with values, equations as LHS (implicit) \| RHS (explicit), declared tau terms, the IR rank count with its status badge |
| `RankReport` | counts per non-periodic direction, CONSISTENT / CHECK / FAIL as computed by the report |
| `Plan` (Cartesian) | the plan's `explain()` and a rank table for the first groups ("k of n shown; `plan.check()` checks all") |
| `PolarPlan`, `SpherePlan` | their `explain()` |
| `EVPResult` | leading eigenvalues with backward residuals, and a small inline SVG of the spectrum (Re, Im) |
| `IVPSolver`, `CFL`, `EmbeddedError`, `TauWatch`, `NewtonSolver`, `NLBVPSolution` | `explain()` laid out as a card; Newton shows `converged` as computed |
| `VerificationReport`, `Check` | pass / fail table |
| runs of `conatus.runs` | their own tables (`Run`, comparisons) |
| `DedalusTranslation` (`cn.from_dedalus`) | the line-by-line report, colour-coded by status |
| `DoctorReport` (`cn.doctor()`), `ResolutionReport` | tables of checks and of spectral tails |

A status badge appears only where the object computes that status itself. The EVP table shows
residuals but no invented pass threshold.

## Parameter widgets (optional)

```python
from conatus.expert import (IVP, CartesianCoordinates, Chebyshev, Domain, Field, Parameter, dt,
                            interp, lap)
from conatus.notebook import parameter_widgets

cs = CartesianCoordinates("x"); x = cs["x"]
u = Field("u", Domain(cs, [Chebyshev(x, 24, (-1, 1), dealias=1)]))
a = Parameter("a", 1.0)
pb = IVP([u], name="heat 1D")
pb.add_equation((dt(u) - lap(u), -a * u))
pb.add_bc(interp(u, x, "left"), interp(u, x, "right"))

w = parameter_widgets(pb, on_change=lambda values: print(values))
w          # one text box per Parameter of the problem (the EVP eigenvalue excluded)
```

If ipywidgets is not installed, `parameter_widgets` returns `None` and says so with a
`UserWarning`. Nothing else changes. The widgets **never modify the IR or a plan**: IR objects are
immutable. Each edit passes the full `{name: value}` dict to your `on_change`, which re-discretizes
or re-solves as you decide, and `w.values()` returns the same dict. The boxes are plain text boxes
rather than sliders, because a slider range would be an invented bound. Text that is not a number
is refused (`ParameterError`) in the widget, without calling your function.

Implementation: `python/conatus/notebook.py` (HTML with escaped text, inline CSS readable in light
and dark themes, no JavaScript).
