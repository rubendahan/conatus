# User layer: `import conatus as cn`

```python
import conatus as cn
```

`import conatus` is light: nothing heavy (solvers, compiled core, GPU) is loaded until it is used.
The names below are reached as `cn.<name>`.

## Geometries

A geometry fixes the coordinates, the spectral bases (chosen once per geometry, printed by
`explain()`) and the names of the boundaries used by {meth}`Model.at <conatus.Model.at>`.

```{eval-rst}
.. currentmodule:: conatus

.. autosummary::
   :toctree: _generated/user
   :nosignatures:

   Box
   Channel
   Disk
   Annulus
   Pipe
   Sphere
   Ball
   Shell
   TaylorCouette
   MappedBox
   periodic
   walls
   line
   half_line
   between
```

`cn.Cylinder` is another name for `cn.Pipe`.

## Wall conditions on a vector unknown

Named conditions for `m.at(...)` (guide: {doc}`../guide/boundary_conditions`).

```{eval-rst}
.. currentmodule:: conatus

.. autosummary::
   :toctree: _generated/user
   :nosignatures:

   no_slip
   free_slip
   navier_slip
```

## The model

```{eval-rst}
.. autosummary::
   :toctree: _generated/user
   :nosignatures:

   Model
```

What the model returns:

```{eval-rst}
.. currentmodule:: conatus.model

.. autosummary::
   :toctree: _generated/user
   :nosignatures:

   Simulation
   Linearization
   Compiled
```

## Ready problems: `cn.problems`

Canonical flows written once, each returning an ordinary problem with its handles and the
reference it reproduces.

```{eval-rst}
.. currentmodule:: conatus.problems

.. autosummary::
   :toctree: _generated/user
   :nosignatures:

   channel
   channel_evp
   plane_couette
   plane_couette_evp
   rayleigh_benard
   rayleigh_benard_evp
   taylor_couette
   taylor_couette_evp
   pipe
   pipe_evp
   kuramoto_sivashinsky
   kolmogorov
   swift_hohenberg
   rotating_shell_convection
   rotating_shell_convection_evp
   Ready
```

## The run registry: `cn.runs`

Record runs, find and compare them, reproduce one bit for bit (guide: {doc}`../guide/runs`).

```{eval-rst}
.. currentmodule:: conatus.runs

.. autosummary::
   :toctree: _generated/user
   :nosignatures:

   track
   record
   find
   get
   compare
   lineage
   reproduce
   to_dataframe
   export
   explain
   Run
   RunList
   Registry
```

## Quantities of a run

Series, spectra and profiles of a simulation, what to watch while it runs, and the statistics of a
result (guide: {doc}`../guide/diagnostics`).

```{eval-rst}
.. currentmodule:: conatus

.. autosummary::
   :toctree: _generated/user
   :nosignatures:

   series
   spectrum
   profile
   view
   time_average
   uncertainty
   StopRun
```

`cn.StopRun` is the exception a callback raises to stop a run.

## Utilities

```{eval-rst}
.. currentmodule:: conatus

.. autosummary::
   :toctree: _generated/user
   :nosignatures:

   doctor
   from_dedalus
   parse_equation
   parse_expr
```

## Regions of the complex plane: `cn.box`, `cn.contour`

`m.eigenvalues(N, ..., region=cn.box(re=(a, b), im=(c, d)))` returns the eigenvalues in the box,
with a count estimated by the argument principle from samples of the determinant on the boundary.
`cn.contour.count` and `cn.contour.beyn` take any operator
({doc}`../guide/dynamics/contour`).

```{eval-rst}
.. currentmodule:: conatus

.. autosummary::
   :toctree: _generated/user
   :nosignatures:

   box
   contour.count
   contour.beyn
   contour.CountResult
   contour.BeynResult
   contour.RegionEVPResult
```
