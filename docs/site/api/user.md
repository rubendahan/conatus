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
   periodic
   walls
```

`cn.Cylinder` is another name for `cn.Pipe`.

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
