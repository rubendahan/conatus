# The expert layer

Under every native model there is an explicit problem that you can read, modify or write yourself:
bases with their sizes, tau terms, the left and right sides of each equation.

```python
import conatus as cn

m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1)    # a native model
comp = m.compile(N=(64, 32))     # the native model at a resolution
comp.problem                      # the expert problem (IVP, EVP, LBVP or NLBVP), as written by Conatus
comp.fields["T"]                  # its unknowns, taus included
comp.ncc                          # data of the given (non-constant) fields
print(comp.explain())             # every automatic decision

plan = comp.discretize()          # the discrete problem: groups, matrices, rank report
sim = m.simulate(N=(64, 32), scheme="RK443", dt=1e-3)
sim.solver, sim.plan              # the same objects behind a Simulation
```

Everything in the expert layer is imported from one place:

```python
import conatus.expert as ex       # or: from conatus.expert import IVP, Chebyshev, ...
```

## When to use it

The native layer covers most problems. Use the expert layer for the cases the native layer leaves
to you:

- conditions that couple several unknowns, or act on one component of a vector (free slip);
- symmetry subspaces, where the parity of each tau must be declared by hand;
- first-order formulations chosen by hand, or several equations sharing the taus of one unknown;
- bases outside the geometry catalogue (Legendre, Jacobi, mapped Chebyshev for infinite lines);
- options of the discretisation and of the time stepper (below).

A name that exists in both layers means the same thing in both: `cn.discretize(problem)` is
`conatus.expert.discretize(problem)`. `conatus.expert.explain()` lists what the import contains
and any name that two modules would export with different meanings.

## Options of the discretisation and of the time stepper

| Option | Where | What it does |
|---|---|---|
| `assembly=` | `discretize(problem, assembly=...)`, `m.discretize(N, assembly=...)` | how the matrices are assembled (the default is printed by `plan.explain()`) |
| `ncc=` | `discretize(problem, ncc={U: lambda y: 1 - y**2})` | values of non-constant coefficients |
| `startup=` | `IVPSolver`, `simulate` | start-up of multistep schemes ({doc}`timesteppers`) |
| `reduced=` | `IVPSolver` | solve in a declared symmetry subspace |
| `complexify=True` | `IVPSolver`, `simulate` | cos/sin pairs solved as complex systems of half size |
| `decouple=True` | `IVPSolver` | block-diagonal groups split into independent systems |
| `time_parameter="t"` | `IVPSolver` | a `Parameter` set to the stage time in the right-hand side |
| `backend=`, `f_backend=` | `IVPSolver`, `simulate` | CPU or GPU; evaluator or fused kernels ({doc}`gpu`) |

You supply these options explicitly, and `explain()` lists the ones in use.

The next pages describe the expert objects: {doc}`operators` (fields, operators, problem
classes, linearisation and adjoints), {doc}`bases` (bases, dealiasing, symmetry subspaces) and
{doc}`taus` (tau terms written by hand, and the rank report).
