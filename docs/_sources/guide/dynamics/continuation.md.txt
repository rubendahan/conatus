# Continuation and bifurcations

Branches of solutions followed through folds by pseudo-arclength, with branch points, Hopf points, cycles and codimension-two curves located on the way.

Pseudo-arclength continuation follows a curve of solutions $G(x, p) = 0$ by stepping along its
tangent and correcting with Newton on the bordered system, so that folds are passed rather than
hit. Events (folds, branch points, Hopf points) are located on the curve by bisection on test
functions.

## A nonlinear boundary-value problem

The Bratu problem $u'' + \lambda e^u = 0$, $u(\pm 1) = 0$, written natively and compiled at a
declared resolution:

```{literalinclude} _check.py
:language: python
:start-after: "[cont-model]"
:end-before: "[/cont-model]"
:dedent: 4
```

`dealias=1` is declared here: `exp(u)` is not a polynomial nonlinearity, so no padding makes it
exact. Newton works on the exact Jacobian of the discretised problem:

```{literalinclude} _check.py
:language: python
:start-after: "[cont-newton]"
:end-before: "[/cont-newton]"
:dedent: 4
```

## Following the branch through the fold

```{literalinclude} _check.py
:language: python
:start-after: "[cont-branch]"
:end-before: "[/cont-branch]"
:dedent: 4
```

Every step-size rule is in `ContinuationSettings`: initial, minimum and maximum steps, adaptation
factor, Newton budget and tolerance, the finite-difference step of $\partial_p G$ (or `None` when
exact derivatives exist).

## Bifurcations of a dynamical system

For a system of ODEs, or a PDE reduced to one by `ir_system`, Conatus computes the exact Jacobian,
$\partial_p G$ and the multilinear forms used by the normal forms.

```{literalinclude} _check.py
:language: python
:start-after: "[cont-ode]"
:end-before: "[/cont-ode]"
:dedent: 4
```

## Branch switching, Hopf points and cycles

```{literalinclude} _check.py
:language: python
:start-after: "[cont-switch]"
:end-before: "[/cont-switch]"
:dedent: 4
```

At a symmetry-breaking pitchfork the symmetry is declared (`symmetry=S`) and the branch point is
classified. At a Hopf point the first Lyapunov coefficient decides between sub- and
supercritical; the cycles born there are continued by orthogonal collocation on a declared mesh.
A branch of cycles can also start from any orbit given as a collocation state `z0` (the mesh
values, the period and the parameter):
`continue_orbits(PeriodicOrbitProblem(system, "r", mesh), z0, settings)`, with the same tests
(`"lpc"`, `"pd"`, `"ns"`, `"bpc"`) and `explain()`.

## Codimension two

```{literalinclude} _check.py
:language: python
:start-after: "[cont-codim2]"
:end-before: "[/cont-codim2]"
:dedent: 4
```

Fold and Hopf curves are continued in two parameters; codimension-two points on them
(Bogdanov–Takens, cusp, generalized Hopf) are reported as events.

Amplitude equations (`amplitude_equation`, `ginzburg_landau`) reduce a PDE near onset to its
Landau or Ginzburg–Landau form; tutorial {doc}`../../tutorials/10_bifurcations` checks them on
Swift–Hohenberg.

**See also:** tutorials {doc}`../../tutorials/04_bratu` and {doc}`../../tutorials/10_bifurcations`;
{doc}`ContinuationSettings <../../api/_generated/_stubs/conatus.expert.ContinuationSettings>`, {doc}`NLBVPContinuation <../../api/_generated/_stubs/conatus.expert.NLBVPContinuation>`,
{doc}`continue_equilibria <../../api/_generated/_stubs/conatus.expert.continue_equilibria>`, {doc}`branch_switch <../../api/_generated/_stubs/conatus.expert.branch_switch>`,
{doc}`continue_codim2 <../../api/_generated/_stubs/conatus.expert.continue_codim2>`, {doc}`normal_form <../../api/_generated/_stubs/conatus.expert.normal_form>`.
