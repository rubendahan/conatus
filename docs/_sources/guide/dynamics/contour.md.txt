# Eigenvalues in a region

The eigenvalues inside a box of the complex plane: how many there are, and which they are.

A dense eigenvalue solver gives every eigenvalue of a small problem. Shift-invert gives a few near
a target and says nothing about the others. A *region* asks which eigenvalues lie in a given box.
Conatus answers in two steps that check each other:

1. **the count**, by the argument principle. The number of eigenvalues inside a closed contour is
   the number of times $\det T(z)$ winds around zero as $z$ goes once around it. The routine
   estimates this winding by sampling the determinant around the box;
2. **the eigenvalues**, by Beyn's contour-integral method. Integrals of $T(z)^{-1}$ along the
   same contour carry the eigenvalues inside. Each one is refined and checked by its residual,
   and together they must account for the count.

When the implemented sampling checks fail, the routine refuses and gives the reason. Passing those
checks is not a mathematical certificate: the samples can miss a narrow excursion of the phase.
The scope of the count is stated with the result (below).

```{figure} ../media/contour_sampling.png
:alt: Left, a rectangle in the complex plane with blue sampling points along its edges, dense near one red cross that sits just inside the right edge; other red crosses inside and outside the box. Right, the phase of the determinant along the boundary, in turns, rising in steps to 4.
:width: 100%

How the count samples the boundary, for a 6 × 6 pencil with known eigenvalues (crosses), four of
them in the box, one 2 × 10⁻³ inside its right edge. Left: the points where the walk evaluated
the determinant. Its shortest step, 2.6 × 10⁻⁴ of the perimeter, is next to the eigenvalue close
to the edge. Right: the phase of det T
along the boundary, in turns; it closes on 4, the count. The walk accepts a step from sampled
values, so this picture shows a test that passed, not a proof. The figure is made by
`docs/site/guide/_fig_contour.py`.
```

## In a native model

```{literalinclude} _check_contour.py
:language: python
:start-after: "[ct-region]"
:end-before: "[/ct-region]"
:dedent: 4
```

`region=cn.box(re=(a, b), im=(c, d))` is the closed rectangle $a \le \mathrm{Re}\, c \le b$,
$c \le \mathrm{Im}\, c \le d$ of the eigenvalue's plane (here the phase speed `c` of the
Orr–Sommerfeld problem). The result is an eigenvalue result like any other (`res.values`,
`res.residuals`, `res.mode(i)`, `res.order()`, `drift=` and the spurious-mode flag work as for the
dense solver), restricted to the box, with three additions:

- `res.count`: the number of eigenvalues of the discretised problem in the box, multiplicities
  included;
- `res.count_method`: how it was obtained (the argument principle, the number of evaluations of
  the determinant, and what the sampled walk can miss);
- `res.multiplicity`: for each eigenvalue returned, the number of eigenvalues counted by the
  argument principle in a small square around it. Close values are first separated, each on
  its own square; 1 means a separated eigenvalue. A value above 1 means a multiple eigenvalue or
  a cluster not separated at that size (a double eigenvalue appears once, with 2; a defective
  eigenvalue, which rounding splits into a small circle of values, is returned once, at their
  mean). The half-side of each square is in `res.contour.enclosure`.

The count is that of the *discrete* pencil, spurious modes of the discretisation included. Filter
them as usual (`drift=N2, drift_tol=1e-6`); `res.count` still gives how many the box held before
the filter. Only the eigenvalues in the box are computed, which on a large problem costs much less
than the full spectrum. With `region=` the method is the contour method: leave `method=` out (or
write `method="contour"`). `method="dense"`, `method="shift-invert"`, `target=` and `left=True`
contradict a region and raise an error.

## Any operator: `cn.contour.count` and `cn.contour.beyn`

The same two steps take an operator directly: a pair `(L, M)` for the pencil $L + zM$, a
Floquet–Hill operator (`HillBlocks`, not assembled), or any function `z -> T(z)` returning a
matrix or a scalar.

```{literalinclude} _check_contour.py
:language: python
:start-after: "[ct-count]"
:end-before: "[/ct-count]"
:dedent: 4
```

The Floquet exponents of Mathieu's equation at a stable point are a neutral pair $\pm i\nu$
(modulo $2i$). The box around the pair holds two; `control=(gap, 0)` first counts a box known to
hold none and refuses to go on if it does not find zero, which checks the operator and the walk
before any count is used. `beyn(..., counted=n)` reuses the count; it evaluates $\det T$ again at
points the count recorded and refuses a count of another operator.

`count` returns the winding number $N = Z - P$ of $\det T$, zeros minus poles inside. A pencil
and a Hill operator linear in the exponent have no pole. For a function you supply, you declare
its poles in the box with `poles=` (`poles=0`: holomorphic there); without it, `count` gives the
winding but `count=None`, and `beyn` refuses to start, since it checks its roots against the
number of zeros.

## Nonlinear eigenproblems

Beyn's method is not limited to pencils: any $T(z)$ holomorphic in the box works (exponentials of
a delay equation, boundary conditions that depend on the eigenvalue, an Evans function,
a polynomial eigenproblem).

```{literalinclude} _check_contour.py
:language: python
:start-after: "[ct-beyn]"
:end-before: "[/ct-beyn]"
:dedent: 4
```

`derivative=` gives $T'(z)$, used only to refine each root (Newton-type inverse iteration);
without it a central difference is used, which slows the refinement but not the checks.
A root is accepted only if its residual $\|T(z)v\| / (s(z)\,\|v\|)$ is at most `residual_tol`
(default $10^{-13}$) and a small square around it, counted by the argument principle, holds an
eigenvalue. The scale $s(z)$ depends on the operator and is stated in `residual_definition`:

| Operator | $s(z)$ |
|---|---|
| pencil $L + zM$ | $\|L\|_1 + \lvert z\rvert\,\|M\|_1$ |
| Hill operator linear in the exponent | an upper bound of $\|T(z)\|_1$ |
| function $T(z)$, nonlinear Hill operator | $\max(\|T(z)\|_1, s_\mathrm{ref})$, $s_\mathrm{ref}$ the median of $\|T\|_1$ at the corners and edge midpoints of the box |

For a pencil the residual is a backward error relative to its coefficients: the root is an
eigenvalue of a pencil within that distance of $L + zM$. For a function the floor
$s_\mathrm{ref}$ keeps a scalar $f$ from scoring $|f(z)|/|f(z)| = 1$; the residual is then relative
to the size of $T$ on the boundary, not a relative perturbation of the near-singular $T(z)$.
How far a root is from the exact eigenvalue depends on the conditioning (about $10^{-8}$ on the
non-normal Orr–Sommerfeld branch below).

## What is checked, and what is refused

| Check | On failure |
|---|---|
| the phase of $\det T$ is followed with increments of at most $\pi/4$ per step (and per half step); $\log\det T$ is resolved by each step | the step is halved, down to $10^{-12}$ of the perimeter |
| the walk closes on its starting point, evaluated again: the winding is an integer | refused (a determinant that does not return the same value at the same point) |
| no eigenvalue on the boundary | refused, with where: move the box |
| a control box, when given, holds its known count | refused |
| every root lies in the box and has a residual at most `residual_tol` | not returned |
| every root returned (or group of roots split by rounding) holds its own argument-principle count on a small square, at least 1, and these multiplicities add up to the count | more quadrature nodes and moments (three rounds), then the box is cut in two, then refused with what was found |
| `counted=`: $\det T$ at the points the count recorded is that of the operator given | refused |

What the sampled walk can miss. The phase and modulus tests see a zero of $\det T$ that comes
close to the boundary, because the modulus dips or the phase turns by about $\pi$ across a step.
They cannot see a zero and a pole on opposite sides of the boundary, both closer to it than the
local step: $|\det T|$ stays near 1 there and the phase turns by $2\pi$ between two samples. A
pencil and a Hill operator linear in the exponent have no poles
(`CountResult.poles_by_construction` is True). For a function with poles near the box the count
can be off by one; `count_method`, `explain()` and `CountResult.scope` say so. Keep the edges of the box away from
the poles of $T$.

The routine does not form the determinant. It reads the phase and the logarithm of the modulus
off the LU factorisation of $T(z)$, which avoids the overflow of multiplying the factors. The row
and column permutations of the factorisation enter with their signs, so the result does not depend
on the pivot order.

```{literalinclude} _check_contour.py
:language: python
:start-after: "[ct-refuse]"
:end-before: "[/ct-refuse]"
:dedent: 4
```

A user-supplied determinant must be a continuous function of $z$: a product of factors picked by
their order (one eigenvalue "the one with largest real part", pivots without their permutation
sign) jumps where the order changes, and the count is then refused. Use
the determinant of the whole matrix, or pass the matrix itself.

## Validation

On the Orr–Sommerfeld problem above ($Re = 10^4$, $\alpha = 1$; $N = 48$ to $128$), the counts of
five boxes, from the single unstable mode to 63 eigenvalues, equal the number of eigenvalues of the
full dense spectrum inside them. Each root agrees with the dense eigenvalue to $5\times10^{-8}$
(to $10^{-11}$ away from the ill-conditioned lower branch), with residuals below $10^{-13}$. One
large box at $N = 128$ whose edge crosses that branch is refused. Other contours need their own
resolution and conditioning checks. The neutral Floquet pair of
Mathieu's equation, a nonlinear problem with known roots, a pencil of size 4000 whose determinant
is about $10^{12000}$, and the refusals above are checked by the test suite.

References: the argument principle (e.g. Ahlfors, *Complex Analysis*); W.-J. Beyn, An integral
method for solving nonlinear eigenvalue problems, *Linear Algebra Appl.* 436 (2012) 3839–3863.

**See also:** {doc}`eigenvalues`, {doc}`floquet`.
