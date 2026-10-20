# Eigenvalues in a region

Every eigenvalue inside a box of the complex plane, with a count you can trust: how many there
are, and which they are.

A dense eigenvalue solver gives every eigenvalue of a small problem; shift-invert gives a few near
a target, and says nothing about the ones it did not return. A *region* asks a different question:
**which eigenvalues lie in this box, and are there any others?** Conatus answers it in two steps
that check each other:

1. **the count**, by the argument principle: the number of eigenvalues inside a closed contour is
   the number of times $\det T(z)$ winds around zero as $z$ goes once around it;
2. **the eigenvalues**, by Beyn's contour-integral method: integrals of $T(z)^{-1}$ along the
   same contour hold exactly the eigenvalues inside. Each one is refined and validated by its
   residual, and together they must account for the count.

Either step that cannot conclude refuses with the reason: no count is ever guessed.

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
- `res.count_method`: how it was certified (the argument principle, the number of evaluations of
  the determinant);
- `res.multiplicity`: the algebraic multiplicity of each eigenvalue returned, its own
  argument-principle count on a small square around it (a double eigenvalue appears once, with
  multiplicity 2; a defective eigenvalue, which rounding splits into a small circle of values, is
  returned once, at their mean, with its multiplicity).

The count is that of the *discrete* pencil, spurious modes of the discretisation included: filter
them as usual (`drift=N2, drift_tol=1e-6`); `res.count` then still says how many the box held
before the filter. Only the eigenvalues in the box are computed: on a large problem this is far
cheaper than the full spectrum. With `region=` the method is the contour method: leave `method=`
out (or write `method="contour"`); `method="dense"`, `method="shift-invert"`, `target=` and
`left=True` contradict a region and are refused, never replaced.

## Any operator: `cn.contour.count` and `cn.contour.beyn`

The same two steps take an operator directly: a pair `(L, M)` for the pencil $L + zM$, a
Floquet–Hill operator (`HillBlocks`, never assembled), or any function `z -> T(z)` returning a
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
before any count is trusted. `beyn(..., counted=n)` reuses the count.

`count` returns the winding number $N = Z - P$ of $\det T$, zeros minus poles inside. A pencil
and a Hill operator linear in the exponent have no pole. For a function you supply, you declare
its poles in the box with `poles=` (`poles=0`: holomorphic there); without it, `count` gives the
winding but `count=None`, and `beyn` refuses to start, since it validates its roots against the
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
without it a central difference is used, which slows the refinement but not the validation:
every root is accepted only if its relative backward residual
$\|T(z)v\| / (\|T(z)\|\,\|v\|)$ is at most `residual_tol` (default $10^{-13}$) and a small square
around it, counted by the argument principle, holds an eigenvalue. The residual is a backward
error: the root is an eigenvalue of an operator within $10^{-13}$ of $T$; how far it is from the
eigenvalue of $T$ itself depends on the conditioning (about $10^{-8}$ on the non-normal
Orr–Sommerfeld branch below).

## What is checked, and what is refused

| Check | On failure |
|---|---|
| the phase of $\det T$ is followed with increments of at most $\pi/4$ per step (and per half step); $\log\det T$ is resolved by every step | the step is halved, down to $10^{-12}$ of the perimeter |
| the walk closes on its starting point, evaluated again: the winding is an integer | refused (a determinant that does not return the same value at the same point) |
| no eigenvalue on the boundary | refused, with where: move the box |
| a control box, when given, holds its known count | refused |
| every root lies in the box and has a residual at most `residual_tol` | not returned |
| every root returned (or group of roots split by rounding) holds its own argument-principle count on a small square, at least 1, and these multiplicities add up to the count | more quadrature nodes and moments (three rounds), then the box is cut in two, then refused with what was found |

The determinant is never formed: its phase and the logarithm of its modulus are read off the LU
factorisation of $T(z)$, so it cannot overflow, whatever the size. The row and column permutations
of the factorisation enter with their signs, so the result does not depend on the pivot order.

```{literalinclude} _check_contour.py
:language: python
:start-after: "[ct-refuse]"
:end-before: "[/ct-refuse]"
:dedent: 4
```

A user-supplied determinant must be a continuous function of $z$: a product of factors picked by
their order (one eigenvalue "the one with largest real part", pivots without their permutation
sign) jumps where the order changes, and the count is refused rather than read off a jump. Use
the determinant of the whole matrix, or pass the matrix itself.

## Validation

On the Orr–Sommerfeld problem above ($Re = 10^4$, $\alpha = 1$; $N = 48$ to $128$), the counts of
five boxes, from the single unstable mode to 63 eigenvalues, equal the number of eigenvalues of the
full dense spectrum inside them, and every root agrees with the dense eigenvalue to $5\times10^{-8}$
(to $10^{-11}$ away from the ill-conditioned lower branch), with residuals below $10^{-13}$. One
large box at $N = 128$ whose edge crosses that branch is refused: a refusal, never a wrong root. The neutral Floquet pair of
Mathieu's equation, a nonlinear problem with known roots, a pencil of size 4000 whose determinant
is about $10^{12000}$, and the refusals above are checked by the test suite.

References: the argument principle (e.g. Ahlfors, *Complex Analysis*); W.-J. Beyn, An integral
method for solving nonlinear eigenvalue problems, *Linear Algebra Appl.* 436 (2012) 3839–3863.

**See also:** {doc}`eigenvalues`, {doc}`floquet`.
