# Equations

A model is written as on paper: unknowns, parameters, known fields and equations in text, with
terms on whichever side you like.

```python
import numpy as np
import conatus as cn

m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5)),
             name="Rayleigh-Benard 2D", Ra=2500, Pr=1)      # parameters by keyword
u, p, T = m.velocity("u"), m.pressure("p"), m.scalar("T")
m.equations("""
    momentum:   dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
    continuity: div(u) = 0
    heat:       dt(T) - lap(T) = -u·grad(T)
""")
m.at("bottom", u=0, T=+0.5)
m.at("top", u=0, T=-0.5)
print(m.explain())
```

## Declarations

| Call | Declares |
|---|---|
| `cn.Model(geometry, name=..., Ra=2500, ...)` | the model and its parameters (numbers, by keyword); later `m.Ra = 3000` or `m.params["Ra"] = 3000` (declared names only, checked), or `m.simulate(..., Ra=3000)` for one run |
| `m.scalar("T")`, `m.vector("B")` | an unknown |
| `m.velocity("u")`, `m.pressure("p")` | a vector and a scalar unknown with a role that Conatus checks (incompressibility, {doc}`boundary_conditions`) |
| `m.given(U=lambda y: 1 - y**2)` | a known field; the argument names say which coordinates it depends on |
| `m.define("L2(f) = dy(dy(f)) - alpha**2*f")` | a named operator, substituted where it is used |
| `m.function("g", np.tanh, derivative=...)` | a pointwise function of a scalar field, used as `g(u)` (Dedalus's `GeneralFunction`); without `derivative` linearization and Newton are refused |
| `m.eigenvalue("c", growth="imag")` | the model is an eigenvalue problem in `c`; `growth=` declares how a mode grows (`"imag"`: phase speed of $e^{i\alpha(x-ct)}$, `"real"`: rate of $e^{st}$); left out, v0.1 guesses from the name (`c` → Im, `s` → Re) with a `DefaultChoiceWarning`, required from v0.2 |
| `m.equations("...")` | equations, one per line, optionally labelled `name:` |
| `m.at("bottom", T=0)` | boundary conditions ({doc}`boundary_conditions`) |

Every unknown is declared before the equations. Names are checked: a name already used, a
coordinate name, an operator name or a name starting with `tau_` is refused.

A function of your own enters the equations by its name. It is evaluated on the dealiased grid
(not alias-free, as `explain()` says), and takes and returns arrays (CuPy arrays on the GPU):

```python
import numpy as np
import conatus as cn

m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="damped by tanh", eps=0.1)
m.scalar("u")
m.function("g", np.tanh, derivative=lambda u: 1 - np.tanh(u)**2)
m.equations("dt(u) - eps*lap(u) = -g(u)")
sim = m.simulate(N=32, scheme="RK443", dt=1e-2).initial(u=lambda x: np.sin(x))
sim.run(steps=20)
```

## The kind of problem follows from the equations

| The equations contain | Problem | Solve with |
|---|---|---|
| `dt(...)` | evolution (IVP) | `m.simulate(N=..., scheme=..., dt=...)` ({doc}`timesteppers`) |
| a declared eigenvalue | eigenvalue problem | `m.eigenvalues(N=...)` ({doc}`dynamics/index`) |
| neither, linear | boundary-value problem | `m.discretize(N).solve()` |

An evolution model also gives its linear stability problem: `m.linearize(about={u: 0, T: lambda z: -z})`
returns normal modes $\propto e^{st}$ about that base state.

## Which terms are implicit

In an evolution problem you do not split the terms into a left and a right side. Conatus does it,
term by term, with one rule: a term **linear** in the unknowns (constant coefficients,
parameters, given fields) is treated implicitly; a **nonlinear** term or a source is explicit.
A term changes side with its sign, and `explain()` prints where each one went:

```text
Term placement (IMEX split, automatic; override with implicit(...) / explicit(...)):
  [0] momentum
      implicit  grad(p)   (linear, constant coefficients)
      implicit  dt(u)   (linear, constant coefficients)
      implicit  -Pr*lap(u)   (linear, constant coefficients)
      implicit  -T*Ra*Pr*ez   (linear, constant coefficients; moved from the right side)
      explicit  -dot(u, grad(u))   (nonlinear, degree(s) [2] in the unknowns)
```

To override the rule, wrap a whole term: `explicit(-u)` treats a linear reaction explicitly,
`implicit(...)` forces a term implicit (refused on a nonlinear term).

## Syntax

| Written | Meaning |
|---|---|
| `dt grad div lap curl dot cross outer integ` | time derivative and vector calculus in the geometry's coordinates |
| `dx(u)`, `dz(u)`, `dr(u)` | partial derivative along a coordinate (`d` + its name) |
| `ex`, `ez` | Cartesian unit vectors (`e` + a coordinate name) |
| `u·grad(u)` or `u@grad(u)` | contraction (dot product); `·` and `−` are accepted |
| `u(z=0)`, `integ(p)` | trace at a point; integral over the domain |
| `exp log sin cos tanh sqrt abs sign` | pointwise functions (reported as not alias-free) |
| `u[0]`, `pi`, `1j`, `2.5e-3` | a component; numbers |

Precedence is Python's: `-u·grad(u)` is the product of `-u` and `grad(u)`. A line ending with an
operator or an open parenthesis continues on the next line; `#` starts a comment. The string is
never passed to `eval`: a small parser reads it, and each name is looked up among your
declarations.

## Errors point at the column

```text
UndeclaredVariableError: name 'lapp' is not declared
  where: dt(T) - lapp(T) = 0
                 ^ (column 9)
  fix:   ... Did you mean 'lap' or 'kappa'?

EquationSyntaxError: unexpected 'lap': an operator is missing
  where: dt(T) - kappa lap(T) = 0
                       ^ (column 15)
  fix:   write '*' for a product (Pr*lap(T), 2*u), '+' or '-' between terms
```

The suggestion is never applied for you. A product of two vectors written `u*u` is refused
(write `u·u` or `outer(u, u)`), and Dedalus spellings that Conatus does not have, such as
`trace(grad(u))`, are refused with the equivalent form (`div(u)`).

## Behind the text

The same strings are accepted by the expert layer (`problem.add_equation("...")`), where each one is
parsed into exactly the expression the Python operators would build. `conatus.ir.parse.GRAMMAR`
is the full grammar, and `to_string(eq)` writes an equation back as text that parses to the same
expression ({doc}`operators`). The parser is also available on its own: `cn.parse_equation`
reads `"lhs = rhs"` and `cn.parse_expr` one expression, against the names you pass (a mapping, a
problem, or a list of fields and parameters):

```python
import conatus as cn
from conatus.expert import CartesianCoordinates, Chebyshev, Domain, Field, Parameter

cs = CartesianCoordinates("x"); x = cs["x"]
u = Field("u", Domain(cs, [Chebyshev(x, 16, (-1, 1), dealias=3/2)]))
nu = Parameter("nu", 0.1)
eq = cn.parse_equation("dt(u) - nu*lap(u) = -u*dx(u)", dict(u=u, nu=nu))
print(eq)                                   # dt(u) - nu*lap(u) = -u*dx(u)
print(cn.parse_expr("u*dx(u)", [u]))        # an IR expression: u*dx(u)
```
