# Error codes

Every typed error of Conatus carries a stable code in its attribute `error_code`. A program can
branch on the code. The message can change from one release to the next, and a code does not: it
keeps its class, and a new class gets a new code.

```python
import conatus as cn
from conatus.errors import ConatusError

m = cn.Model(cn.Box(x=cn.walls(0, 1)), name="heat")
m.scalar("u")
m.equations("dx(dx(u)) = 0")
try:
    m.eigenvalues(8)                  # no eigenvalue was declared
except ConatusError as err:
    print(err.error_code, err.as_dict()["type"])
```

```text
CN150 OptionError
```

`err.as_dict()` returns the code, the class name, `message`, `where`, `why`, `fix` and `doc`. The fix
is a suggestion, and Conatus does not apply it. `python -m conatus agent check` reports the same
`error_code` in its JSON, and the MCP server ({doc}`studio_agents`) returns it with a refusal.

A catch by class keeps working: a code names a class, and the classes keep their parents
({doc}`errors`). A subclass private to a module has its parent's code.

| Code | Class | Meaning |
|---|---|---|
| `CN001` | `ConatusError` | base class of the library's errors |
| `CN100` | `EquationError` | an equation or the whole problem is inconsistent (also `ProblemError`) |
| `CN101` | `NonlinearLHSError` | a nonlinear term on the implicit side |
| `CN102` | `UndeclaredVariableError` | an equation names an unknown that was not declared |
| `CN103` | `ParameterError` | a parameter is missing, unknown or inconsistent |
| `CN104` | `BoundaryConditionError` | a boundary condition is missing or inconsistent |
| `CN105` | `EquationSyntaxError` | a string equation does not follow the grammar |
| `CN110` | `DiscretizationError` | the problem cannot be discretized as declared |
| `CN111` | `RankError` | the discretized system has the wrong rank |
| `CN120` | `SpectralError` | a spectral basis or transform cannot do what was asked |
| `CN130` | `ResolutionError` | the resolution cannot represent what was asked |
| `CN140` | `CoordinateError` | a coordinate or a domain is invalid |
| `CN150` | `OptionError` | a required choice is missing or an option is invalid |
| `CN151` | `ArgumentError` | the arguments of a call do not match its signature (an unknown or former keyword, a missing argument) |
| `CN160` | `SymmetryError` | a symmetry assumption does not hold |
| `CN161` | `SymmetryBreakingError` | a computation left a declared symmetry subspace |
| `CN170` | `MissingFileError` | a file that was named does not exist |
| `CN171` | `UnsafeFileError` | a file holds links, external storage or pickled objects, which the loaders refuse |
| `CN180` | `InputTypeError` | an argument has the wrong type |
| `CN181` | `DtypeError` | an array has the wrong data type |
| `CN182` | `TensorError` | a tensor has the wrong rank or shape |
| `CN190` | `ConvergenceError` | an iteration did not converge |
| `CN191` | `NewtonNotConverged` | a Newton iteration did not converge |
| `CN200` | `NumericalError` | a value became non-finite or a floating-point check failed |
| `CN210` | `BackendError` | a backend (GPU, compiled core) cannot do what was asked |
| `CN211` | `MissingDependencyError` | an optional dependency is not installed |
| `CN220` | `UnsupportedError` | the combination is not supported |
| `CN221` | `LiteralTooLargeError` | a translated script declares a literal above the limit |
| `CN222` | `XDMFNotCovered` | an output cannot be described in XDMF |
| `CN230` | `AmbiguousNameError` | a name of `conatus.expert` refers to several objects |
| `CN231` | `NameNotExportedError` | `conatus.expert` does not export a name (the message names its module) |
| `CN240` | `VJPSuspended` | an adjoint sweep was suspended and needs resuming |
| `CN241` | `SingularPoint` | a contour operator met a singular point |
| `CN242` | `PlanRefused` | a planned sphere adjoint cannot be built |
| `CN243` | `ParameterDerivativeError` | a parameter derivative is not available |
| `CN244` | `UndecidedTrajectory` | a trajectory reached neither fate in its budget |
| `CN300` | `CommandRefused` | the agent channel refused a command |
| `CN301` | `StudioError` | the hub cannot be reached, or its answer is not the API |
| `CN302` | `LaunchError` | a Studio job request was refused |
| `CN303` | `StopRun` | a run was stopped by a command |
| `CN304` | `Refusal` | the MCP server refused a tool call |

The warning classes have no codes.

A hub or spool answer also holds a numeric `code`, which is an HTTP-like status ({doc}`studio_agents`).
It is a different field, and `error_code` is the stable one.
