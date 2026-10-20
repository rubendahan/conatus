# For AI agents

What an AI coding agent reads before using Conatus, and the JSON interface it uses while it works.

An agent that writes and runs Conatus code should be given one page,
`AGENTS.md`, at the root of the repository. It lists the whole usable surface, one call per line, grouped by
task, and the rules that keep an agent from the usual mistakes: reading a refusal and its
suggested fix instead of working around it, writing every tolerance that decides digits,
validating by residuals and `explain()`, checking cost and memory before a large run. Its content
is checked against the library by the test suite: every name it cites exists, every major public
element is there, and its example runs.

## Ask the library, in JSON

Three commands answer an agent without any prose to parse. Each prints one JSON object (keys
sorted, `"api": 1`; within an API version fields are only added) and says what happened by its
exit code.

```bash
python -m conatus agent                          # versions, CPU/GPU, the user layer, watched runs
python -m conatus agent explain cn.Model.simulate     # signature, required arguments and tolerances
                                                      # (and "deprecated": former argument names)
python -m conatus agent explain cn.problems.native.rayleigh_benard   # a ready model: its unknowns too
python -m conatus agent check run.py             # dry run of a script: what Conatus would refuse
```

`check` runs the script for real up to its first expensive call (a time integration, an
eigenvalue solve, a Newton search, a continuation…) and stops there. On the way, `simulate` builds
its solver, so a search that takes `sim.solver` is reached too; calls that write files or announce
a Studio run (`watch`, `output`, `checkpoints`, `save`) are skipped. At the stop, the call's
arguments are checked against its signature, the model is discretised at the requested resolution
(so an under-resolved coefficient or a missing boundary condition is refused here, with its fix),
and the `explain()` texts are collected. With `backend="cuda"`, `simulate` itself is the stop (no
device memory is allocated) and the device memory of the simulation is predicted; `--memory` does
the same prediction for a CPU simulation. No intercepted call integrates or solves (the list is
`conatus.agent.EXPENSIVE`).

`check` is not a sandbox. The script runs in the current process with your privileges: its own
code (writing a file, starting a subprocess) runs as written, and only the Conatus calls of that
list are intercepted, in that process only (not in a child process or an MPI launch). A script
that catches the stop (a bare `except:` or `except BaseException`) and goes on is reported as an
error, since the lines after it ran for real. The timeout cannot interrupt a long call inside
compiled code. Check only scripts you would run yourself.

For this script, where the drift test of an eigenvalue solve was asked for without its tolerance,

```{literalinclude} outputs/os_check.py
:language: python
:linenos:
```

the report says what is missing, where, and the value Conatus suggests (exit code 1):

```{literalinclude} outputs/agent_check.json
:language: json
```

| exit code | `agent info` / `explain` / `check` |
|---|---|
| 0 | done; for `check`, nothing refused or missing |
| 1 | `check`: a refusal, a missing argument or tolerance, or an error in the script |
| 2 | usage error; `explain FILE.py` (explain never runs a script: use `check`) |
| 4 | `explain`: no such public name |

The same functions are available from Python: `conatus.agent.info()`, `conatus.agent.describe(name)`
and `conatus.agent.check(path)` return the dictionaries printed above.

## Follow a run that is already going

A long run declared with `sim.watch(...)` is followed and piloted with
`python -m conatus studio status|watch|send|frame|log`, with its own exit codes and per-agent
tokens: see {doc}`studio_agents`.

## AGENTS.md

The page itself, as an agent reads it:

```{include} ../../../AGENTS.md
:start-line: 1
```
