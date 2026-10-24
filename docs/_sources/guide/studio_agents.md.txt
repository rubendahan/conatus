# Pilot a run from a script or an agent

Everything the Studio shows about a run, a script can read as JSON, and the commands of the
Controls sheet, a script can send (experimental, like the Studio: {doc}`experimental`). The same channel serves a person on a phone, a shell script
and an AI agent that follows a long run and acts on it. The command schema
(`conatus.telemetry.protocol`) lists the supported actions and the confirmation fields each one
needs. A run changes only what it declared changeable, and every command is written to an audit
log.

## Declare what may be changed

By default a watched run can be paused, resumed, checkpointed, stopped or forked from outside. To
let the time step be changed too, declare it with its bounds:

```python
import numpy as np
import conatus as cn

m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2e4, Pr=1)
sim = m.simulate(N=(32, 16), scheme="RK443", dt=2e-3)
sim.initial(T=lambda x, z: -z + 1e-3 * np.cos(np.pi * x) * np.cos(np.pi * z))
sim.watch("rb_demo", film={"T": cn.view("T", scale="signed")},
          settable={"dt": (1e-3, 4e-3)})      # dt may be set between 1e-3 and 4e-3, nothing else
sim.run(until=0.2)
sim.unwatch()
```

- The new value is applied between two steps and recorded in the run (event `param_set`, with the
  old value, the step and who asked).
- A value outside the bounds, or a parameter that was not declared, is refused (event
  `set_refused`, with the reason). The value is not clipped or rounded to fit.
- With `run(until=T)` the time left must stay a whole number of steps: a step that would miss `T`
  is refused, and the refusal gives the nearest step that lands on it. With `run(steps=n)`, `n`
  steps are taken in all.
- Only Runge–Kutta schemes can change their step; a multistep scheme (SBDF, CNAB) is refused when
  the watch is declared. A parameter of the equations is compiled into the operators: changing it
  is a new run (fork from a checkpoint), not a `set`.

`w.explain()` lists what is settable and how.

## Read and act from the command line

```bash
python -m conatus studio runs                       # every run, live first
python -m conatus studio status rb_demo             # state, step, time, dt, scalars, alerts, log tail
python -m conatus studio watch rb_demo              # one JSON line per change, until the run ends
python -m conatus studio frame rb_demo --png rb.png # the latest picture of the default view
python -m conatus studio send rb_demo pause
python -m conatus studio send rb_demo set dt=2e-3
python -m conatus studio send rb_demo resume
python -m conatus studio send rb_demo stop          # answers with a confirm_token (exit code 5)
python -m conatus studio send rb_demo stop --confirm <confirm_token>
python -m conatus studio log rb_demo -n 50          # log records and events (what the run did)
python -m conatus studio scalars rb_demo --names energy --max 200
python -m conatus studio ping | whoami | board      # reachability, who you are, the dashboard
```

A command that succeeds (exit 0) has been **written** to the run's control file; the run applies
it at its next check (`controls_every_s`, 1 s by default, and only between two steps). Confirm it in
the events: `command_received`, then `param_set` (or `set_refused` with the reason), `resumed`,
`checkpoint_written`, … (`status` or `log`).

The output is JSON (keys sorted, API version `api: 1`; fields are only ever added within a
version). The exit code says what happened without parsing anything:

| code | meaning |
|---|---|
| 0 | done |
| 1 | refused: unknown command, out of bounds, the run has ended |
| 2 | usage error |
| 3 | not authorised: bad token, or a read-only token sending a command |
| 4 | no such run, or no such picture |
| 5 | confirmation needed: send the same command again with `--confirm <confirm_token>` |
| 6 | the hub cannot be reached |

Stop and fork are confirmed in two steps: the first answer carries a token bound to the sender,
the run, the command and its parameters, valid two minutes from its issue and once.

The JSON of a run holds `state` (`running`, `paused`, `stale`, `finished`, `stopped`, `failed`),
`t`, `iteration`, `dt`, the last value of every scalar, `alerts` (stable codes: `non_finite`,
`silent`, `relay_down`, `log_errors`, `scalar_error`, `dropped`, `film_full`, `set_refused`,
`failed`, `telemetry_error`), `commands` (those that make sense now), `settable` (the declared
parameters and their bounds), and the tail of the log and of the events.

## Where the command line talks

- **On the cluster** (a session in a job, the spool on the shared file system): no server is
  needed. The command line reads the spool and appends the command to the run's control file itself, with
  the same checks; file permissions are the authentication, and commands are logged in
  `<spool>/studio_audit.jsonl`. Control files are created `0600`, and a run ignores (with an
  error in its log) a control file that belongs to another account or that group or others can
  write: nobody else on the cluster can pause, stop or re-time your run.
  The spool directory is created `0700`, and the telemetry files and the audit log in it are
  `0600`, whatever your umask. Other accounts cannot read the logs, numbers or films written from this version on. To
  share them with your group, copy them or change their mode yourself.
- **Through a hub**: `--hub https://…` (or `$CN_STUDIO_URL`) with a token in a file
  (`--token-file`, `$CN_STUDIO_TOKEN`, or `~/.config/conatus/studio_token`). Plain `http://` is
  accepted only to `localhost`, through an ssh tunnel to a hub running in a job, so the token does
  not cross the network in clear text.

## Tokens for agents

Create a separate scoped token for each agent:

```bash
H="python -m conatus.studio.hub --state ~/.config/conatus/hub"
$H addagent watcher --scope read --token-out ~/.config/conatus/watcher_token
$H addagent claude --scope command --runs 'local:rb_*' --token-out ~/.config/conatus/studio_token
$H agents                      # names, scopes, run patterns (tokens are not shown)
$H delagent claude             # revoke
```

- `read`: state, scalars, log, pictures. Every command is refused (and logged).
- `command`: also the declared run commands. The command scope excludes job submission and access
  to the audit log.
- `--runs`: the run keys the token can reach at all (patterns); other runs do not exist for it.

The token is written to a file readable only by you and shown nowhere else; the hub keeps only
its digest.

## From Python

```python
import numpy as np
import conatus as cn
from conatus.studio.client import connect

sim = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2e4, Pr=1).simulate(
    N=(32, 16), scheme="RK443", dt=2e-3)
sim.initial(T=lambda x, z: -z + 1e-3 * np.cos(np.pi * x) * np.cos(np.pi * z))
sim.watch("rb_demo", film={"T": cn.view("T")})
sim.run(steps=10)                                # a watched run (usually in another process)

cl = connect()                                   # same resolution as the command line
run = cl.status("rb_demo")["run"]
print(run["state"], run["t"], run["alerts"])
blown_up = any(a["code"] == "non_finite" for a in run["alerts"])
answer = cl.send("rb_demo", "stop")              # (on blown_up, say) a live run: refused once
if answer.get("needs_confirmation"):             # ... with a token to send it again
    answer = cl.send("rb_demo", "stop", confirm=answer["confirm_token"])
print(answer.get("ok"), answer.get("error", ""))  # a finished run refuses the stop
sim.unwatch()
```

`cl.wait(run, since=wall, timeout=25)` returns as soon as the run has something new (a long poll
on the hub), which is how `watch` follows a run without polling.

`connect()` returns a `HubClient` (through a hub, with a token) or a `SpoolClient` (the local
telemetry spool, no hub); both have the same methods (`ping`, `whoami`, `runs`, `status`, `log`,
`scalars`, `frame`, `board`, `send`, `wait`). A hub that cannot be reached, or an answer that is not
the API, raises `StudioError` (its `code` is 0 when nothing answered).

## The board

The Studio's **Board** screen shows the same state for every run, who is piloting them (people
and agent tokens, with their scope and their last action) and the recent commands. It is read
only. The JSON behind it is `GET /api/v1/board`; the whole API is described in the hub's module
documentation (`conatus.studio.hub`).
