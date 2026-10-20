# The Studio

The Studio shows a running simulation as a film, from a browser or a phone, and lets you pause,
checkpoint, stop or fork it; watching never changes a bit of the computation. The Studio and its
hub are experimental in this release ({doc}`experimental`): `sim.watch` records the telemetry as
documented, the page, the server, the hub and the client may change.

```{literalinclude} _check_features.py
:language: python
:start-after: "# [watch]"
:end-before: "# [/watch]"
:dedent: 4
```

```{literalinclude} outputs/studio_watch.txt
:language: text
```

The run is the same with and without the watch, bit for bit: the Studio only reads.

## Only what you declare

`sim.watch(...)` says what the Studio may show, and nothing else is sent:

- **film**: the views of the film, by name. A view is an unknown (`"c"`), a function of the
  simulation that returns an array (`vorticity(sim)`), or `cn.view(source, scale=...)` to choose
  the colour scale: `smoke` (one-signed), `signed` (zero in black) or `dye`. A 3D field is shown
  as an isosurface, a volume or slices (`cn.view(q, mode="isosurface")`).
- **stats**: numbers and curves. A function returns one number per call (a time series);
  `cn.series`, `cn.spectrum` and `cn.profile` declare curves with their formula.
- **every**: the wall-clock time between two frames of a view.

`w = sim.watch(...)` returns the watch; `w.explain()` (and `sim.explain()`) lists what will be
sent and what is lost on the way: frames are stored on 8 bits and decimated, and the numbers of
the computation are never touched.

## What you see

1. **The gallery**: one card per run, live runs first.
2. **The film**: the flow in large, on a dark background. Pinch or scroll to zoom, drag to move,
   double-tap to return to the initial view; the view chips switch between the declared views.
3. **The timeline**: play, 1× to 8×, *Live* (the end of the film is the present of the run), drag
   to rewind. The film is kept for the whole run, losslessly re-encoded, within a declared size
   budget; older frames are thinned out, never silently.
4. Three sheets, closed by default: **Stats** (the declared curves, synchronised with the
   playhead), **Controls** (pause, resume, checkpoint; stop and fork ask for a second tap),
   **Details** (provenance, settings, `explain()`, logs and what was lost). A time step declared
   settable (`sim.watch(..., settable={"dt": (lo, hi)})`) can be changed from Controls, within
   its bounds.
5. **The board**: every run with what needs attention (a non-finite value, a silent run, a relay
   down...), who is piloting it, and the recent commands; read only.

A second screen compares one quantity across two to six runs. On a phone the Studio can be added
to the home screen. The link is always shown: *live* with its round-trip time, *reconnecting*
with the next attempt (tap to retry now), or *offline*; the Studio reconnects by itself when the
page comes back or the network returns, and a light stream (no 3D data, fewer points) can be
chosen on the board for a slow connection.

Scripts and agents read the same state and send the same commands: {doc}`studio_agents`.

## Opening the Studio

The computation writes its film and numbers to a spool directory; the Studio reads it. Choose who
serves it:

- **On a cluster.** The server runs inside a batch job on a compute node, never on a login node.
  A tunnel through the login node brings it to your laptop:

  ```bash
  ssh -J <login-node> -L 8765:localhost:8765 <compute-node>
  # then open http://localhost:8765/login with the one-session token of the job's user_token file
  # (0600; never printed in the job log, which other members of the group can read)
  ```

- **On your own machine.** `python -m conatus.studio.hub serve --local --spool <dir> --port 8765`.
  The one-session login token goes to a 0600 file, `hub_state/local_user_token` (`--state DIR`, or
  `--user-token-out FILE`), written atomically and never through a symbolic link; only its path is
  printed. `--print-token` prints the token itself (it then lands in any log of the process).

The Studio needs `tornado` (`pip install "conatus[studio]"`); the library never
imports it.


## Without the native layer

An expert time stepper is watched with `conatus.telemetry.Telemetry`, which takes the same
declarations (`views=`, `stats=`) and gives a callback for `solver.run(n, callback=..., every=...)`
and a wrapper for Newton–Krylov searches.

## Limits and timings

Every number the Studio, the hub, the relay and the telemetry use on their own (when a quiet run
counts as *silent*, how long a confirmation or a session lives, how many failed logins are allowed,
how large a request may be, how much film the hub keeps) is declared in one table,
`conatus.studio.constants`, and printed by
`python -c "import conatus.studio.constants as c; print(c.explain())"`. A test fails if the code and
the table ever differ, or if a new constant is not listed.

```{include} _generated/studio_constants.md
```

