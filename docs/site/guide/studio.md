# The Studio

The Studio shows a running simulation as a film, from a browser or a phone, and lets you pause,
checkpoint, stop or fork it. Watching does not change the computation. The Studio and its hub are
experimental in this release ({doc}`experimental`). `sim.watch` records the telemetry as
documented; the page, the server, the hub and the client may change.

```{literalinclude} _check_features.py
:language: python
:start-after: "# [watch]"
:end-before: "# [/watch]"
:dedent: 4
```

```{literalinclude} outputs/studio_watch.txt
:language: text
```

Telemetry reads the declared fields and does not change them: the run gives the same numbers, bit
for bit, with and without the watch (the check script runs the same simulation without the watch
and compares the two final states). The Controls panel can send pause, checkpoint and stop
commands.

## Only what you declare

`sim.watch(...)` declares what the Studio may show. Only these items are sent:

- **film**: the views of the film, by name. A view is an unknown (`"c"`), a function of the
  simulation that returns an array (`vorticity(sim)`), or `cn.view(source, scale=...)` to choose
  the colour scale: `smoke` (one-signed), `signed` (zero in black) or `dye`. A 3D field is shown
  as an isosurface, a volume or slices (`cn.view(q, mode="isosurface")`).
- **stats**: numbers and curves. A function returns one number per call (a time series);
  `cn.series`, `cn.spectrum` and `cn.profile` declare curves with their formula.
- **every**: the wall-clock time between two frames of a view.

`w = sim.watch(...)` returns the watch; `w.explain()` (and `sim.explain()`) lists what will be
sent and what is lost on the way: frames are stored on 8 bits and decimated. Watching does not
change the numbers of the computation.

## What you see

1. **The gallery**: one card per run, live runs first.
2. **The film**: the flow in large, on a dark background. Pinch or scroll to zoom, drag to move,
   double-tap to return to the initial view; the view chips switch between the declared views.
3. **The timeline**: play, 1× to 8×, *Live* (the end of the film is the present of the run), drag
   to rewind. The film is kept for the whole run, losslessly re-encoded, within a declared size
   budget. When the budget is exceeded, older frames are removed and the interface reports the
   thinning.
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

```{figure} media/studio_local.svg
:alt: Inside your user account, the simulation writes to a spool directory; the Studio server reads it, writes a one-session token to a 0600 file whose path it prints, and serves a page on 127.0.0.1. The browser opens that page, directly or through an ssh tunnel, and logs in with the token.
:width: 100%

The local Studio. Everything in the dashed box runs under your account. The server listens on the
loopback address only; the token file is readable by you alone.
```

- **On a cluster.** Start the server inside an allocated compute job, following the cluster's
  policy. A tunnel through the login node brings it to your laptop:

  ```bash
  ssh -J <login-node> -L 8765:localhost:8765 <compute-node>
  # then open http://localhost:8765/login with the one-session token of the job's user_token file
  # (0600; never printed in the job log, which other members of the group can read)
  ```

- **On your own machine.** `python -m conatus.studio.hub serve --local --spool <dir> --port 8765`.
  The one-session login token goes to a 0600 file named `local_user_token`, written atomically and
  never through a symbolic link. The command prints the path of that file; read the token there.
  Without `--state`, the file is in a new private directory `conatus_hub_*` under the system
  temporary directory, which changes at every start and is not removed when the hub stops. For a
  fixed place, pass `--state ./hub_state` (or set `$CN_HUB_STATE`), or `--user-token-out FILE`.
  `--print-token` prints the token itself (it then lands in any log of the process).

The Studio needs `tornado` (`pip install "conatus[studio]"`). The solver library
does not import it.

**Other services on `localhost`.** Browsers keep cookies per host name, not per port. After you
sign in at `http://localhost:8765`, the browser also sends the Studio's session cookie to any other
web service you open on `localhost`, whatever its port. Such a service could reuse the cookie to
act in your Studio until the session ends. To avoid it, open the Studio at a name of its own, such
as `http://studio.localhost:8765` (Chrome, Edge and Firefox send `*.localhost` to this machine
without DNS), and sign in there. The cookie is then sent only to `studio.localhost`. A session
works only at the name and port where you signed in.


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

