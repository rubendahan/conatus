"""Every tunable constant of the Studio, the hub, the relay and the telemetry, in one table (A3).

Critique passes 2 and 3 (A3): the timings, limits and security settings of the Studio were
scattered over nine modules (``SILENT_S``, ``LIVE_S``, ``CONFIRM_TTL_S``, ``max_per_hour``,
``LoginThrottle(5, 900)``, ...), each a reasonable choice, none listed where a user or an
operator could read them all. :data:`TUNABLES` is that list, like :mod:`conatus.tolerances` for
the numerical tolerances: each entry names where the value lives (a module
constant ``module.NAME``, or the default of an argument ``module.Callable(arg)``), the value, its
unit, what it decides, and how to change it. :func:`explain` prints the table.

The development test suite fails when a listed value differs from the code, or when
a numeric module constant, a numeric default of a public callable or of a public method, or a
numeric default of a command-line option (``argparse``, read with ``ast``: :func:`cli_defaults`)
of ``conatus.studio`` / ``conatus.telemetry`` is not listed (:data:`NOT_TUNABLE` holds the few
numbers that are not settings: protocol versions, error codes, record offsets, query starts). Changing a value means changing the
code and this table in the same commit.

This module imports nothing from the Studio (no tornado): ``python -c "import
conatus.studio.constants as c; print(c.explain())"`` works with the library alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Tunable", "TUNABLES", "NOT_TUNABLE", "explain", "lookup", "cli_defaults"]


@dataclass(frozen=True)
class Tunable:
    """One declared constant: where it lives, its value and unit, what it decides, how to change it."""

    where: str        # "conatus.studio.agentapi.SILENT_S" or "conatus.studio.auth.LoginThrottle(window_s)"
    value: Any
    unit: str
    kind: str         # "timing" | "limit" | "security" | "display"
    meaning: str
    change: str = "edit the constant (and this table)"


_S, _B = "s", "bytes"
_ARG = "argument of the constructor (and this table for the default)"

TUNABLES: tuple[Tunable, ...] = (
    # ---- agent API (conatus.studio.agentapi): what an agent sees ------------------------------------
    Tunable("conatus.studio.agentapi.SILENT_S", 60.0, _S, "timing",
            "a live run with no record for longer gets the alert 'silent'"),
    Tunable("conatus.studio.agentapi.LIVE_S", 30.0, _S, "timing",
            "a record within the last LIVE_S seconds makes a run 'running' (else 'stale')"),
    Tunable("conatus.studio.agentapi.CONFIRM_TTL_S", 120.0, _S, "security",
            "a confirmation token (stop, fork) is valid this long, and once"),
    Tunable("conatus.studio.agentapi.Confirmations(ttl_s)", 120.0, _S, "security",
            "same, for a Confirmations store built by hand", _ARG),
    Tunable("conatus.studio.agentapi.MAX_PARAMS_BYTES", 4096, _B, "limit",
            "fork/set parameters: a small JSON object at most"),
    Tunable("conatus.studio.agentapi.last_values(n)", 200, "values", "display",
            "scalars returned by an agent's 'last values' query", "argument"),
    Tunable("conatus.studio.agentapi.run_view(log)", 20, "lines", "display",
            "log lines in an agent's view of a run", "argument"),
    Tunable("conatus.studio.agentapi.run_view(events)", 20, "events", "display",
            "events in an agent's view of a run", "argument"),
    Tunable("conatus.studio.agentapi.frame_png(upscale)", 0, "factor", "display",
            "nearest-neighbour upscaling of an agent's frame PNG (0: to about 256 px across)", "argument"),
    # ---- authentication (conatus.studio.auth) ----------------------------------------------------
    Tunable("conatus.studio.auth.SCRYPT_N", 16384, "iterations", "security",
            "scrypt cost parameter N of stored password hashes"),
    Tunable("conatus.studio.auth.SCRYPT_R", 8, "", "security", "scrypt block size r"),
    Tunable("conatus.studio.auth.SCRYPT_P", 1, "", "security", "scrypt parallelisation p"),
    Tunable("conatus.studio.auth.hash_password(n)", 16384, "iterations", "security",
            "= SCRYPT_N (default of the argument)", "argument"),
    Tunable("conatus.studio.auth.hash_password(r)", 8, "", "security", "= SCRYPT_R", "argument"),
    Tunable("conatus.studio.auth.hash_password(p)", 1, "", "security", "= SCRYPT_P", "argument"),
    Tunable("conatus.studio.auth.TOKEN_BYTES", 32, _B, "security",
            "random bytes of relay, user and agent tokens and session cookies (256 bits)"),
    Tunable("conatus.studio.auth.MIN_PASSWORD", 12, "characters", "security",
            "shorter passwords are refused at creation"),
    Tunable("conatus.studio.auth.Sessions(ttl_s)", 43200, _S, "security",
            "lifetime of a browser session cookie (12 h)", _ARG),
    Tunable("conatus.studio.auth.LoginThrottle(max_failures)", 5, "failures", "security",
            "failed password logins per client address within the window before refusal "
            "(a valid token is never refused; all clients of a compute node share 127.0.0.1)", _ARG),
    Tunable("conatus.studio.auth.LoginThrottle(window_s)", 900.0, _S, "security",
            "sliding window of the login throttle (15 min)", _ARG),
    # ---- clients (conatus.studio.client) ---------------------------------------------------------
    Tunable("conatus.studio.client.TAIL_BYTES", 8 << 20, _B, "limit",
            "spool transport: bytes of a run file read from its end"),
    Tunable("conatus.studio.client.read_run(tail_bytes)", 8 << 20, _B, "limit", "= TAIL_BYTES", "argument"),
    Tunable("conatus.studio.client.SpoolClient(tail_bytes)", 8 << 20, _B, "limit", "= TAIL_BYTES", _ARG),
    Tunable("conatus.studio.client.read_run(max_scalars)", 20000, "records", "limit",
            "scalar records kept per run when reading a spool file", "argument"),
    Tunable("conatus.studio.client.read_run(max_logs)", 2000, "lines", "limit",
            "log lines kept per run when reading a spool file", "argument"),
    Tunable("conatus.studio.client.HubClient(timeout)", 30.0, _S, "timing",
            "HTTP timeout of a hub client request", _ARG),
    Tunable("conatus.studio.client.connect(timeout)", 30.0, _S, "timing",
            "HTTP timeout of conatus.studio.connect()", "argument"),
    # ---- comparison (conatus.studio.compare) -----------------------------------------------------
    Tunable("conatus.studio.compare.MAX_RUNS", 8, "runs", "display",
            "runs compared at once (the page's categorical palette has 8 slots, never cycled)"),
    Tunable("conatus.studio.compare.compare_runs(max_points)", 2000, "points", "display",
            "points per curve of a comparison (uniform stride)", "argument"),
    # ---- hub (conatus.studio.hub) ----------------------------------------------------------------
    Tunable("conatus.studio.hub.MAX_PARAMS_BYTES", 4096, _B, "limit",
            "fork/set parameters accepted by the hub"),
    Tunable("conatus.studio.hub.ACK_TIMEOUT_S", 10.0, _S, "timing",
            "how long the hub waits for the relay to acknowledge a command"),
    Tunable("conatus.studio.hub.FILM_MAX_FRAMES", 720, "frames", "limit",
            "film frames kept per run and name in the hub's memory (then thinned)"),
    Tunable("conatus.studio.hub.MAX_BODY_BYTES", 1 << 20, _B, "security",
            "largest HTTP request body (commands and jobs are a few kB)"),
    Tunable("conatus.studio.hub.MAX_WAITS", 4, "requests", "security",
            "concurrent long polls per caller (more: HTTP 429)"),
    Tunable("conatus.studio.hub.AUDIT_FIELD_BYTES", 512, _B, "limit",
            "a command's fields longer than this are not copied whole into the audit log"),
    Tunable("conatus.studio.hub.Hub(max_scalars)", 20000, "records", "limit",
            "scalar records kept per run in the hub's memory", _ARG),
    Tunable("conatus.studio.hub.Hub(max_logs)", 2000, "lines", "limit",
            "log lines kept per run in the hub's memory", _ARG),
    Tunable("conatus.studio.hub.Hub(max_runs)", 200, "runs", "limit",
            "runs kept by the hub (beyond it the stalest run is evicted)", _ARG),
    Tunable("conatus.studio.hub.Hub(session_ttl_s)", 43200, _S, "security",
            "browser session lifetime on the hub (12 h)", _ARG),
    Tunable("conatus.studio.hub.Hub(film_mb)", 512.0, "MiB", "limit",
            "film memory budget of the hub", "hub serve --film-mb (default 512)"),
    Tunable("conatus.studio.hub.Hub(film_max_frames)", 720, "frames", "limit",
            "= FILM_MAX_FRAMES", _ARG),
    # ---- job launcher (conatus.studio.launch) ----------------------------------------------------
    Tunable("conatus.studio.launch.MAX_PARAMS_BYTES", 4096, _B, "security",
            "parameters of a job prepared from the Studio"),
    Tunable("conatus.studio.launch.LOG_TAIL_BYTES", 16384, _B, "display",
            "bytes of a job log shown in the Studio"),
    Tunable("conatus.studio.launch.Launcher(max_per_hour)", 20, "submissions", "security",
            "Slurm submissions from the Studio per sliding hour (more: HTTP 429)", _ARG),
    # ---- run registry page (conatus.studio.registry) ---------------------------------------------
    Tunable("conatus.studio.registry.MAX_LIST", 500, "rows", "display",
            "registry rows sent to the page at most"),
    # ---- relay (conatus.studio.relay) ------------------------------------------------------------
    Tunable("conatus.studio.relay.SLURM_MIN_PERIOD_S", 120.0, _S, "security",
            "smallest period of the relay's job-queue poll (at least 120 s between two scheduler calls)"),
    Tunable("conatus.studio.relay.MAX_PARAMS_BYTES", 4096, _B, "security",
            "fork parameters accepted by the relay"),
    Tunable("conatus.studio.relay.FILM_BYTES_PER_PUSH", 2 << 20, _B, "limit",
            "film history sent per poll at most (live data goes first)"),
    Tunable("conatus.studio.relay.MAX_LINE_BYTES", 4 << 20, _B, "security",
            "a single telemetry line longer than this is skipped and logged"),
    Tunable("conatus.studio.relay.PING_INTERVAL_S", 20.0, _S, "timing",
            "WebSocket ping period of the relay"),
    Tunable("conatus.studio.relay.PING_TIMEOUT_S", 20.0, _S, "timing",
            "WebSocket ping timeout of the relay (then reconnect)"),
    Tunable("conatus.studio.relay.Relay(poll_s)", 1.0, _S, "timing",
            "spool poll period", "relay --poll"),
    Tunable("conatus.studio.relay.Relay(max_rate)", 256000.0, "bytes/s", "limit",
            "upload rate limit (token bucket)", _ARG),
    Tunable("conatus.studio.relay.Relay(max_batch)", 131072, _B, "limit",
            "largest message sent at once", _ARG),
    Tunable("conatus.studio.relay.Relay(backoff_s)", 1.0, _S, "timing",
            "first reconnection delay (doubled up to max_backoff_s)", _ARG),
    Tunable("conatus.studio.relay.Relay(max_backoff_s)", 60.0, _S, "timing",
            "longest reconnection delay", _ARG),
    Tunable("conatus.studio.relay.Relay(slurm_s)", 120.0, _S, "timing",
            "job-queue poll period of the relay (>= SLURM_MIN_PERIOD_S; 0 = off)", "relay --slurm"),
    Tunable("conatus.studio.relay.Relay(connect_timeout_s)", 20.0, _S, "timing",
            "WebSocket connection timeout", _ARG),
    Tunable("conatus.studio.relay.read_header(max_bytes)", 1 << 20, _B, "security",
            "largest spool header read", "argument"),
    # ---- local Studio server (conatus.studio.webapp) ---------------------------------------------
    Tunable("conatus.studio.webapp.MAX_FRAMES_PER_REQUEST", 64, "frames", "limit",
            "= conatus.telemetry.film.MAX_FRAMES_PER_REQUEST (re-exported)"),
    Tunable("conatus.studio.webapp.LITE_MAX_SCALARS", 2000, "records", "limit",
            "scalars per run in a light snapshot (uniform stride, last kept)"),
    # ---- telemetry (conatus.telemetry) -----------------------------------------------------------
    Tunable("conatus.telemetry.film.MAX_FRAMES_PER_REQUEST", 64, "frames", "limit",
            "frames in one HTTP answer of the film endpoint"),
    Tunable("conatus.telemetry.film.MAX_BYTES_PER_REQUEST", 8 << 20, _B, "limit",
            "about this many bytes per film answer (one frame at least)"),
    Tunable("conatus.telemetry.film.READ_CHUNK", 16 << 20, _B, "limit",
            "film file indexed per poll (a 256 MiB film: 16 polls)"),
    Tunable("conatus.telemetry.sink.FILM_GROWTH", 4, "factor", "limit",
            "film thinning: level j keeps every FILM_GROWTH**j-th frame"),
    Tunable("conatus.telemetry.sink.Telemetry(control_interval)", 1.0, _S, "timing",
            "minimum wall time between two reads of the control file", _ARG),
    Tunable("conatus.telemetry.sink.Telemetry(max_px3d)", 48, "points/axis", "limit",
            "3D frames strided to at most this many points per axis", _ARG),
    Tunable("conatus.telemetry.sink.Telemetry(max_points)", 16384, "points", "limit",
            "points of a particle frame", _ARG),
    Tunable("conatus.telemetry.sink.Telemetry(film_max_bytes)", 0, _B, "limit",
            "film budget of a bare Telemetry (0 = no film; sim.watch passes film_mb)", _ARG),
    Tunable("conatus.telemetry.protocol.encode_curve(max_points)", 4096, "points", "limit",
            "points of a curve record", "argument"),
    Tunable("conatus.telemetry.protocol.film_record(compress_level)", 6, "zlib level", "limit",
            "deflate level of film frames", "argument"),
    # ---- defaults of methods and of the command lines (the first table left them out)
    Tunable("conatus.studio.client.HubClient.status(log)", 20, "lines", "display",
            "log lines of a run's full view", "argument"),
    Tunable("conatus.studio.client.HubClient.status(events)", 20, "events", "display",
            "events of a run's full view", "argument"),
    Tunable("conatus.studio.client.HubClient.scalars(max)", 2000, "rows", "display",
            "scalar rows returned at most (uniform stride)", "argument"),
    Tunable("conatus.studio.client.HubClient.log(n)", 100, "records", "display",
            "log records and events returned", "argument"),
    Tunable("conatus.studio.client.HubClient.wait(timeout)", 25.0, _S, "timing",
            "long-poll period of a wait for a change", "argument"),
    Tunable("conatus.studio.client.SpoolClient.status(log)", 20, "lines", "display",
            "= HubClient.status(log)", "argument"),
    Tunable("conatus.studio.client.SpoolClient.status(events)", 20, "events", "display",
            "= HubClient.status(events)", "argument"),
    Tunable("conatus.studio.client.SpoolClient.scalars(max)", 2000, "rows", "display",
            "= HubClient.scalars(max)", "argument"),
    Tunable("conatus.studio.client.SpoolClient.log(n)", 100, "records", "display",
            "= HubClient.log(n)", "argument"),
    Tunable("conatus.studio.client.SpoolClient.wait(timeout)", 25.0, _S, "timing",
            "= HubClient.wait(timeout)", "argument"),
    Tunable("conatus.studio.client.SpoolClient.wait(poll_s)", 0.5, _S, "timing",
            "spool file poll period of a wait", "argument"),
    Tunable("conatus.studio.hub.Hub.pilots_of(n)", 3, "users", "display",
            "last distinct senders of a command shown for a run", "argument"),
    Tunable("conatus.studio.hub.Hub.run_view(log)", 20, "lines", "display",
            "= agentapi.run_view(log), on the hub", "argument"),
    Tunable("conatus.studio.hub.Hub.run_view(events)", 20, "events", "display",
            "= agentapi.run_view(events), on the hub", "argument"),
    Tunable("conatus.telemetry.film.FilmIndex.frames(limit)", 64, "frames", "limit",
            "= MAX_FRAMES_PER_REQUEST (frames per call)", "argument"),
    Tunable("conatus.telemetry.sink.Telemetry.close(timeout)", 30.0, _S, "timing",
            "wait for the writer thread to flush the last records at close", "argument"),
    Tunable("conatus.studio.cli --timeout", 30.0, _S, "timing",
            "= HubClient(timeout), from the command line", "python -m conatus studio --timeout"),
    Tunable("conatus.studio.cli --log", 20, "lines", "display", "= HubClient.status(log)",
            "studio status --log"),
    Tunable("conatus.studio.cli --events", 20, "events", "display", "= HubClient.status(events)",
            "studio status --events"),
    Tunable("conatus.studio.cli --timeout->wait_timeout", 25.0, _S, "timing",
            "= HubClient.wait(timeout)", "studio watch --timeout"),
    Tunable("conatus.studio.cli -n", 100, "records", "display", "= HubClient.log(n)", "studio log -n"),
    Tunable("conatus.studio.cli --max", 2000, "rows", "display", "= HubClient.scalars(max)",
            "studio scalars --max"),
    Tunable("conatus.studio.hub --film-mb", 512.0, "MiB", "limit", "= Hub(film_mb)",
            "hub serve --film-mb"),
    Tunable("conatus.studio.hub --slurm", 0.0, _S, "timing",
            "job-queue poll period of the local relay of `hub serve --local` (0 = off)", "hub serve --slurm"),
    Tunable("conatus.studio.relay --poll", 1.0, _S, "timing", "= Relay(poll_s)", "relay --poll"),
    Tunable("conatus.studio.relay --max-rate", 256000.0, "bytes/s", "limit", "= Relay(max_rate)",
            "relay --max-rate"),
    Tunable("conatus.studio.relay --max-backoff", 60.0, _S, "timing", "= Relay(max_backoff_s)",
            "relay --max-backoff"),
)

# Numbers that are not settings (excluded from the completeness test, with the reason).
NOT_TUNABLE: dict[str, str] = {
    "conatus.studio.agentapi.API_VERSION": "version of the agent API (a contract, not a setting)",
    "conatus.studio.relay.PROTOCOL_VERSION": "telemetry protocol version (re-exported)",
    "conatus.telemetry.protocol.PROTOCOL_VERSION": "telemetry protocol version",
    "conatus.telemetry.sink.PROTOCOL_VERSION": "telemetry protocol version (re-exported)",
    "conatus.studio.client.StudioError(code)": "HTTP status code of an error object",
    "conatus.studio.launch.LaunchError(code)": "HTTP status code of an error object",
    "conatus.telemetry.film.FilmEntry(offset)": "field of a film index record",
    "conatus.telemetry.film.FilmEntry(length)": "field of a film index record",
    "conatus.studio.client.HubClient.scalars(since)": "start of a query (0 = from the beginning)",
    "conatus.studio.client.SpoolClient.scalars(since)": "start of a query (0 = from the beginning)",
    "conatus.studio.cli --since": "start of a query (0 = from the beginning)",
    "conatus.studio.cli --count": "stop after this many lines (0 = until the run ends): a request",
    "conatus.studio.cli --for->duration": "watch duration (0 = until the run ends): a request",
    "conatus.studio.hub.Hub.command(api)": "which route sent a command (0 = browser), not a setting",
    "conatus.studio.hub.V1.answer(code)": "HTTP status code of an answer",
    "conatus.telemetry.film.FilmIndex.frames(start)": "start of a query (film index)",
    "conatus.telemetry.film.FilmIndex.frames(step)": "stride of a query (1 = every frame)",
}


def cli_defaults(module: str) -> dict[str, Any]:
    """Numeric defaults of the command-line options of ``module`` (read from its source with
    ``ast``, nothing is run): ``"--flag"`` or ``"--flag->dest"`` (when ``dest=`` is given) -> value."""
    import ast
    import importlib.util
    spec = importlib.util.find_spec(module)
    tree = ast.parse(open(spec.origin, encoding="utf-8").read())
    out: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        d = kw.get("default")
        if not isinstance(d, ast.Constant) or isinstance(d.value, bool) \
                or not isinstance(d.value, (int, float)):
            continue
        dest = kw.get("dest")
        name = node.args[0].value + (f"->{dest.value}" if isinstance(dest, ast.Constant) else "")
        out[name] = d.value
    return out


def lookup(where: str) -> Any:
    """The value in the code of an entry: ``module.NAME``, ``module.Callable(arg)``,
    ``module.Class.method(arg)`` or ``module --flag`` (a command-line default)."""
    import importlib
    import inspect
    if " " in where:                                   # command-line option of a module
        mod, flag = where.split(" ", 1)
        return cli_defaults(mod)[flag]
    if where.endswith(")"):
        target, arg = where[:-1].split("(")
        mod, name = target.rsplit(".", 1)
        try:
            obj = getattr(importlib.import_module(mod), name)
        except ModuleNotFoundError:                    # module.Class.method(arg)
            mod, cls = mod.rsplit(".", 1)
            obj = getattr(getattr(importlib.import_module(mod), cls), name)
        fn = obj.__init__ if inspect.isclass(obj) else obj
        return inspect.signature(fn).parameters[arg].default
    mod, name = where.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)


def explain(kind: str | None = None) -> str:
    """The table, one line per constant (``kind``: only "timing", "limit", "security" or
    "display")."""
    rows = [t for t in TUNABLES if kind is None or t.kind == kind]
    lines = [f"Studio, hub, relay and telemetry constants ({len(rows)}; conatus.studio.constants):"]
    for t in rows:
        unit = f" {t.unit}" if t.unit else ""
        lines.append(f"  {t.where} = {t.value!r}{unit} [{t.kind}]: {t.meaning}; change: {t.change}")
    return "\n".join(lines)
