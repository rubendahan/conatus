"""Conatus Studio, remote access: outbound relay and hosted hub.

Optional plug, never required by the library. Needs the ``studio`` extra (``pip install
".[studio]"`` from the source tree, i.e. tornado). Importing this package imports nothing heavy; the modules are
used explicitly:

* :mod:`conatus.studio.relay` runs next to the spool (inside a job, on a compute node), opens
  an **outbound** WebSocket to the hub, pushes the telemetry spool and writes the commands it
  receives into ``<run_id>.control.jsonl`` (protocol v1 unchanged). No port is opened on the cluster.
* :mod:`conatus.studio.hub` runs outside the cluster (small VM, Fly.io, Render), receives several
  relays, serves the webapp to authenticated users over HTTPS and forwards their commands.
* :mod:`conatus.studio.auth` (stdlib only): password hashing, token hashing, sessions, audit log.

 § "Remote access"; deployment: ``studio/hub/DEPLOY.md``.
"""
