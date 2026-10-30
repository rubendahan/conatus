"""F793 (audit of 0.2.1, N03, F44). The server starts first on a spool directory that does not exist
yet, and a run started afterwards is seen live.

Oracle: the hub's own answer about a run that is still advancing, over HTTP."""
from _acceptance import check, passed, skipped, start

start(793, "Studio first view: server, then run")
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402

try:
    import tornado  # noqa: F401
except ImportError:
    skipped("needs conatus[studio]")

work = Path(tempfile.mkdtemp())
port = 18000 + os.getpid() % 2000
log = open(work / "hub.log", "w")
hub = subprocess.Popen([sys.executable, "-m", "conatus.studio.hub", "serve", "--local", "--spool", "studio_spool",
                        "--port", str(port)], cwd=work, stdout=log, stderr=subprocess.STDOUT)
try:
    for _ in range(600):
        time.sleep(0.2)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/login", timeout=2)
            break
        except OSError:
            continue
    else:
        check(False, "the server did not answer: " + (work / "hub.log").read_text()[-400:])
    script = """
import numpy as np, conatus as cn
box = cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5))
m = cn.Model(box, name="rb", Ra=2e4, Pr=1)
m.velocity("u"), m.pressure("p"), m.scalar("T")
m.equations("dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)\\ndiv(u) = 0\\ndt(T) - lap(T) = -u·grad(T)")
m.at("bottom", u=0, T=+0.5)
m.at("top", u=0, T=-0.5)
sim = m.simulate(N=(32, 16), scheme="RK443", dt=1e-3)
sim.initial(T=lambda x, z: -z + 1e-3 * np.cos(np.pi * x) * np.cos(np.pi * z))
sim.watch("rb_demo", film={"T": cn.view("T", scale="signed")}, every_s=0.5, spool="studio_spool")
t0 = __import__("time").perf_counter()
sim.run(steps=6000)
print("elapsed", __import__("time").perf_counter() - t0, "max|T|", float(np.abs(sim.grid("T")).max()))
sim.unwatch()
"""
    run = subprocess.Popen([sys.executable, "-c", script], cwd=work, stdout=subprocess.PIPE, text=True)
    time.sleep(6.0)
    alive = run.poll() is None
    token_file = re.search(r"\S*local_user_token\S*", (work / "hub.log").read_text())
    check(token_file is not None, "the server printed no token file")
    token = Path(token_file.group(0).rstrip(".,:;)")).read_text().strip()

    def live_runs():
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/runs", headers={"Authorization": f"Bearer {token}"})
        return json.loads(urllib.request.urlopen(req, timeout=5).read())
    seen = {}
    while run.poll() is None and "rb_demo" not in json.dumps(seen):
        seen = live_runs()
        time.sleep(1.0)
    files = sorted(p.name for p in (work / "studio_spool").rglob("*"))[:20] if (work / "studio_spool").exists() else []
    out, _ = run.communicate(timeout=600)
    check("rb_demo" in json.dumps(seen), f"the server does not list the running run: {seen}")
    time.sleep(3.0)
    check("rb_demo" in json.dumps(live_runs()), "the run left the list when it ended")
    check(alive, f"the run ended within 6 s: {out}")
    check(run.returncode == 0, f"run exit {run.returncode}: {out}")
    check(any("rb_demo" in f for f in files), f"no spool record of the run: {files}")
    m = re.search(r"elapsed ([\d.]+) max\|T\| ([\d.eE+-]+)", out)
    check(m and float(m.group(2)) < 2.0, f"run unphysical: {out}")
    print(json.dumps({"elapsed_s": float(m.group(1)), "max_T": float(m.group(2))}))
finally:
    hub.terminate()
    hub.wait(20)
passed("the server lists the run while it advances and after it ends")
