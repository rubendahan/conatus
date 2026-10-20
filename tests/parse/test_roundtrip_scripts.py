"""tests/parse/roundtrip_scripts.py reports a script that fails or times out as a failed round
trip (review finding 100: it returned 0 when the equations declared before the failure
round-tripped, so a crashing tutorial counted as checked)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roundtrip_scripts as R  # noqa: E402

SCRIPT = '''
from conatus.expert import IVP, CartesianCoordinates, Domain, Field, RealFourier, dt, lap
cs = CartesianCoordinates("x")
D = Domain(cs, [RealFourier(cs["x"], 8, (0, 1), dealias=1)])
u = Field("u", D)
pb = IVP([u], name="heat")
pb.add_equation((dt(u) - lap(u), 0))
{tail}
'''


def _run(tmp_path, tail, *extra):
    s = tmp_path / "s.py"
    s.write_text(SCRIPT.format(tail=tail))
    out = tmp_path / "out.json"
    code = R.main([str(out), *extra, str(s)])
    return code, json.loads(out.read_text())


def test_a_completed_script_passes(tmp_path):
    code, summary = _run(tmp_path, "")
    assert code == 0 and summary["equations"] >= 1 and not summary["scripts_incomplete"]


def test_a_failing_script_fails_even_if_its_equations_round_trip(tmp_path):
    code, summary = _run(tmp_path, "raise RuntimeError('boom')")
    assert summary["ok"] == summary["equations"] >= 1          # what was declared round-trips
    assert code == 1 and summary["scripts_incomplete"][0]["status"] == "exit 1"


def test_a_timed_out_script_fails(tmp_path):
    code, summary = _run(tmp_path, "import time; time.sleep(60)", "--timeout", "8")
    assert code == 1 and summary["scripts_incomplete"][0]["status"].startswith("timeout")
