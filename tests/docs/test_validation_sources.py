"""The Validation page links a row to a public script only when that script prints the row's number
(external review of the live 0.1.2 site, chain of evidence).

web/src/content/reproduced.json lists, per ledger row, the script, the command, the file where its
output is recorded, the printed text and the pairs [printed number, row number]. These tests check
that every listed file exists, that the printed text appears verbatim in the recorded output, that
the recorded output belongs to the script as it is now (sha256 in the header, when the output has
one), and that each printed number agrees with the row's number to the printed precision. When the
site data has been generated (web/src/generated/ledger.json), the row number must also appear in
that row of the ledger. The oracle is the recorded output of an independent run of the script, not
the page's own text.
"""
from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = json.loads((ROOT / "web/src/content/reproduced.json").read_text())
LEDGER = ROOT / "web/src/generated/ledger.json"
CASES = [(rid, k, r) for rid, rs in DATA["rows"].items() for k, r in enumerate(rs)]
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def number(text: str) -> tuple[Decimal, Decimal]:
    """Value of a number as printed ("0.2375", "2.8e-14", "2.8×10⁻¹⁴") and half a unit of its last
    printed digit (the rounding it may carry)."""
    m = re.fullmatch(r"(-?\d+(?:\.(\d+))?)(?:e([-+]?\d+)|×10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+))?", text.strip())
    assert m, f"not a number: {text!r}"
    exp = int(m.group(3) or (m.group(4) or "0").translate(SUP))
    decimals = len(m.group(2) or "")
    return Decimal(m.group(1)).scaleb(exp), Decimal(5).scaleb(exp - decimals - 1)


def test_number_parser():
    """The parser on the three notations of the page and of the outputs."""
    assert number("2.8×10⁻¹⁴") == (Decimal("2.8e-14"), Decimal("5e-16"))
    assert number("1.3e-07") == (Decimal("1.3e-7"), Decimal("5e-9"))
    assert number("5772.22") == (Decimal("5772.22"), Decimal("0.005"))


@pytest.mark.parametrize("row,k,r", CASES, ids=[f"{rid}-{k}" for rid, k, _ in CASES])
def test_printed_number_is_the_row_number(row, k, r):
    script, out = ROOT / r["script"], ROOT / r["output"]
    assert script.is_file(), r["script"]
    assert out.is_file(), r["output"]
    assert r["command"].startswith("python " + r["script"]), r["command"]
    text = out.read_text()
    for p in r["printed"]:
        assert p in text, f"{r['output']} does not contain {p!r}"
    head = text.split("\n", 1)[0]
    full = re.search(r"script sha256 ([0-9a-f]{64})", head)
    short = re.search(r"script sha256 ([0-9a-f]{16})\s*$", head)
    sha = hashlib.sha256(script.read_bytes()).hexdigest()
    if full:
        assert full.group(1) == sha, f"{r['output']} was written by another version of {r['script']}"
    elif short:
        assert short.group(1) == sha[:16], f"{r['output']} was written by another version of {r['script']}"
    printed = " ".join(r["printed"])
    assert r["numbers"], "a linked row names the number it reproduces"
    for p, v in r["numbers"]:
        assert p in printed, f"{p!r} is not in the printed text"
        (a, da), (b, db) = number(p), number(v)
        assert abs(a - b) <= da + db, f"{row}: printed {p} vs row {v}"


@pytest.mark.skipif(not LEDGER.exists(), reason="site data not generated (web/build.sh writes it)")
def test_row_numbers_are_in_the_ledger():
    entries = {e["id"]: e for e in json.loads(LEDGER.read_text())["entries"]}
    for rid, rs in DATA["rows"].items():
        assert rid in entries, f"no ledger row {rid}"
        for r in rs:
            for _, v in r["numbers"]:
                assert v in entries[rid]["value"], f"{rid}: {v!r} not in {entries[rid]['value']!r}"


def test_no_typed_version_in_the_site_data():
    """published.json and reproduced.json once carried a typed "version" that named a release other
    than the one that recorded the outputs, so the pages named a release that had not produced them.
    The pages now read the version from the header of each recorded output (Published.tsx,
    Reproduced.tsx); neither file may carry a version of its own."""
    for name in ("published.json", "reproduced.json"):
        text = (ROOT / "web/src/content" / name).read_text()
        assert '"version"' not in text, f"web/src/content/{name} types a version"


def test_headed_outputs_name_the_version_that_wrote_them():
    """Each recorded output with a header ("# script | script sha256 ... | conatus V | ...") names a
    version in the form Reproduced.tsx and Published.tsx read, and that version is a released one or
    the package's own (a link to a tag that does not exist would be broken)."""
    init = (ROOT / "python/conatus/__init__.py").read_text()
    current = re.search(r'^__version__ = "([^"]+)"', init, re.M).group(1)
    log = ROOT / "CHANGELOG.md"
    released = set(re.findall(r"^## \[(\d+\.\d+\.\d+)\]", log.read_text(), re.M)) if log.exists() else set()
    outputs = {r["output"] for _, _, r in CASES} | {
        str(p.relative_to(ROOT)) for p in (ROOT / "examples/published").glob("*/output.txt")}
    for out in sorted(outputs):
        head = (ROOT / out).read_text().split("\n", 1)[0]
        if not head.startswith("# ") or " | conatus " not in head:
            continue
        m = re.search(r" \| conatus ([^ |]+) \| ", head)
        assert m, f"{out}: header without a readable version: {head}"
        assert m.group(1) in released | {current}, f"{out}: recorded by {m.group(1)}, not a release"
