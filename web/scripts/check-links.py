"""Relative links of a built site that point to no file (portable build: file:// has no server).

    python web/scripts/check-links.py scratch/web_portable

Checks every href/src of every .html file that is relative or root-relative (external links,
mailto:, data: and pure #fragments are skipped); a link to a directory needs its index.html.
Prints the number of links checked and each broken one; exit code 1 if any is broken.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ATTR = re.compile(r'''(?:href|src)\s*=\s*["']([^"']+)["']''')


def main(root: Path) -> int:
    n, bad = 0, []
    for page in sorted(root.rglob("*.html")):
        for url in ATTR.findall(page.read_text(errors="replace")):
            u = urlsplit(url)
            if u.scheme or url.startswith(("#", "//")) or not u.path:
                continue
            path = unquote(u.path)
            target = (root / path.lstrip("/")) if path.startswith("/") else (page.parent / path)
            if target.is_dir():
                target = target / "index.html"
            n += 1
            if not target.exists():
                bad.append(f"{page.relative_to(root)}: {url}")
    for b in bad[:50]:
        print("broken:", b)
    print(f"links checked: {n}, broken: {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
