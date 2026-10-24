"""Links of a built site that point to no file.

    python web/scripts/check-links.py scratch/web_portable
    python web/scripts/check-links.py SITE --repo https://github.com/OWNER/NAME --tree EXPORTED_TREE
    python web/scripts/check-links.py SITE --repo https://github.com/OWNER/NAME --online

Checks every href/src of every .html file that is relative or root-relative (mailto:, data: and
pure #fragments are skipped); a link to a directory needs its index.html. A link with a fragment
into the documentation (a path under docs/, built by Sphinx) also needs that id in the target page.

With --repo, the links into that repository (<repo>/blob/<ref>/<path> and <repo>/tree/<ref>/<path>)
are checked too, whatever the ref (main or a release tag that does not exist yet):
  --tree DIR   offline: <path> must exist in DIR, the tree that will be pushed and tagged
               (the public export); a typo in a path is caught before the tag exists;
  --online     each distinct URL must answer HTTP 200 (at release, once the tag is pushed).
Prints the number of links checked and each broken one; exit code 1 if any is broken.
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlsplit

ATTR = re.compile(r'''(?:href|src)\s*=\s*["']([^"']+)["']''')
IDS = re.compile(r'''\sid\s*=\s*["']([^"']+)["']''')


def repo_link(url: str, repo: str) -> tuple[str, str] | None:
    """(ref, path) of a link <repo>/(blob|tree)/<ref>/<path>, else None."""
    m = re.match(re.escape(repo.rstrip("/")) + r"/(?:blob|tree)/([^/#?]+)/([^#?]*)", url)
    return (m.group(1), unquote(m.group(2)).rstrip("/")) if m else None


def online_ok(url: str) -> bool:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "conatus-link-check"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except urllib.error.URLError:
        return False


def main(root: Path, repo: str | None = None, tree: Path | None = None, online: bool = False) -> int:
    n, bad = 0, []
    ids: dict[Path, set[str]] = {}                       # target page -> its ids (read once)
    repo_urls: dict[str, str] = {}                     # url -> first page that links it
    for page in sorted(root.rglob("*.html")):
        for url in ATTR.findall(page.read_text(errors="replace")):
            u = urlsplit(url)
            if repo and repo_link(url, repo):
                repo_urls.setdefault(url, str(page.relative_to(root)))
                continue
            if u.scheme or url.startswith(("#", "//")) or not u.path:
                continue
            path = unquote(u.path)
            target = (root / path.lstrip("/")) if path.startswith("/") else (page.parent / path)
            if target.is_dir():
                target = target / "index.html"
            n += 1
            if not target.exists():
                bad.append(f"{page.relative_to(root)}: {url}")
            elif u.fragment and target.suffix == ".html" and "docs" in target.relative_to(root).parts:
                if target not in ids:
                    ids[target] = set(IDS.findall(target.read_text(errors="replace")))
                if unquote(u.fragment) not in ids[target]:
                    bad.append(f"{page.relative_to(root)}: {url} (no id {u.fragment!r})")
    n_repo, refs = 0, set()
    if repo and (tree or online):
        for url, page in sorted(repo_urls.items()):
            ref, path = repo_link(url, repo)
            refs.add(ref)
            n_repo += 1
            ok = online_ok(url) if online else (tree / path).exists()
            if not ok:
                bad.append(f"{page}: {url}" + ("" if online else f" (no {path} in {tree})"))
    for b in bad[:50]:
        print("broken:", b)
    extra = f", repository links checked: {n_repo} (refs {', '.join(sorted(refs)) or '-'})" if repo else ""
    print(f"links checked: {n}{extra}, broken: {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("site", type=Path)
    ap.add_argument("--repo", help="repository URL whose blob/tree links are checked")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--tree", type=Path, help="offline: the tree those links must resolve in")
    g.add_argument("--online", action="store_true", help="each repository link must answer HTTP 200")
    a = ap.parse_args()
    if a.repo and not (a.tree or a.online):
        ap.error("--repo needs --tree or --online")
    sys.exit(main(a.site, a.repo, a.tree, a.online))
