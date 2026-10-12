// Internal links: a click loads the next page's chunk first, then swaps the page at once (no
// transition effect: the header stays, the new page replaces the old). Without this, the router would
// show an empty page while the chunk loads. Modified clicks, new tabs, downloads, external links,
// same-page anchors and /docs/ (the Sphinx site) are left to the browser.
import { stripBase } from "./base";
import { preload } from "./pages";

type Navigate = (to: string) => void;

function targetOf(a: HTMLAnchorElement): string | null {
  if (a.target && a.target !== "_self") return null;
  if (a.hasAttribute("download") || a.getAttribute("rel")?.includes("external")) return null;
  const href = a.getAttribute("href");
  if (!href) return null;
  if (__HASH_ROUTER__) {
    if (!href.startsWith("#/")) return null;
    return href.slice(1);
  }
  const url = new URL(a.href, location.href);
  if (url.origin !== location.origin) return null;
  const path = stripBase(url.pathname);
  if (path.startsWith("/docs/") || /\.[a-z0-9]+$/i.test(path)) return null;       // Sphinx pages, files
  if (path === stripBase(location.pathname) && url.search === location.search) return null; // anchors, same page
  return path + url.search + url.hash;
}

/** Install the click interceptor; returns the cleanup. `navigate` is the router's. */
export function installTransitions(navigate: Navigate): () => void {
  let busy = false;
  const onClick = (e: MouseEvent) => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const a = (e.target as Element | null)?.closest?.("a") as HTMLAnchorElement | null;
    if (!a) return;
    const to = targetOf(a);
    if (!to) return;
    e.preventDefault();                   // react-router's <Link> skips prevented clicks
    if (busy) return;
    busy = true;
    const path = to.split(/[?#]/)[0];
    preload(path).catch(() => undefined).then(() => { navigate(to); busy = false; });
  };
  document.addEventListener("click", onClick, true);
  return () => document.removeEventListener("click", onClick, true);
}
