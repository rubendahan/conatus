// Client-side full-text search of the site and of the documentation (W1). Each result is labelled by kind
// (guide, tutorial, methods, API, site page), and task guides and tutorials rank above the methods and API
// pages for the same match (presentation review of 0.1.1, priority 9: "restart" listed the mathematics of
// exact restarts before the guide's recipe). The index
// (search-index.json) is written by scripts/prerender.mjs from the rendered pages and, with
// --with-docs, from the built Sphinx pages (one document per section) and fetched on first open, with MiniSearch
// loaded as a separate chunk. Keyboard: "/" or Ctrl/Cmd+K opens, arrows move, Enter opens, Esc closes.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { withBase } from "../lib/base";

interface Doc { id: string; path: string; page: string; heading: string; text: string }
interface Hit { id: string; path: string; page: string; heading: string; snippet: string }

// At most MAX_HITS results are listed; the hint says how many matched in all (finding 78).
const MAX_HITS = 12;

/** The kind of a page, from its path, and how much a match on it counts. */
export function pageKind(path: string): { label: string; boost: number } {
  if (path.startsWith("/docs/guide/")) return { label: "Guide", boost: 1.8 };
  if (path.startsWith("/docs/tutorials/") || path.startsWith("/tutorials")) return { label: "Tutorial", boost: 1.4 };
  if (path.startsWith("/docs/math/")) return { label: "Methods", boost: 0.6 };
  if (path.startsWith("/docs/api/")) return { label: "API", boost: 0.7 };
  if (path.startsWith("/docs/")) return { label: "Docs", boost: 1 };
  return { label: "Site", boost: 1.1 };
}
type Engine = { search: (q: string) => { hits: Hit[]; total: number } };

let enginePromise: Promise<Engine> | null = null;
function loadEngine(): Promise<Engine> {
  if (enginePromise) return enginePromise;
  enginePromise = (async () => {
    const [{ default: MiniSearch }, docs] = await Promise.all([
      import("minisearch"),
      fetch(withBase("/search-index.json")).then((r) => r.json() as Promise<Doc[]>),
    ]);
    const ms = new MiniSearch<Doc>({
      fields: ["heading", "page", "text"],
      storeFields: ["path", "page", "heading", "text"],
      searchOptions: {
        boost: { heading: 3, page: 2 }, prefix: true, fuzzy: 0.15, combineWith: "AND",
        boostDocument: (_id: unknown, _term: string, stored?: Record<string, unknown>) => pageKind(String(stored?.path ?? "")).boost,
      },
    });
    ms.addAll(docs);
    return {
      search(q: string) {
        const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
        const all = ms.search(q);
        const hits = all.slice(0, MAX_HITS).map((r) => {
          const text: string = (r as unknown as Doc).text;
          const low = text.toLowerCase();
          const at = Math.max(0, ...terms.map((t) => low.indexOf(t)).filter((i) => i >= 0).slice(0, 1));
          const start = Math.max(0, at - 60);
          const snippet = (start > 0 ? "…" : "") + text.slice(start, start + 170) + (text.length > start + 170 ? "…" : "");
          const d = r as unknown as Doc;
          return { id: String(r.id), path: d.path, page: d.page, heading: d.heading, snippet };
        });
        return { hits, total: all.length };
      },
    };
  })();
  // a failed load (offline, a deploy in progress) is not cached: the next opening retries
  enginePromise.catch(() => { enginePromise = null; });
  return enginePromise;
}

function Highlight({ text, q }: { text: string; q: string }) {
  const terms = q.split(/\s+/).filter((t) => t.length > 1).map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!terms.length) return <>{text}</>;
  const re = new RegExp(`(${terms.join("|")})`, "gi");
  return <>{text.split(re).map((part, i) => (i % 2 ? <mark key={i}>{part}</mark> : part))}</>;
}

export default function Search() {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<Hit[]>([]);
  const [total, setTotal] = useState(0);
  const [active, setActive] = useState(0);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const inputRef = useRef<HTMLInputElement>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const openerRef = useRef<HTMLElement | null>(null);
  const navigate = useNavigate();

  const show = useCallback(() => {
    openerRef.current = document.activeElement as HTMLElement;
    setOpen(true);
  }, []);
  const hide = useCallback(() => {
    setOpen(false);
    openerRef.current?.focus?.();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
      if ((e.key === "k" && (e.metaKey || e.ctrlKey)) || (e.key === "/" && !typing)) { e.preventDefault(); show(); }
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("cn:open-search", show); // other components can open the search
    return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("cn:open-search", show); };
  }, [show]);

  useEffect(() => {
    const d = dialogRef.current;
    if (!d) return;
    if (open && !d.open) {
      d.showModal();
      inputRef.current?.focus();
      if (status === "idle" || status === "error") {
        setStatus("loading");
        loadEngine().then(() => setStatus("ready"), () => setStatus("error"));
      }
    } else if (!open && d.open) d.close();
  }, [open, status]);

  useEffect(() => {
    if (status !== "ready" || !q.trim()) { setHits([]); setTotal(0); return; }
    let live = true;
    loadEngine().then((e) => { if (live) { const r = e.search(q.trim()); setHits(r.hits); setTotal(r.total); setActive(0); } });
    return () => { live = false; };
  }, [q, status]);

  const go = (h: Hit) => {
    hide();
    const hash = h.id.includes("#") ? "#" + h.id.split("#")[1] : "";
    // documentation pages are the Sphinx site under /docs/: a full page load, not a route
    if (h.path.startsWith("/docs/")) { window.location.href = withBase(h.path) + (hash === "#top" ? "" : hash); return; }
    navigate(h.path + hash);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setActive((a) => Math.max(0, Math.min(a + 1, hits.length - 1))); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => Math.max(a - 1, 0)); }
    else if (e.key === "Enter" && hits[active]) { e.preventDefault(); go(hits[active]); }
  };

  const hint = useMemo(() => (status === "loading" ? "Loading the index…" : status === "error" ? "The search index could not be loaded." : q ? (total > hits.length ? `${total} results, the first ${hits.length} shown: add a word to narrow them` : `${total} result${total === 1 ? "" : "s"}`) : "Try: Floquet, Orszag, checkpoints, resume_or_start, h5py"), [status, q, hits.length, total]);

  return (
    <>
      <button type="button" className="search-btn" onClick={show} aria-haspopup="dialog" aria-label="Search the site (press /)">
        <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.8"/><line x1="15.5" y1="15.5" x2="21" y2="21" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/></svg>
        <span className="search-btn-label">Search</span>
        <kbd className="search-btn-kbd">/</kbd>
      </button>
      <dialog ref={dialogRef} className="search-dialog" aria-label="Search" onClose={() => setOpen(false)}
        onClick={(e) => { if (e.target === dialogRef.current) hide(); }}>
        <div className="search-panel" onKeyDown={onKeyDown}>
          <div className="search-field">
            <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.8"/><line x1="15.5" y1="15.5" x2="21" y2="21" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/></svg>
            <input ref={inputRef} type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search the site and the documentation…"
              aria-label="Search query" aria-controls="search-results" aria-activedescendant={hits[active] ? `sr-${active}` : undefined}
              autoComplete="off" spellCheck={false} />
            <button type="button" className="icon-btn" onClick={hide} aria-label="Close search"><kbd>Esc</kbd></button>
          </div>
          <p className="search-hint" role="status" aria-live="polite">{hint}</p>
          <ul id="search-results" role="listbox" className="search-results">
            {hits.map((h, i) => (
              <li key={h.id} id={`sr-${i}`} role="option" aria-selected={i === active} className={i === active ? "active" : undefined}
                onMouseEnter={() => setActive(i)} onClick={() => go(h)}>
                <span className="sr-page"><span className="sr-kind">{pageKind(h.path).label}</span>{h.page}{h.heading && h.heading !== h.page ? <> · <Highlight text={h.heading} q={q} /></> : null}</span>
                <span className="sr-snippet"><Highlight text={h.snippet} q={q} /></span>
              </li>
            ))}
          </ul>
        </div>
      </dialog>
    </>
  );
}
