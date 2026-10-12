// Gallery lightbox: one plate at full resolution on lab black, with its name and caption.
// A modal <dialog> (focus stays inside, Esc closes), body scroll locked while open. The image opens
// with a shared-element morph from its thumbnail (FLIP on transform, 280 ms) and closes back into it;
// nothing morphs under prefers-reduced-motion. Zoom: click (mouse) or double-tap (touch) toggles
// fit ↔ max(2×, 1:1) at the pointer; pinch zooms on touch; drag pans a zoomed image. Unzoomed,
// a horizontal swipe or the arrow keys go to the previous / next plate.
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { plainCaption, plainTitle, src, type GalleryItem } from "../data/gallery";
import "../styles/lightbox.css";

const EASE = "cubic-bezier(0.2, 0, 0, 1)";
const MS = 280;
// useLayoutEffect warns on the server (prerender); the morph only exists in the browser.
const useIso = typeof window === "undefined" ? useEffect : useLayoutEffect;
const reduced = () => typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/** Transform that maps rect `to` onto rect `from` (both viewport rects). */
function flip(from: DOMRect, to: DOMRect): string {
  const s = from.width / to.width;
  const dx = from.left + from.width / 2 - (to.left + to.width / 2);
  const dy = from.top + from.height / 2 - (to.top + to.height / 2);
  return `translate(${dx}px, ${dy}px) scale(${s})`;
}

export interface LightboxProps {
  items: GalleryItem[];                    // navigation order (the filtered plates)
  index: number | null;                    // open plate, null when closed
  morph: boolean;                          // morph from the thumbnail on this opening
  origin: (slug: string) => HTMLElement | null;   // the thumbnail <img> of a plate
  onChange: (index: number | null) => void;
}

export default function Lightbox({ items, index, morph, origin, onChange }: LightboxProps) {
  const dlg = useRef<HTMLDialogElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const img = useRef<HTMLImageElement>(null);
  const [full, setFull] = useState<string | null>(null);   // full-resolution src once loaded
  const [zoomed, setZoomed] = useState(false);
  const closing = useRef(false);
  const g = index !== null ? items[index] : null;

  // ---- zoom / pan state, applied directly to the image transform ----
  const z = useRef({ s: 1, tx: 0, ty: 0 });
  const apply = useCallback((animate: boolean) => {
    const el = img.current;
    if (!el) return;
    const { s, tx, ty } = z.current;
    el.style.transition = animate && !reduced() ? `transform 220ms ${EASE}` : "none";
    el.style.transform = s === 1 && tx === 0 && ty === 0 ? "" : `translate(${tx}px, ${ty}px) scale(${s})`;
    setZoomed(s > 1.01);
  }, []);
  const clamp = () => {
    const el = img.current, st = stage.current;
    if (!el || !st) return;
    const { s } = z.current;
    const mx = Math.max(0, (el.offsetWidth * s - st.clientWidth) / 2);
    const my = Math.max(0, (el.offsetHeight * s - st.clientHeight) / 2);
    z.current.tx = Math.max(-mx, Math.min(mx, z.current.tx));
    z.current.ty = Math.max(-my, Math.min(my, z.current.ty));
  };
  const reset = () => { z.current = { s: 1, tx: 0, ty: 0 }; apply(false); };
  /** Offset of a viewport point from the image centre (untransformed layout box). */
  const offset = (x: number, y: number) => {
    const st = stage.current!.getBoundingClientRect();
    return { ox: x - (st.left + st.width / 2), oy: y - (st.top + st.height / 2) };
  };
  const zoomTo = (s: number, x: number, y: number, animate = true) => {
    const { ox, oy } = offset(x, y);
    const k = s / z.current.s;
    z.current = { s, tx: ox - (ox - z.current.tx) * k, ty: oy - (oy - z.current.ty) * k };
    clamp(); apply(animate);
  };
  const toggleZoom = (x: number, y: number) => {
    const el = img.current;
    if (!el) return;
    if (z.current.s > 1.01) { z.current = { s: 1, tx: 0, ty: 0 }; apply(true); return; }
    const oneToOne = (g?.width ?? el.naturalWidth) / Math.max(1, el.offsetWidth);
    zoomTo(Math.min(4, Math.max(2, oneToOne)), x, y);
  };

  // ---- open / close ----
  const go = useCallback((d: number) => {
    if (index === null || items.length < 2) return;
    onChange((index + d + items.length) % items.length);
  }, [index, items.length, onChange]);

  const requestClose = useCallback(() => {
    const d = dlg.current, el = img.current;
    if (!d || closing.current || index === null) return;
    closing.current = true;
    const done = () => {
      closing.current = false;
      d.classList.remove("is-closing");
      d.close();
      onChange(null);
    };
    const th = origin(items[index].slug);
    const r = th?.getBoundingClientRect();
    const visible = r && r.bottom > 0 && r.top < window.innerHeight && r.width > 0;
    if (reduced() || !el) { done(); return; }
    reset();
    d.classList.add("is-closing");
    if (visible && r) {
      const to = el.getBoundingClientRect();
      el.style.transition = `transform ${MS}ms ${EASE}`;
      el.style.transform = flip(r, to);
    }
    window.setTimeout(done, MS);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index, items, onChange, origin]);

  // show the dialog and run the opening morph
  useIso(() => {
    const d = dlg.current;
    if (!d || index === null || d.open) return;
    d.showModal();
    document.documentElement.classList.add("lb-lock");
    const el = img.current, th = origin(items[index].slug);
    if (!morph || reduced() || !el || !th) return;
    const from = th.getBoundingClientRect(), to = el.getBoundingClientRect();
    if (!from.width || !to.width) return;
    d.classList.add("is-opening");
    el.style.transition = "none";
    el.style.transform = flip(from, to);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      el.style.transition = `transform ${MS}ms ${EASE}`;
      el.style.transform = "";
      d.classList.remove("is-opening");
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index]);

  useEffect(() => {
    if (index === null) document.documentElement.classList.remove("lb-lock");
    return () => document.documentElement.classList.remove("lb-lock");
  }, [index]);

  // new plate: fit, show the small image at once, swap in the full resolution when loaded
  useEffect(() => {
    if (!g) return;
    z.current = { s: 1, tx: 0, ty: 0 };
    setZoomed(false);
    setFull(null);
    let alive = true;
    const im = new Image();
    im.onload = () => { if (alive) setFull(src(g.image)); };
    im.src = src(g.image);
    return () => { alive = false; };
  }, [g]);

  // preload the neighbours
  useEffect(() => {
    if (index === null || items.length < 2) return;
    for (const d of [1, -1]) { const n = items[(index + d + items.length) % items.length]; new Image().src = src(n.image_small); }
  }, [index, items]);

  const onKey = (e: KeyboardEvent) => {
    if (e.key === "ArrowRight") { e.preventDefault(); go(1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); go(-1); }
  };

  // ---- pointers: pan, pinch, swipe, tap / double-tap ----
  const ptr = useRef(new Map<number, { x: number; y: number }>());
  const gest = useRef({ x0: 0, y0: 0, t0: 0, tx0: 0, ty0: 0, d0: 0, s0: 1, mx: 0, my: 0, moved: false, pinch: false, lastTap: 0, lx: 0, ly: 0 });
  const onDown = (e: PointerEvent) => {
    if (e.button !== 0 && e.pointerType === "mouse") return;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    ptr.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const G = gest.current;
    if (ptr.current.size === 1) {
      Object.assign(G, { x0: e.clientX, y0: e.clientY, t0: performance.now(), tx0: z.current.tx, ty0: z.current.ty, moved: false, pinch: false });
    } else if (ptr.current.size === 2) {
      const [a, b] = [...ptr.current.values()];
      Object.assign(G, { d0: Math.hypot(a.x - b.x, a.y - b.y), s0: z.current.s, mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2, tx0: z.current.tx, ty0: z.current.ty, pinch: true, moved: true });
    }
  };
  const onMove = (e: PointerEvent) => {
    if (!ptr.current.has(e.pointerId)) return;
    ptr.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const G = gest.current;
    if (ptr.current.size >= 2 && G.pinch) {
      const [a, b] = [...ptr.current.values()];
      const s = Math.min(4, Math.max(1, (G.s0 * Math.hypot(a.x - b.x, a.y - b.y)) / Math.max(1, G.d0)));
      const { ox, oy } = offset(G.mx, G.my);
      const k = s / G.s0;
      z.current = { s, tx: ox - (ox - G.tx0) * k + ((a.x + b.x) / 2 - G.mx), ty: oy - (oy - G.ty0) * k + ((a.y + b.y) / 2 - G.my) };
      clamp(); apply(false);
      return;
    }
    const dx = e.clientX - G.x0, dy = e.clientY - G.y0;
    if (!G.moved && Math.hypot(dx, dy) > 6) G.moved = true;
    if (!G.moved || G.pinch) return;
    if (z.current.s > 1.01) {                         // pan
      z.current.tx = G.tx0 + dx; z.current.ty = G.ty0 + dy;
      clamp(); apply(false);
    } else if (items.length > 1 && img.current) {     // swipe feedback
      img.current.style.transition = "none";
      img.current.style.transform = `translateX(${dx}px)`;
    }
  };
  const onUp = (e: PointerEvent) => {
    if (!ptr.current.has(e.pointerId)) return;
    ptr.current.delete(e.pointerId);
    const G = gest.current;
    if (G.pinch) { if (ptr.current.size === 0) { G.pinch = false; if (z.current.s < 1.05) reset(); } return; }
    const dx = e.clientX - G.x0, dy = e.clientY - G.y0;
    if (G.moved) {
      if (z.current.s <= 1.01) {
        if (Math.abs(dx) > 60 && Math.abs(dx) > 1.5 * Math.abs(dy) && items.length > 1) go(dx < 0 ? 1 : -1);
        else apply(true);                              // spring back
      }
      return;
    }
    // a tap: mouse click toggles zoom; on touch, a double tap does
    if (e.pointerType === "mouse") { toggleZoom(e.clientX, e.clientY); return; }
    const now = performance.now();
    if (now - G.lastTap < 320 && Math.hypot(e.clientX - G.lx, e.clientY - G.ly) < 30) { G.lastTap = 0; toggleZoom(e.clientX, e.clientY); }
    else { G.lastTap = now; G.lx = e.clientX; G.ly = e.clientY; }
  };
  const onCancel = (e: PointerEvent) => { ptr.current.delete(e.pointerId); gest.current.pinch = false; apply(true); };

  const coarse = typeof window !== "undefined" && !!window.matchMedia?.("(pointer: coarse)").matches;
  return (
    <dialog ref={dlg} className="lb" aria-labelledby="lb-title" onKeyDown={onKey}
      onCancel={(e) => { e.preventDefault(); requestClose(); }}>
      {g && (
        <div className="lb-frame">
          <header className="lb-bar">
            <p className="lb-count">{index! + 1} / {items.length}</p>
          </header>
          <button type="button" className="lb-btn lb-close" onClick={requestClose} aria-label="Close">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" /></svg>
          </button>
          <div ref={stage} className={"lb-stage" + (zoomed ? " is-zoomed" : "")}
            onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onCancel}>
            <img ref={img} key={g.slug} className="lb-img" src={full ?? src(g.image_small)} width={g.width} height={g.height}
              alt={g.alt} draggable={false} style={{ ["--ar" as string]: String(g.width / g.height) }} />
          </div>
          {items.length > 1 && (
            <div className="lb-nav">
              <button type="button" className="lb-btn lb-prev" onClick={() => go(-1)} aria-label="Previous plate">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 5l-7 7 7 7" /></svg>
              </button>
              <button type="button" className="lb-btn lb-next" onClick={() => go(1)} aria-label="Next plate">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 5l7 7-7 7" /></svg>
              </button>
            </div>
          )}
          <div className="lb-info">
            <h2 id="lb-title" className="lb-title">{plainTitle(g)}</h2>
            <p className="lb-caption">{plainCaption(g)}</p>
            <p className="lb-hint" aria-hidden="true">{coarse
              ? (zoomed ? "Drag to pan · pinch or double-tap to fit" : "Double-tap or pinch to zoom · swipe for the next plate")
              : (zoomed ? "Drag to pan · click to fit" : "Click to zoom · ← → for the next plate · Esc to close")}</p>
          </div>
        </div>
      )}
    </dialog>
  );
}
