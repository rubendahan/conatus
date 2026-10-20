// FluidHero: the live Kelvin-Helmholtz demo of the home page (WebGL2 solver in ./kh.ts).
//
// The opening is one continuous physical sequence. The server renders a poster of the state the solver
// starts from: the tanh shear layer marked by parallel smoke lines and one straight fluorescein filament
// (the streamlines of Stuart's solution at small rho, i.e. the base flow with a barely visible
// fundamental undulation). When the solver boots (after `load` + idle), it cross-fades in from that same
// state, and the reader watches the layer roll up. If WebGL2 is refused (software renderer, no float
// targets), the poster itself advances through the exact Stuart family
//     psi = ln(cosh y - rho cos x),   rho: 0.05 -> 0.55 over ~4 s,
// every frame of which is an exact steady solution of the Euler equations (a family of solutions, not
// a time evolution; the caption of the page says the flow shown is an illustration). Reduced motion:
// the poster is drawn at rho = 0.55 at once, and the solver (if any) shows one evolved still frame.
//
// Reader input: press and drag across the layer to inject fluorescein (and stir it gently); a vertical
// swipe still scrolls the page (touch-action: pan-y). Runs only while visible and the tab is shown.
// Controls (top-right): Pause/Play and a one-line hint; no numeric readout.
// Testing: `?sim=soft` in the URL accepts a software GL renderer (headless screenshots).

import { useEffect, useRef, useState, type MutableRefObject } from 'react';
import type { KHSim } from './kh';
import { startKH } from './host';

// ---- poster: streamlines of Stuart's cat's-eye flow, cosh y - rho cos x = L (lengths in units of delta) ----
const W = 1200, H = 600;
const SPAN_D = 17.4;                 // visible height in delta: the solver shows 0.62 Ly = 17.4 delta
const A = H / SPAN_D;                // pixels per delta
const X0 = -Math.PI;                 // left edge of the poster (a trough of the undulation)
const PERIODS = W / A / (2 * Math.PI);
const SPACING = 28 / 64;             // smoke-line spacing of the solver's dye field (Ly/64), in delta
const RHO0 = 0.05, RHO1 = 0.55;
const DYE_Y = 1.5;                   // the solver's fluorescein filament: 1.5 delta above the layer

interface Line { level: number; sign: 1 | -1; op: number; w: number }
// Levels L = cosh(y_j) of the solver's initial smoke lines y_j = +-(j + 1/2) * spacing, with the solver's
// brightness envelope (brighter near the layer). A level is fixed along the family: as rho grows, the
// lines with L < 1 + rho close into the cat's eyes, the others bulge around them.
const LINES: Line[] = [];
for (let j = 0; (j + 0.5) * SPACING < SPAN_D / 2 + 1; j++) {
  const y = (j + 0.5) * SPACING;
  const env = 0.035 + 0.765 * Math.exp(-Math.pow(y / 28 / 0.17, 2));
  for (const sign of [1, -1] as const) {
    if (sign === 1 && Math.abs(y - DYE_Y) < SPACING / 2) continue;   // the filament replaces that line
    LINES.push({ level: Math.cosh(y), sign, op: Math.min(0.9, 0.12 + 0.95 * env), w: 0.7 + 0.8 * env });
  }
}
const DYE_LEVEL = Math.cosh(DYE_Y);

// One decimal is enough at the poster's 1200 x 600 viewBox; Math.round is several times cheaper than toFixed.
const r1 = (v: number) => Math.round(v * 10) / 10;
const px = (x: number) => r1((x - X0) * A);
const py = (y: number) => r1(H / 2 - A * y);

// Points per period of the open streamlines: the server poster and the reduced-motion still use the
// finest; the animated family adapts it to the measured cost of a frame (see posterFamily).
const PTS_MAX = 64, PTS_MIN = 8;
// The prerendered poster (rho = 0.05, open lines barely undulating) draws its open lines with 16 points
// per period: the chords stay within 0.06 px of the curve at the 1200 x 600 viewBox, and the inline SVG
// is about 4x smaller than at 64 (it was 182 KB of the 247 KB home page, 44 of its 64 KB compressed, all
// on the path to the first paint). The closed eyes (innermost line only) keep the finest sampling: their
// ends have a square-root profile (1.5 px off with 12 points). Server and client render the same
// strings (hydration); later frames of the family refine them.
const PTS_POSTER = 16;
const posterPer = (L: number) => (L - RHO0 >= 1 ? PTS_POSTER : PTS_MAX);
interface Grid { n: number; x: Float64Array; cos: Float64Array; sx: string[] }
const grids = new Map<number, Grid>();
/** Abscissae of the open lines for `per` points per period: shared by every line of a frame. */
function gridOf(per: number): Grid {
  let g = grids.get(per);
  if (!g) {
    const n = Math.round(PERIODS * per), x1 = X0 + W / A;
    const x = new Float64Array(n + 1), cos = new Float64Array(n + 1), sx: string[] = [];
    for (let i = 0; i <= n; i++) { x[i] = X0 + ((x1 - X0) * i) / n; cos[i] = Math.cos(x[i]); sx.push(String(px(x[i]))); }
    g = { n, x, cos, sx };
    grids.set(per, g);
  }
  return g;
}

/** Path of the level set cosh y - rho cos x = L on the visible window (open lines, or closed eyes). */
function levelPath(L: number, sign: 1 | -1, rho: number, per: number = PTS_MAX): string {
  const x1 = X0 + W / A;
  if (L - rho >= 1) {                         // open streamline, one branch
    const g = gridOf(per), parts: string[] = new Array(g.n + 1);
    for (let i = 0; i <= g.n; i++)            // guard rounding at L - rho = 1
      parts[i] = (i ? 'L' : 'M') + g.sx[i] + ' ' + py(sign * Math.acosh(Math.max(1, L + rho * g.cos[i])));
    return parts.join('');
  }
  if (sign === -1 || L + rho <= 1) return ''; // closed eye: drawn once, from the upper line
  // closed curve around each eye centre x = 2 pi k (where cos x = 1): |x - 2 pi k| <= acos((1 - L) / rho)
  const half = Math.acos(Math.max(-1, Math.min(1, (1 - L) / rho)));
  const m = Math.max(12, Math.round((48 * per) / PTS_MAX));
  let d = '';
  for (let k = Math.floor(X0 / (2 * Math.PI)) - 1; 2 * Math.PI * k - half < x1; k++) {
    const c = 2 * Math.PI * k;
    if (c + half < X0) continue;
    let up = '', dn = '';
    for (let i = 0; i <= m; i++) {
      const x = c - half + (2 * half * i) / m;
      const y = Math.acosh(Math.max(1, L + rho * Math.cos(x)));
      up += (up ? 'L' : 'M') + px(x) + ' ' + py(y);
      dn = 'L' + px(x) + ' ' + py(-y) + dn;
    }
    d += up + dn + 'Z';
  }
  return d;
}

function Poster({ paths, dye }: { paths: MutableRefObject<(SVGPathElement | null)[]>; dye: MutableRefObject<SVGPathElement | null> }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid slice" aria-hidden="true"
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
      <g fill="none" stroke="#EDEBE6" strokeLinecap="round" strokeLinejoin="round">
        {LINES.map((l, i) => (
          <path key={i} ref={(el) => { paths.current[i] = el; }} d={levelPath(l.level, l.sign, RHO0, posterPer(l.level))}
            strokeOpacity={l.op} strokeWidth={l.w} />
        ))}
      </g>
      <path ref={dye} d={levelPath(DYE_LEVEL, 1, RHO0, posterPer(DYE_LEVEL))} fill="none" stroke="#C6F03C" strokeWidth={1.8} strokeOpacity={0.95} />
    </svg>
  );
}

// Scoped CSS for the controls (inline styles cannot express :hover / :focus-visible).
const CSS = `.cn-fluid-ui{position:absolute;top:12px;right:12px;z-index:2;display:flex;flex-direction:column;align-items:flex-end;gap:6px;
pointer-events:none;font:400 11px/1.3 var(--cn-font-code,monospace);color:#A3A29D;text-align:right}
.cn-fluid-row{display:flex;align-items:center;gap:8px}
.cn-fluid-hint{font-family:var(--cn-font-text,sans-serif);font-size:11px;color:#8A8A86;text-shadow:0 0 6px #0B0B0B}
.cn-fluid-btn{pointer-events:auto;display:inline-flex;align-items:center;gap:.45em;padding:.3em .6em;
font:400 11px/1 var(--cn-font-text,sans-serif);letter-spacing:.04em;
color:#A3A29D;background:transparent;border:1px solid rgba(237,235,230,.18);border-radius:var(--cn-radius,2px);cursor:pointer}
.cn-fluid-btn:hover{color:#EDEBE6;border-color:rgba(237,235,230,.5)}
.cn-fluid-btn:focus-visible{outline:2px solid #C6F03C;outline-offset:2px}
.cn-fluid-btn svg{width:.8em;height:.8em;fill:currentColor}`;

export default function FluidHero(props: { className?: string; label?: string; interactive?: boolean }): JSX.Element {
  const { className, label, interactive = true } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<KHSim | null>(null);
  const syncRef = useRef<() => void>(() => {});
  const pausedRef = useRef(false);
  const pathRefs = useRef<(SVGPathElement | null)[]>([]);
  const dyeRef = useRef<SVGPathElement | null>(null);
  const [paused, setPaused] = useState(false);
  const [live, setLive] = useState(false);         // canvas has a frame: fade the poster out
  const [animated, setAnimated] = useState(false); // WebGL running and motion allowed: show controls
  const [dragged, setDragged] = useState(false);   // the hint goes once the reader has injected dye

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let teardown = () => {};
    let idle = 0, timer = 0, chunkT = 0, cancelled = false, posterRaf = 0;
    const reducedNow = () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

    // Poster through the exact Stuart family (fallback when there is no solver). The cost of a frame
    // is measured and the number of points per line adapted to it (PTS_MIN..PTS_MAX), at most 30
    // updates per second: no frame becomes a long task, even on a slow phone or under Lighthouse's
    // CPU throttling (the family used to cost ~1 s of blocking time on the home page). The last frame
    // (rho = 0.55, where the family stops) is redrawn at full resolution a few lines per task.
    const drawPoster = (rho: number, per: number, from = 0, to = LINES.length) => {
      for (let i = from; i < to; i++) pathRefs.current[i]?.setAttribute('d', levelPath(LINES[i].level, LINES[i].sign, rho, per));
      if (to === LINES.length) dyeRef.current?.setAttribute('d', levelPath(DYE_LEVEL, 1, rho, per));
    };
    const finest = (rho: number) => {
      let i = 0;
      const chunk = () => {
        if (cancelled) return;
        const t0 = performance.now();
        while (i < LINES.length && performance.now() - t0 < 6) { drawPoster(rho, PTS_MAX, i, i + 1); i++; }
        if (i < LINES.length) chunkT = window.setTimeout(chunk, 0); else drawPoster(rho, PTS_MAX, LINES.length, LINES.length);
      };
      chunk();
    };
    const posterFamily = () => {
      if (cancelled) return;
      if (reducedNow()) { finest(RHO1); return; }
      const t0 = performance.now(), dur = 4200;
      let per = 16, last = -1e9;
      const frame = (now: number) => {
        if (cancelled) return;
        if (now - last < 31) { posterRaf = requestAnimationFrame(frame); return; }   // <= 30 updates/s
        last = now;
        const s = Math.max(0, Math.min(1, (now - t0) / dur));   // rAF time may precede t0
        const e = s < 0.5 ? 4 * s * s * s : 1 - Math.pow(-2 * s + 2, 3) / 2;   // ease in-out
        if (s >= 1) { finest(RHO1); return; }
        const c0 = performance.now();
        drawPoster(RHO0 + (RHO1 - RHO0) * e, per);
        const cost = performance.now() - c0;
        if (cost > 6 && per > PTS_MIN) per /= 2; else if (cost < 1.5 && per < PTS_MAX) per *= 2;
        posterRaf = requestAnimationFrame(frame);
      };
      posterRaf = requestAnimationFrame(frame);
    };

    // The solver is created only after `load` and an idle slot (fallback 1.2 s): it never
    // competes with the page's first paint / hydration, and the poster shows until then. It runs in
    // a worker on an OffscreenCanvas where the browser allows it (host.ts), else on this thread,
    // created in short tasks; the reader's pointer events are forwarded to it.
    const boot = async () => {
      if (cancelled) return;
      let sim: KHSim;
      const soft = /[?&]sim=soft\b/.test(window.location.search + window.location.hash);
      const weak = matchMedia('(max-width: 640px)').matches || (navigator.hardwareConcurrency || 8) <= 4;
      try {
        sim = await startKH(canvas, {
          interactive,
          allowSoftware: soft,
          weak,
          onReady: () => setLive(true),
          onFail: () => { setLive(false); setAnimated(false); posterFamily(); },
        }, () => cancelled);
      } catch {
        if (!cancelled) posterFamily();   // no WebGL2 / float targets / software renderer: the exact family
        return;
      }
      if (cancelled) { sim.dispose(); return; }
      simRef.current = sim;

      const motion = matchMedia('(prefers-reduced-motion: reduce)');
      let visible = true, reduced = motion.matches, stillDone = false, stilling = false;

      // Reduced motion: 480 steps off-screen in short slices, then one still frame.
      const still = () => {
        if (stilling) return;
        stilling = true;
        sim.still(480, () => { stilling = false; stillDone = true; });
      };
      const sync = () => {
        setAnimated(!reduced);
        if (reduced) { sim.stop(); if (!stillDone) still(); return; }
        const run = visible && !document.hidden && !pausedRef.current;
        if (run) sim.start(); else sim.stop();
      };
      syncRef.current = sync;

      // attention budget: the hero runs only while at least 35% of it is on screen
      const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting && e.intersectionRatio >= 0.35; sync(); },
        { threshold: [0, 0.35, 0.7, 1] });
      io.observe(canvas);
      const onVis = () => sync();
      document.addEventListener('visibilitychange', onVis);
      const onMotion = () => { reduced = motion.matches; stillDone = false; sync(); };
      motion.addEventListener('change', onMotion);
      let rt = 0;
      const ro = new ResizeObserver(() => {
        clearTimeout(rt);
        rt = window.setTimeout(() => { sim.resize(); if (reduced) { stillDone = false; sync(); } else sim.render(); }, 150);
      });
      ro.observe(canvas);
      // reader input, forwarded with coordinates relative to the canvas
      const fwd = (kind: 'down' | 'move' | 'up') => (e: PointerEvent) => {
        const r = canvas.getBoundingClientRect();
        if (kind === 'down' && e.button === 0) { try { canvas.setPointerCapture(e.pointerId); } catch { /* optional */ } }
        sim.pointer({ kind, x: e.clientX - r.left, y: e.clientY - r.top, w: r.width, h: r.height, button: e.button });
      };
      const onDown = fwd('down'), onMove = fwd('move'), onUp = fwd('up');
      if (interactive) {
        canvas.addEventListener('pointerdown', onDown);
        canvas.addEventListener('pointermove', onMove);
        canvas.addEventListener('pointerup', onUp);
        canvas.addEventListener('pointercancel', onUp);
      }
      sync();

      teardown = () => {
        io.disconnect(); ro.disconnect(); clearTimeout(rt);
        canvas.removeEventListener('pointerdown', onDown);
        canvas.removeEventListener('pointermove', onMove);
        canvas.removeEventListener('pointerup', onUp);
        canvas.removeEventListener('pointercancel', onUp);
        document.removeEventListener('visibilitychange', onVis);
        motion.removeEventListener('change', onMotion);
        sim.dispose();
        simRef.current = null;
        syncRef.current = () => {};
      };
    };
    const whenIdle = () => {
      if (typeof window.requestIdleCallback === 'function') idle = window.requestIdleCallback(() => { void boot(); }, { timeout: 2500 });
      else timer = setTimeout(() => { void boot(); }, 1200) as unknown as number;   // Safari < 18: no idle callback
    };
    if (document.readyState === 'complete') whenIdle();
    else window.addEventListener('load', whenIdle, { once: true });

    return () => {
      cancelled = true;
      cancelAnimationFrame(posterRaf);
      window.removeEventListener('load', whenIdle);
      if (idle) window.cancelIdleCallback(idle);
      clearTimeout(timer); clearTimeout(chunkT);
      teardown();
    };
  }, [interactive]);

  const toggle = () => {
    pausedRef.current = !pausedRef.current;
    setPaused(pausedRef.current);
    syncRef.current();
  };

  // The root stays out of hit-testing: only the canvas (drag target) and the controls take pointers,
  // so the hero's text and buttons above it keep working.
  return (
    <div className={className} role="group" aria-label={label}
      style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden', pointerEvents: 'none' }}>
      <style>{CSS}</style>
      <div aria-hidden="true" style={{ position: 'absolute', inset: 0, zIndex: -2, background: '#0B0B0B', opacity: live ? 0 : 1, transition: 'opacity .8s' }}>
        <Poster paths={pathRefs} dye={dyeRef} />
      </div>
      <canvas ref={canvasRef} aria-hidden="true"
        style={{ position: 'absolute', inset: 0, zIndex: -2, width: '100%', height: '100%', display: 'block', opacity: live ? 1 : 0, transition: 'opacity .8s',
          pointerEvents: interactive && live ? 'auto' : 'none', touchAction: 'pan-y', cursor: interactive && live ? 'crosshair' : undefined }}
        onPointerDown={() => setDragged(true)} />
      {animated && live && (
        <div className="cn-fluid-ui">
          <div className="cn-fluid-row">
            <button type="button" className="cn-fluid-btn" onClick={toggle}>
              <svg viewBox="0 0 10 10" aria-hidden="true">
                {paused ? <path d="M2 1l7 4-7 4z" /> : <path d="M2 1h2v8H2zM6 1h2v8H6z" />}
              </svg>
              {paused ? 'Play' : 'Pause'}
            </button>
          </div>
          {interactive && !dragged && <span className="cn-fluid-hint">Drag across the layer to inject dye</span>}
        </div>
      )}
    </div>
  );
}
