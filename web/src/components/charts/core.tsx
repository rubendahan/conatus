// Chart primitives shared by every figure of the site: container width, scales, series styles,
// tooltip layer and the figure frame (title, legend, source, hardware, data table).
//
// Colours follow the brand charter (section III): series 1 = quantity followed (ink on paper, fluorescein on
// black), 2 = published reference / comparison code (dashed, ring markers), 3 = other runs (grey,
// square markers), 4 = threshold / failure (rubric). Each series also has its own marker shape or dash,
// so identity never rests on colour alone; every figure ships its data as a table (<details>).
import { createContext, useContext, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";

export type SeriesId = "s1" | "s2" | "s3" | "s4" | "s5";
export interface SeriesDef { id: string; label: string; cls: SeriesId }

// ---- container width (SSR renders at the default width; the client re-measures after mount) ----
const useIso = typeof window === "undefined" ? useEffect : useLayoutEffect;
export function useWidth<T extends HTMLElement>(fallback = 640): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(fallback);
  useIso(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((es) => { const cw = Math.round(es[0].contentRect.width); if (cw > 0) setW(cw); });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

// ---- scales and ticks -----------------------------------------------------------------------
export interface Scale { (x: number): number; ticks: number[]; fmt: (x: number) => string; domain: [number, number] }

const round2 = (v: number) => Math.round(v * 100) / 100;

export function linearScale(d0: number, d1: number, r0: number, r1: number, nTicks = 5): Scale {
  const span = d1 - d0 || 1;
  const raw = span / nTicks;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const ticks: number[] = [];
  for (let t = Math.ceil(d0 / step) * step; t <= d1 + step * 1e-9; t += step) ticks.push(+t.toPrecision(12));
  const dec = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  // Coordinates rounded to 1/100 px: the server (Node) and the browser may differ in the last bit of
  // floating-point results, which would break hydration of the prerendered SVG.
  const s = ((x: number) => round2(r0 + ((x - d0) / span) * (r1 - r0))) as Scale;
  s.ticks = ticks; s.domain = [d0, d1];
  s.fmt = (x) => (Math.abs(x) >= 1000 ? x.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",") : x.toFixed(dec));
  return s;
}

/** Log10 scale; domain snapped outward to 1-2-5 values; ticks at decades (plus 2 and 5 when narrow). */
export function logScale(lo: number, hi: number, r0: number, r1: number): Scale {
  const snapDown = (x: number) => { const e = 10 ** Math.floor(Math.log10(x)); return [5, 2, 1].map((m) => m * e).find((v) => v <= x * (1 + 1e-9)) ?? e; };
  const snapUp = (x: number) => { const e = 10 ** Math.floor(Math.log10(x)); return [1, 2, 5, 10].map((m) => m * e).find((v) => v >= x * (1 - 1e-9)) ?? 10 * e; };
  const d0 = snapDown(lo), d1 = snapUp(hi);
  const l0 = Math.log10(d0), l1 = Math.log10(d1);
  const s = ((x: number) => round2(r0 + ((Math.log10(x) - l0) / (l1 - l0 || 1)) * (r1 - r0))) as Scale;
  const ticks: number[] = [];
  const narrow = l1 - l0 <= 2.2;
  for (let e = Math.floor(l0); e <= Math.ceil(l1); e++) {
    for (const m of narrow ? [1, 2, 5] : [1]) { const v = m * 10 ** e; if (v >= d0 * (1 - 1e-9) && v <= d1 * (1 + 1e-9)) ticks.push(v); }
  }
  s.ticks = ticks; s.domain = [d0, d1];
  s.fmt = (x) => (x >= 1000 ? x.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",") : x >= 1 ? String(+x.toPrecision(3)) : String(+x.toPrecision(2)));
  return s;
}

// ---- markers ---------------------------------------------------------------------------------
/** Marker per series: s1 disc, s2 ring, s3 square, s4 diamond, s5 triangle. Surface ring around each. */
export function Marker({ cls, x, y, r = 5 }: { cls: SeriesId; x: number; y: number; r?: number }) {
  const c = `mk mk-${cls}`;
  if (cls === "s3") return <rect className={c} x={x - r * 0.9} y={y - r * 0.9} width={r * 1.8} height={r * 1.8} rx={1} />;
  if (cls === "s4") return <path className={c} d={`M${x} ${y - r * 1.2}L${x + r * 1.2} ${y}L${x} ${y + r * 1.2}L${x - r * 1.2} ${y}Z`} />;
  if (cls === "s5") return <path className={c} d={`M${x} ${y - r * 1.15}L${x + r * 1.1} ${y + r * 0.8}L${x - r * 1.1} ${y + r * 0.8}Z`} />;
  return <circle className={c} cx={x} cy={y} r={cls === "s2" ? r - 0.5 : r} />;
}

/** Legend key mirroring the mark: marker for dot charts, rect for bars, stroke for lines. */
export function Legend({ series, kind = "marker" }: { series: SeriesDef[]; kind?: "marker" | "bar" | "line" }) {
  return (
    <ul className="chart-legend" aria-label="Legend">
      {series.map((s) => (
        <li key={s.id}>
          <svg width="22" height="12" aria-hidden="true" className="chart-legend-key">
            {kind === "marker" && <Marker cls={s.cls} x={11} y={6} r={4.5} />}
            {kind === "bar" && <rect className={`bar bar-${s.cls}`} x={1} y={1} width={20} height={10} rx={2} />}
            {kind === "line" && <line className={`ln ln-${s.cls}`} x1={1} x2={21} y1={6} y2={6} />}
          </svg>
          {s.label}
        </li>
      ))}
    </ul>
  );
}

// ---- tooltip layer ---------------------------------------------------------------------------
export interface TipState { x: number; y: number; content: ReactNode }
type TipApi = { show: (t: TipState) => void; hide: () => void };
const TipCtx = createContext<TipApi>({ show: () => {}, hide: () => {} });
export const useTip = () => useContext(TipCtx);

/** Props that make an SVG group a hover + keyboard-focus target showing `content` at (x, y). */
export function tipTarget(api: TipApi, x: number, y: number, content: ReactNode, label: string) {
  return {
    tabIndex: 0, role: "img" as const, "aria-label": label,
    onPointerEnter: () => api.show({ x, y, content }),
    onPointerLeave: api.hide,
    onFocus: () => api.show({ x, y, content }),
    onBlur: api.hide,
  };
}

/** A tooltip row: value first (strong), series second, keyed with a short stroke. */
export function TipRow({ cls, value, label }: { cls?: SeriesId; value: ReactNode; label: ReactNode }) {
  return (
    <div className="tip-row">
      {cls && <svg width="14" height="8" aria-hidden="true"><line className={`ln ln-${cls}`} x1={1} x2={13} y1={4} y2={4} /></svg>}
      <strong className="num">{value}</strong> <span>{label}</span>
    </div>
  );
}

/** Plot area: positions the tooltip above the hovered mark and flips it at the edges. */
export function Plot({ width, children, className }: { width: number; children: ReactNode; className?: string }) {
  const [tip, setTip] = useState<TipState | null>(null);
  const api = useRef<TipApi>({ show: setTip, hide: () => setTip(null) }).current;
  const side = tip && tip.x > width * 0.62 ? "left" : "right";
  return (
    <TipCtx.Provider value={api}>
      <div className={"chart-plot" + (className ? " " + className : "")}>
        {children}
        {tip && (
          <div className={`chart-tip tip-${side}`} role="status" style={{ left: tip.x, top: tip.y }}>
            {tip.content}
          </div>
        )}
      </div>
    </TipCtx.Provider>
  );
}

// ---- axes ----------------------------------------------------------------------------------
export function XAxis({ scale, y, top, label, labelX }: { scale: Scale; y: number; top: number; label?: string; labelX?: number }) {
  return (
    <g className="axis" aria-hidden="true">
      {scale.ticks.map((t) => (
        <g key={t}>
          <line className="grid" x1={scale(t)} x2={scale(t)} y1={top} y2={y} />
          <text className="tick" x={scale(t)} y={y + 16} textAnchor="middle">{scale.fmt(t)}</text>
        </g>
      ))}
      <line className="baseline" x1={scale(scale.domain[0])} x2={scale(scale.domain[1])} y1={y} y2={y} />
      {label && <text className="axis-label" x={labelX ?? scale(scale.domain[1])} y={y + 34} textAnchor="end">{label}</text>}
    </g>
  );
}

// ---- figure frame --------------------------------------------------------------------------
export function ChartFigure({ id, title, subtitle, legend, controls, children, source, hardware, data, note }: {
  id?: string; title: ReactNode; subtitle?: ReactNode; legend?: ReactNode; controls?: ReactNode;
  children: ReactNode; source: ReactNode; hardware?: ReactNode; data: ReactNode; note?: ReactNode;
}) {
  return (
    <figure className="chart" id={id} aria-labelledby={id ? id + "-t" : undefined}>
      <div className="chart-head">
        <h3 className="chart-title" id={id ? id + "-t" : undefined}>{title}</h3>
        {subtitle && <p className="chart-sub">{subtitle}</p>}
        {(legend || controls) && <div className="chart-bar">{legend}{controls}</div>}
      </div>
      {children}
      {note && <div className="chart-note">{note}</div>}
      <figcaption className="chart-source">
        {hardware && <span className="chart-hw">{hardware}</span>}
        <span>Source: {source}</span>
      </figcaption>
      <details className="chart-data">
        <summary>Data</summary>
        {data}
      </details>
    </figure>
  );
}

/** Segmented control (buttons with aria-pressed), used to switch a figure between views. */
export function Segmented<T extends string>({ label, options, value, onChange }: {
  label: string; options: { id: T; label: string }[]; value: T; onChange: (v: T) => void;
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.id} type="button" aria-pressed={o.id === value} onClick={() => onChange(o.id)}>{o.label}</button>
      ))}
    </div>
  );
}
