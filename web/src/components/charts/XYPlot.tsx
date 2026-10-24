// Scatter / line chart on two numeric axes (linear or log10), for the published cases: the paper's
// values (series s2, rings) drawn over the values Conatus computes (series s1, discs or a line).
// Each point is a hover and keyboard-focus target; the figure frame ships the data as a table.
import { type ReactNode } from "react";
import { linearScale, logScale, Marker, Plot, TipRow, tipTarget, useTip, useWidth, XAxis, type Scale, type SeriesId } from "./core";

export interface XYSeries {
  id: string;
  label: string;
  cls: SeriesId;
  points: [number, number][];
  line?: boolean;          // draw as a polyline (no markers)
  r?: number;              // marker radius
  quiet?: boolean;         // markers without hover/focus targets (dense series; the data table lists them)
}

export interface XYPlotProps {
  series: XYSeries[];
  xLabel: string;
  yLabel: string;
  xLog?: boolean;
  yLog?: boolean;
  xDomain?: [number, number];
  yDomain?: [number, number];
  ariaLabel: string;
  fmt?: (x: number, y: number) => ReactNode;
  height?: number;
}

const M = { top: 12, right: 16, bottom: 44, left: 64 };

function YAxis({ scale, x, right, label }: { scale: Scale; x: number; right: number; label: string }) {
  return (
    <g className="axis" aria-hidden="true">
      {scale.ticks.map((t) => (
        <g key={t}>
          <line className="grid" x1={x} x2={right} y1={scale(t)} y2={scale(t)} />
          <text className="tick" x={x - 6} y={scale(t) + 4} textAnchor="end">{scale.fmt(t)}</text>
        </g>
      ))}
      <line className="baseline" x1={x} x2={x} y1={scale(scale.domain[0])} y2={scale(scale.domain[1])} />
      <text className="axis-label" x={x - 50} y={scale(scale.domain[1]) - 2} transform={`rotate(-90 ${x - 50} ${scale(scale.domain[1]) - 2})`} textAnchor="end">{label}</text>
    </g>
  );
}

function extent(vals: number[], log: boolean): [number, number] {
  const v = log ? vals.filter((x) => x > 0) : vals;
  let lo = Math.min(...v), hi = Math.max(...v);
  if (!log) { const pad = (hi - lo || Math.abs(hi) || 1) * 0.05; lo -= pad; hi += pad; }
  return [lo, hi];
}

export function XYPlot(props: XYPlotProps) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const w = Math.max(width, 280);
  return (
    <div className="chart-box" ref={ref}>
      <Plot width={w}><XYSvg {...props} w={w} /></Plot>
    </div>
  );
}

function XYSvg({ series, xLabel, yLabel, xLog, yLog, xDomain, yDomain, ariaLabel, fmt, height = 340, w }: XYPlotProps & { w: number }) {
  const tip = useTip();
  const all = series.flatMap((s) => s.points);
  const [x0, x1] = xDomain ?? extent(all.map((p) => p[0]), !!xLog);
  const [y0, y1] = yDomain ?? extent(all.map((p) => p[1]), !!yLog);
  const xs = xLog ? logScale(x0, x1, M.left, w - M.right) : linearScale(x0, x1, M.left, w - M.right, w < 480 ? 4 : 6);
  const ys = yLog ? logScale(y0, y1, height - M.bottom, M.top) : linearScale(y0, y1, height - M.bottom, M.top, 5);
  for (const sc of [xs, ys]) { const f0 = sc.fmt; sc.fmt = (v) => f0(Math.abs(v) < 1e-12 ? 0 : v); }   // no "-0.0" tick
  const inside = (p: [number, number]) => (!xLog || p[0] > 0) && (!yLog || p[1] > 0)
    && p[0] >= xs.domain[0] && p[0] <= xs.domain[1] && p[1] >= ys.domain[0] && p[1] <= ys.domain[1];
  const show = fmt ?? ((x: number, y: number) => `${x.toPrecision(6)}, ${y.toPrecision(6)}`);
  return (
    <svg className="chart-svg" width={w} height={height} role="img" aria-label={ariaLabel}>
      <XAxis scale={xs} y={height - M.bottom} top={M.top} label={xLabel} />
      <YAxis scale={ys} x={M.left} right={w - M.right} label={yLabel} />
      {/* published values (rings, s2) underneath, so that the computed marks inside them stay visible */}
      {[...series.filter((s) => s.cls === "s2"), ...series.filter((s) => s.cls !== "s2")].map((s) => s.line ? (
        <polyline key={s.id} className={`ln ln-${s.cls}`} points={s.points.filter(inside).map((p) => `${xs(p[0])},${ys(p[1])}`).join(" ")} />
      ) : (
        <g key={s.id}>
          {s.points.filter(inside).map((p, k) => (
            <g key={k} {...(s.quiet ? {} : tipTarget(tip, xs(p[0]), ys(p[1]), <TipRow cls={s.cls} value={show(p[0], p[1])} label={s.label} />, `${s.label}: ${p[0]}, ${p[1]}`))}>
              <Marker cls={s.cls} x={xs(p[0])} y={ys(p[1])} r={s.r ?? 4} />
            </g>
          ))}
        </g>
      ))}
    </svg>
  );
}
