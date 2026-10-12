// Dot plot (Cleveland): one row per case, one marker per series on a shared x axis (log or linear),
// a hairline joining the extremes of the row, and optional right-hand annotation columns (ratios,
// speedups). Each row is one hover / keyboard-focus target whose tooltip lists every series.
import type { ReactNode } from "react";
import { Marker, Plot, TipRow, XAxis, linearScale, logScale, tipTarget, useTip, useWidth, type SeriesDef } from "./core";

export interface DotValue { series: string; value: number | null; text: string }
export interface DotRow { id: string; label: string; sublabel?: string; values: DotValue[]; notes?: (string | null)[]; missing?: string }

export function DotPlot(props: {
  rows: DotRow[]; series: SeriesDef[]; log?: boolean; axisLabel: string; noteHeaders?: string[]; zero?: boolean; ariaLabel: string;
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  return (
    <div ref={ref} className="chart-box">
      <Plot width={w}><DotSvg {...props} w={w} /></Plot>
    </div>
  );
}

function DotSvg({ rows, series, log, axisLabel, noteHeaders = [], zero, ariaLabel, w }: {
  rows: DotRow[]; series: SeriesDef[]; log?: boolean; axisLabel: string; noteHeaders?: string[]; zero?: boolean; ariaLabel: string; w: number;
}) {
  const tip = useTip();
  const narrow = w < 520;
  const labelW = Math.round(Math.min(150, Math.max(78, w * 0.26)));
  const noteW = noteHeaders.length ? (narrow ? 50 : 66) : 0;
  const x0 = labelW + 10, x1 = w - noteW * noteHeaders.length - 14;
  const top = noteHeaders.length ? 26 : 10, rowH = 42;
  const yAxis = top + rows.length * rowH + 4;
  const h = yAxis + 42;
  const vals = rows.flatMap((r) => r.values.map((v) => v.value)).filter((v): v is number => v != null && v > 0);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const sx = log ? logScale(lo, hi, x0, x1) : linearScale(zero ? 0 : lo * 0.9, hi * 1.05, x0, x1);
  const cls = (id: string) => series.find((s) => s.id === id)?.cls ?? "s3";
  const slabel = (id: string) => series.find((s) => s.id === id)?.label ?? id;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="group" aria-label={ariaLabel} className="chart-svg">
      <XAxis scale={sx} y={yAxis} top={top - 4} label={axisLabel} labelX={x1} />
      {noteHeaders.map((hd, k) => (
        <text key={hd} className="note-head" x={x1 + 14 + noteW * (k + 1) - 4} y={14} textAnchor="end">{hd}</text>
      ))}
      {rows.map((r, i) => {
        const y = top + i * rowH + rowH / 2;
        const present = r.values.filter((v) => v.value != null) as (DotValue & { value: number })[];
        const xs = present.map((v) => sx(v.value));
        const xmax = xs.length ? Math.max(...xs) : x0;
        const content: ReactNode = (
          <>
            <div className="tip-title">{r.label}{r.sublabel ? ` · ${r.sublabel}` : ""}</div>
            {r.values.map((v) => <TipRow key={v.series} cls={cls(v.series)} value={v.value == null ? "not measured" : v.text} label={slabel(v.series)} />)}
            {r.notes?.map((n, k) => n && <TipRow key={k} value={n} label={noteHeaders[k]} />)}
          </>
        );
        const aria = `${r.label}: ` + r.values.map((v) => `${slabel(v.series)} ${v.value == null ? "not measured" : v.text}`).join(", ");
        return (
          <g key={r.id} className="dot-row" {...tipTarget(tip, xmax, y - 10, content, aria)}>
            <rect className="hit" x={0} y={y - rowH / 2} width={w} height={rowH} />
            <text className="row-label" x={labelW} y={r.sublabel ? y - 3 : y + 4} textAnchor="end">{r.label}</text>
            {r.sublabel && <text className="row-sub" x={labelW} y={y + 11} textAnchor="end">{r.sublabel}</text>}
            {xs.length > 1 && <line className="dumbbell" x1={Math.min(...xs)} x2={Math.max(...xs)} y1={y} y2={y} />}
            {present.map((v) => <Marker key={v.series} cls={cls(v.series)} x={sx(v.value)} y={y} />)}
            {r.missing && (xmax + 150 < x1
              ? <text className="row-missing" x={xmax + 12} y={y + 4}>{r.missing}</text>
              : <text className="row-missing" x={(xs.length ? Math.min(...xs) : x0) - 12} y={y + 4} textAnchor="end">{r.missing}</text>)}
            {r.notes?.map((n, k) => (
              <text key={k} className="row-note num" x={x1 + 14 + noteW * (k + 1) - 4} y={y + 4} textAnchor="end">{n ?? "—"}</text>
            ))}
          </g>
        );
      })}
    </svg>
  );
}
