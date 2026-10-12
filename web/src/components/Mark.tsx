// The Conatus lens (brand charter section I): a rim of thickness D/20 and nine polishing rings whose
// radii are the projections of Chebyshev nodes, R cos((2k-1)π/36), k = 1..9, plus a centre dot.
// Below 32 px the favicon (3 rings) is used instead: `small` switches to it.
const D = 200;
const R = 94;
const RINGS = Array.from({ length: 9 }, (_, i) => 86 * Math.cos(((2 * (i + 1) - 1) * Math.PI) / 36));

export default function Mark({ size = 32, title, className }: { size?: number; title?: string; className?: string }) {
  const small = size < 32;
  const rings = small ? [55.17, 46.77, 31.25, 10.98] /* favicon.svg ×6.25 */ : RINGS;
  return (
    <svg className={className} width={size} height={size} viewBox={`0 0 ${D} ${D}`} role={title ? "img" : undefined}
      aria-hidden={title ? undefined : true} aria-label={title} focusable="false">
      <circle cx={100} cy={100} r={R} fill="none" stroke="currentColor" strokeWidth={small ? 19 : 10} />
      {rings.map((r, i) => (
        <circle key={i} cx={100} cy={100} r={r.toFixed(3)} fill="none" stroke="currentColor" strokeWidth={small ? 10 : 1.4} />
      ))}
      <circle cx={100} cy={100} r={small ? 10 : 2.2} fill="currentColor" />
    </svg>
  );
}
