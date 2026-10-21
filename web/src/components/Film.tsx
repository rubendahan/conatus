// A short looping film of a real Conatus run (web/gallery/films.py): muted, inline, with one
// Pause / Play button (external review 0.1.0, finding 75: the films could not be stopped). The
// poster is the page's image; the video sources are attached only once the film nears the viewport
// (lazy, nothing downloaded below the fold), it plays only while visible and not paused by the
// reader, and under prefers-reduced-motion it never plays (the poster stays, no button). The box
// keeps the film's aspect ratio, so nothing moves when the video arrives. When the film loops through
// a cross-fade or a fade through black, the caption says so: that transition is a display effect,
// not part of the run (finding 77).
import { useEffect, useRef, useState } from "react";
import { withBase } from "../lib/base";
import type { FilmItem } from "../data/films";

const reduced = () => typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/** The sentence that says how the loop closes, from the manifest's `loop` field. */
export function loopNote(loop?: string | null): string | null {
  if (!loop) return null;
  if (/black/i.test(loop)) return "The film fades through black where it loops: a display transition, not part of the run.";
  if (/fade/i.test(loop)) return "The last frames cross-fade into the first where the film loops: a display transition, not part of the run.";
  return null;
}

export default function Film({ f }: { f: FilmItem }) {
  const ref = useRef<HTMLVideoElement>(null);
  const [active, setActive] = useState(false);       // sources attached
  const [paused, setPaused] = useState(false);       // paused by the reader
  const [motion, setMotion] = useState(false);       // motion allowed (after hydration): show the button
  const pausedRef = useRef(false);
  const syncRef = useRef<() => void>(() => {});

  useEffect(() => {
    const v = ref.current;
    if (!v || typeof IntersectionObserver === "undefined") return;
    v.muted = true;                                  // the attribute alone is not reflected by React
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    let visible = false;
    const sync = () => {
      setMotion(!reduced());
      if (reduced() || pausedRef.current) { v.pause(); return; }
      if (visible) { setActive(true); v.play().catch(() => { /* autoplay refused: the poster stays */ }); }
      else v.pause();
    };
    syncRef.current = sync;
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; sync(); }, { rootMargin: "200px 0px" });
    io.observe(v);
    mq?.addEventListener?.("change", sync);
    sync();
    return () => { io.disconnect(); mq?.removeEventListener?.("change", sync); };
  }, []);

  useEffect(() => {                                  // sources just attached: load, then play
    const v = ref.current;
    if (!active || !v || reduced()) return;
    v.load();
    if (!pausedRef.current) v.play().catch(() => { /* autoplay refused: the poster stays */ });
  }, [active]);

  const toggle = () => {
    pausedRef.current = !pausedRef.current;
    setPaused(pausedRef.current);
    syncRef.current();
  };
  const note = loopNote(f.loop);

  return (
    <figure className="film">
      <div className="film-box" style={{ aspectRatio: `${f.width} / ${f.height}` }}>
        <video ref={ref} width={f.width} height={f.height} poster={withBase(f.poster)} muted loop playsInline
          preload="none" aria-label={`${f.title}: a looping film of a Conatus run. ${f.alt}`}>
          {active && <source src={withBase(f.video_webm)} type="video/webm" />}
          {active && <source src={withBase(f.video_mp4)} type="video/mp4" />}
        </video>
        {motion && (
          <button type="button" className="film-btn" onClick={toggle}
            aria-label={`${paused ? "Play" : "Pause"} the film: ${f.title}`}>
            <svg viewBox="0 0 10 10" aria-hidden="true">
              {paused ? <path d="M2 1l7 4-7 4z" /> : <path d="M2 1h2v8H2zM6 1h2v8H6z" />}
            </svg>
            {paused ? "Play" : "Pause"}
          </button>
        )}
      </div>
      <figcaption>
        <strong>{f.title}</strong>
        <span>{f.caption}{f.details && <> {f.details}</>}</span>
        {(f.legend || f.span) && <span className="film-legend">{[f.legend, f.span].filter(Boolean).join(" ")}</span>}
        {note && <span>{note}</span>}
      </figcaption>
    </figure>
  );
}
