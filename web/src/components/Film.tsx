// A short looping film of a real Conatus run (web/gallery/films.py): no controls, muted, inline.
// The poster is the page's image; the video sources are attached only once the film nears the
// viewport (lazy, nothing downloaded below the fold), it plays only while visible, and under
// prefers-reduced-motion it never plays (the poster stays). The box keeps the film's aspect ratio,
// so nothing moves when the video arrives.
import { useEffect, useRef, useState } from "react";
import { withBase } from "../lib/base";
import type { FilmItem } from "../data/films";

const reduced = () => typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

export default function Film({ f }: { f: FilmItem }) {
  const ref = useRef<HTMLVideoElement>(null);
  const [active, setActive] = useState(false);       // sources attached

  useEffect(() => {
    const v = ref.current;
    if (!v || typeof IntersectionObserver === "undefined") return;
    v.muted = true;                                  // the attribute alone is not reflected by React
    const motion = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    let visible = false;
    const sync = () => {
      if (reduced()) { v.pause(); return; }
      if (visible) { setActive(true); v.play().catch(() => { /* autoplay refused: the poster stays */ }); }
      else v.pause();
    };
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; sync(); }, { rootMargin: "200px 0px" });
    io.observe(v);
    motion?.addEventListener?.("change", sync);
    return () => { io.disconnect(); motion?.removeEventListener?.("change", sync); };
  }, []);

  useEffect(() => {                                  // sources just attached: load, then play
    const v = ref.current;
    if (!active || !v || reduced()) return;
    v.load();
    v.play().catch(() => { /* autoplay refused: the poster stays */ });
  }, [active]);

  return (
    <figure className="film">
      <div className="film-box" style={{ aspectRatio: `${f.width} / ${f.height}` }}>
        <video ref={ref} width={f.width} height={f.height} poster={withBase(f.poster)} muted loop playsInline
          preload="none" aria-label={`${f.title}: a looping film of a Conatus run. ${f.alt}`}>
          {active && <source src={withBase(f.video_webm)} type="video/webm" />}
          {active && <source src={withBase(f.video_mp4)} type="video/mp4" />}
        </video>
      </div>
      <figcaption>
        <strong>{f.title}</strong>
        <span>{f.caption}</span>
      </figcaption>
    </figure>
  );
}
