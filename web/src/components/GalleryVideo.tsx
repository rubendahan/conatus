// The film of a gallery case (web/gallery/gallery_films.py), muted and looping. The poster is drawn
// first and nothing else is downloaded (preload="none"); the video sources are attached when the film
// comes near the viewport. It plays only while visible and pauses off-screen (IntersectionObserver).
// Under prefers-reduced-motion it never starts on its own: the poster stays, with a Play button.
// The Pause / Play button is outside the link that opens the lightbox, so it never opens it.
import { useEffect, useRef, useState, type CSSProperties, type ReactNode, type Ref } from "react";
import { src, type GalleryItem } from "../data/gallery";

export const reducedMotion = () =>
  typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/** WebM and MP4 sources of a film: the 800 px pair, or the desktop pair for a wide screen. */
export function filmSources(g: GalleryItem, large: boolean) {
  const f = g.film!;
  return large ? [{ s: f.webm, type: "video/webm" }, { s: f.mp4, type: "video/mp4" }]
    : [{ s: f.webm_small, type: "video/webm" }, { s: f.mp4_small, type: "video/mp4" }];
}

/** The loop sentence under a film, from the manifest's `loop`. */
export function loopSentence(loop: string): string {
  if (/black/i.test(loop)) return "The film fades through black where it loops: a display transition, not part of the run.";
  if (/cross-fade/i.test(loop)) return "The last frames are cross-faded into the first where the film loops: a display transition, not part of the run.";
  return "One period of the motion: the frame after the last is the first.";
}

interface Props {
  g: GalleryItem;
  large?: boolean;                       // desktop files (lightbox on a wide screen)
  className?: string;
  style?: CSSProperties;
  videoRef?: Ref<HTMLVideoElement>;
  dataPlate?: string;                    // thumbnail marker for the lightbox morph
  wrap?: (video: ReactNode) => ReactNode;   // e.g. the link that opens the lightbox (the button stays outside)
}

export default function GalleryVideo({ g, large = false, className, style, videoRef, dataPlate, wrap = (v) => v }: Props) {
  const own = useRef<HTMLVideoElement | null>(null);
  const [active, setActive] = useState(false);          // sources attached
  const [paused, setPaused] = useState(false);          // paused by the reader (or not started)
  const [ready, setReady] = useState(false);            // after hydration: the button exists
  const userRef = useRef<boolean | null>(null);         // the reader's choice: true = paused
  const visRef = useRef(false);
  const syncRef = useRef<() => void>(() => {});
  const f = g.film!;

  const setRefs = (v: HTMLVideoElement | null) => {
    own.current = v;
    if (typeof videoRef === "function") videoRef(v);
    else if (videoRef) (videoRef as { current: HTMLVideoElement | null }).current = v;
  };

  useEffect(() => {
    const v = own.current;
    if (!v) return;
    v.muted = true;                                       // the attribute alone is not reflected by React
    setReady(true);
    const sync = () => {
      const wantPause = userRef.current ?? reducedMotion();   // reduced motion: paused until Play
      setPaused(wantPause);
      if (wantPause || !visRef.current) { v.pause(); return; }
      setActive(true);
      v.play().catch(() => { /* autoplay refused: the poster stays */ });
    };
    syncRef.current = sync;
    if (typeof IntersectionObserver === "undefined") { visRef.current = true; sync(); return; }
    const io = new IntersectionObserver(([e]) => { visRef.current = e.isIntersecting; sync(); }, { rootMargin: "150px 0px" });
    io.observe(v);
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    mq?.addEventListener?.("change", sync);
    return () => { io.disconnect(); mq?.removeEventListener?.("change", sync); };
  }, []);

  useEffect(() => {                                       // sources just attached: load, then play
    const v = own.current;
    if (!active || !v) return;
    v.load();
    if (!(userRef.current ?? reducedMotion()) && visRef.current) v.play().catch(() => { /* poster stays */ });
  }, [active, large]);

  const toggle = () => {
    userRef.current = !(userRef.current ?? reducedMotion());
    syncRef.current();
  };

  const video = (
    <video ref={setRefs} className={className} style={style} draggable={false} data-plate={dataPlate} width={g.width} height={g.height}
        poster={src(large ? f.poster : f.poster_small)} muted loop playsInline preload="none"
        aria-label={`${g.title}: a looping film of a Conatus run, ${f.duration_s.toFixed(0)} s. ${g.alt}`}>
      {active && filmSources(g, large).map((x) => <source key={x.s} src={src(x.s)} type={x.type} />)}
    </video>
  );
  return (
    <>
      {wrap(video)}
      {ready && (
        <button type="button" className="film-btn" onClick={toggle} onPointerDown={(e) => e.stopPropagation()}
          aria-label={`${paused ? "Play" : "Pause"} the film: ${g.title}`}>
          <svg viewBox="0 0 10 10" aria-hidden="true">
            {paused ? <path d="M2 1l7 4-7 4z" /> : <path d="M2 1h2v8H2zM6 1h2v8H6z" />}
          </svg>
          {paused ? "Play" : "Pause"}
        </button>
      )}
    </>
  );
}
