// Gallery: a grid of short looping films of Conatus runs (web/gallery/gallery_films.py; brand charter III
// colour maps), each with its name and the exact field it shows (manifest `field`). A case without a film
// keeps its still image. The lightbox shows the full film, the colour scale, the run's measured values and
// the script, linked at the release tag.
// Data: public/gallery/manifest.json, written by web/gallery/collect.py (never by hand).
// A card opens in the lightbox (components/Lightbox.tsx) and its URL hash (#slug) opens it on
// load; without JavaScript the card is a plain link to the film (MP4) or the full-resolution image.
import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import { PageHead, PageLink, Section } from "../components/ui";
import Lightbox from "../components/Lightbox";
import GalleryVideo from "../components/GalleryVideo";
import { gallery, plainCaption, plainTitle, src, type GalleryItem } from "../data/gallery";
import "../styles/gallery.css";
import "../styles/films.css";

const DOMAIN_LABEL: Record<string, string> = {
  shear: "Shear flows", convection: "Convection", patterns: "Patterns", pipe: "Pipes", rotating: "Rotating flows",
  turbulence: "Turbulence", dynamics: "Dynamics",
};
const fig = (g: GalleryItem) => gallery.indexOf(g) + 1;
const thumb = (slug: string) => document.querySelector<HTMLElement>(`[data-plate="${CSS.escape(slug)}"]`);

/** Write the open plate into the URL hash without a navigation (works with the hash router too). */
function setHash(slug: string | null) {
  const h = window.location.hash;
  let url: string;
  if (h.startsWith("#/")) url = "#" + h.slice(1).split("#")[0] + (slug ? "#" + slug : "");
  else url = slug ? "#" + slug : window.location.pathname + window.location.search;
  window.history.replaceState(window.history.state, "", url);
}

function Plate({ g, onOpen }: { g: GalleryItem; onOpen: (e: MouseEvent, g: GalleryItem) => void }) {
  const n = fig(g);
  const h800 = Math.round((800 * g.height) / g.width);
  const open = (
    <span className="gplate-open" aria-hidden="true">
      <svg viewBox="0 0 16 16"><path d="M9.5 2.5h4v4M13.5 2.5 9 7M6.5 13.5h-4v-4M2.5 13.5 7 9" /></svg>
    </span>
  );
  return (
    <article className="gplate" id={g.slug} aria-labelledby={`${g.slug}-t`}>
      <figure className="gplate-fig">
        {g.film ? (
          <div className="gplate-media film-box" style={{ aspectRatio: `${g.width} / ${g.height}` }}>
            <GalleryVideo g={g} dataPlate={g.slug} wrap={(video) => (
              <a href={src(g.film!.mp4)} className="gplate-img" onClick={(e) => onOpen(e, g)}
                aria-label={`${plainTitle(g)}: open the film at full size`}>
                {video}
                {open}
              </a>
            )} />
          </div>
        ) : (
          <a href={src(g.image)} className="gplate-img" onClick={(e) => onOpen(e, g)}
            aria-label={`${plainTitle(g)}: open the image at full resolution`}>
            <img data-plate={g.slug} src={src(g.image_small)} srcSet={`${src(g.image_small)} 800w, ${src(g.image)} ${g.width}w`}
              sizes="(max-width: 900px) 100vw, 50vw" width={800} height={h800} alt={g.alt} loading={n > 2 ? "lazy" : "eager"} decoding="async" />
            {open}
          </a>
        )}
      </figure>
      <div className="gplate-meta">
        <p className="gplate-kicker">{DOMAIN_LABEL[g.domain] ?? g.domain} · {g.geometry}{g.film ? "" : " · still image"}</p>
        <h2 id={`${g.slug}-t`}>{plainTitle(g)}</h2>
        <p className="gplate-caption">{plainCaption(g)}</p>
      </div>
    </article>
  );
}

export default function Gallery() {
  const domains = useMemo(() => Array.from(new Set(gallery.map((g) => g.domain))), []);
  const [domain, setDomain] = useState<string | null>(null);
  const shown = domain ? gallery.filter((g) => g.domain === domain) : gallery;
  const [open, setOpen] = useState<number | null>(null);
  const [morph, setMorph] = useState(true);

  const onOpen = (e: MouseEvent, g: GalleryItem) => {
    if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;   // new tab: the image
    e.preventDefault();
    setMorph(true);
    setOpen(shown.indexOf(g));
  };
  const onChange = useCallback((i: number | null) => {
    setMorph(false);
    setOpen(i);
  }, []);

  // hash on load (a shared link, or a plate linked from the home page) opens that plate
  useEffect(() => {
    const slug = decodeURIComponent(window.location.hash.split("#").pop() ?? "");
    const i = gallery.findIndex((g) => g.slug === slug);
    if (i >= 0) { setMorph(false); setOpen(i); }
  }, []);
  const wasOpen = useRef(false);
  useEffect(() => {
    if (open === null && !wasOpen.current) return;           // leave an unrelated anchor (#method) alone
    wasOpen.current = open !== null;
    setHash(open === null ? null : shown[open]?.slug ?? null);
  }, [open, shown]);

  return (
    <>
      <PageHead eyebrow="Gallery" title="Fluids, everywhere">
        <p>Short films of flows computed with Conatus. Each frame is a state of the run at a recorded time, drawn in the
          colours of <i>An Album of Fluid Motion</i>: smoke greys from low to high, or signed fields through black at zero,
          from red to glass green. Each card names the field it shows. Open one to see the film at full size, with its
          colour scale, the time it covers, the values the run measured and the script that made it.</p>
        <p>Results of published papers computed again, with the paper's values next to Conatus's and the gap between
          them, are on the <PageLink to="/published">Published cases</PageLink> page.</p>
      </PageHead>
      <Section id="plates" className="gallery-section">
        {gallery.length === 0 ? (
          <p className="note">The gallery is empty in this build.</p>
        ) : (
          <>
            <div className="chips" role="group" aria-label="Filter by domain">
              <button type="button" className="chip" aria-pressed={domain === null} onClick={() => setDomain(null)}>All </button>
              {domains.map((d) => (
                <button key={d} type="button" className="chip" aria-pressed={domain === d} onClick={() => setDomain(d)}>
                  {DOMAIN_LABEL[d] ?? d}
                </button>
              ))}
            </div>
            <div className="gplates">
              {shown.map((g) => <Plate key={g.slug} g={g} onOpen={onOpen} />)}
            </div>
            <Lightbox items={shown} index={open !== null && open < shown.length ? open : null} morph={morph}
              origin={thumb} onChange={onChange} />
          </>
        )}
      </Section>

    </>
  );
}
