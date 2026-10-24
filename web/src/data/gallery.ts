// Gallery manifest, written by web/gallery/collect.py from real Conatus runs (never by hand).
// Loaded with a glob so the site still builds before the first images exist.
import { withBase } from "../lib/base";

export interface GalleryItem {
  slug: string; title: string; image: string; image_small: string; width: number; height: number;
  alt: string; caption: string; colormap: "smoke" | "signed"; run_id: string; script: string;
  example_script: string | null;
  geometry: string; domain: string;
  /** The exact field shown, its colour scale and the region covered (web/gallery/fields.json). */
  field?: string; colour?: string; extent?: string;
  /** What the run that drew the picture measured, from its own diagnostics (collect.py). */
  measured?: string;
  /** The colour range the run used; null when each panel is normalised separately. */
  scale?: { kind: "smoke" | "signed"; lo: number; hi: number } | null;
  /** The film of the case (web/gallery/gallery_films.py); `image` is then its poster. */
  film?: GalleryFilm;
}

export interface GalleryFilm {
  webm: string; mp4: string; webm_small: string; mp4_small: string; poster: string; poster_small: string;
  duration_s: number; fps: number; frames: number;
  bytes: Record<string, number>;
  times: [number, number, number | string];     // first and last simulated time, time between frames
  loop: string;                                 // how the loop closes (said under the film)
  span: string;                                 // simulated time shown and the playback length
  case_script: string;                          // the gallery script whose problem the film reruns
  generation: { command: string; sha256: Record<string, string> };
}

const found = import.meta.glob<GalleryItem[]>("../../public/gallery/manifest.json", { eager: true, import: "default" });
export const gallery: GalleryItem[] = Object.values(found)[0] ?? [];
export const bySlug = (slug: string) => gallery.find((g) => g.slug === slug);
export const src = (p: string) => withBase(p);

/** Title without parameter values ("Kuramoto–Sivashinsky chaos, L = 22" → "Kuramoto–Sivashinsky chaos"). */
export const plainTitle = (g: GalleryItem) =>
  g.title.replace(/,?\s*(at\s+)?(L|Re_tau|Re_τ|Ra|Re)\s*=?\s*[\d.]+(e\d+)?/g, "").replace(/\s+/g, " ").trim();
/** One sentence naming the exact field the image shows (manifest `field`); the visual description
 *  stays in the alt text. Older manifests without it fall back to the first clause of the alt text. */
export const plainCaption = (g: GalleryItem) => {
  if (g.field) return g.field;
  const s = g.alt.split(/[:.;]\s/)[0].replace(/\s*\([^)]*\)/g, "").trim();
  return s.endsWith(".") ? s : s + ".";
};

/** The two brand colormaps (web/gallery/brand_cmaps.py), as CSS gradients for the lightbox scale bar. */
export const RAMPS = {
  smoke: "linear-gradient(to right, #0B0B0B 0%, #3A3A38 30%, #8A8A86 62%, #EDEBE6 100%)",
  signed: "linear-gradient(to right, #D4574A 0%, #6E2019 30%, #0B0B0B 50%, #3F5245 70%, #9DBBA6 100%)",
} as const;
/** A scale-bar end value: three significant digits, true minus sign. */
export const tick = (v: number) => (Math.abs(v) < 1e-12 ? "0" : Number(v.toPrecision(3)).toString().replace("-", "\u2212"));
