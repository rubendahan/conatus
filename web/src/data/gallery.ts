// Gallery manifest, written by web/gallery/collect.py from real Conatus runs (never by hand).
// Loaded with a glob so the site still builds before the first images exist.
import { withBase } from "../lib/base";

export interface GalleryItem {
  slug: string; title: string; image: string; image_small: string; width: number; height: number;
  alt: string; caption: string; colormap: "smoke" | "signed"; run_id: string; script: string;
  example_script: string | null;
  validated: { value: string; reference: string; source: string; run_id: string | null };
  geometry: string; domain: string;
  /** The exact field shown and its colour scale (web/gallery/fields.json, read from the scripts). */
  field?: string; colour?: string;
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
