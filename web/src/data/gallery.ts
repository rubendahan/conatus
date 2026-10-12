// Gallery manifest, written by web/gallery/collect.py from real Conatus runs (never by hand).
// Loaded with a glob so the site still builds before the first images exist.
import { withBase } from "../lib/base";

export interface GalleryItem {
  slug: string; title: string; image: string; image_small: string; width: number; height: number;
  alt: string; caption: string; colormap: "smoke" | "signed"; run_id: string; script: string;
  example_script: string | null;
  validated: { value: string; reference: string; source: string; run_id: string | null };
  geometry: string; domain: string;
}

const found = import.meta.glob<GalleryItem[]>("../../public/gallery/manifest.json", { eager: true, import: "default" });
export const gallery: GalleryItem[] = Object.values(found)[0] ?? [];
export const bySlug = (slug: string) => gallery.find((g) => g.slug === slug);
export const src = (p: string) => withBase(p);

/** Title without parameter values ("Kuramoto–Sivashinsky chaos, L = 22" → "Kuramoto–Sivashinsky chaos"). */
export const plainTitle = (g: GalleryItem) =>
  g.title.replace(/,?\s*(at\s+)?(L|Re_tau|Re_τ|Ra|Re)\s*=?\s*[\d.]+(e\d+)?/g, "").replace(/\s+/g, " ").trim();
/** One plain sentence saying what the image shows: the first clause of its description (alt text). */
export const plainCaption = (g: GalleryItem) => {
  const s = g.alt.split(/[:.;]\s/)[0].replace(/\s*\([^)]*\)/g, "").trim();
  return s.endsWith(".") ? s : s + ".";
};
