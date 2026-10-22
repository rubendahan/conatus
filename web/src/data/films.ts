// Films manifest, written by web/gallery/films.py --collect from real Conatus runs (never by hand).
// Loaded with a glob so the site still builds before the first films exist.
export interface FilmItem {
  slug: string; title: string; alt: string; caption: string;
  video_webm: string; video_mp4: string; poster: string;
  width: number; height: number; duration_s: number;
  run_id?: string; script: string; example_script: string | null; domain: string;   // run_id: private builds only
  loop?: string | null;                              // how the loop closes (display only, said in the details)
  shows?: string | null;                             // first visible line: what the film shows
  key?: string | null;                               // second visible line: the key parameter
  legend?: string | null;                            // what the colours show (quantity and range)
  details?: string | null;                           // physics and numerics of the run
  span?: string | null;                              // simulated time shown, and the playback length
}

const found = import.meta.glob<FilmItem[]>("../../public/films/manifest.json", { eager: true, import: "default" });
export const films: FilmItem[] = Object.values(found)[0] ?? [];
