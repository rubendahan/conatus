// Public base path (Vite `base`, e.g. "/conatus/" on a GitHub Pages project site).
export const base: string = import.meta.env.BASE_URL || "/";
// A relative base ("./", portable build with hash routes) has no router basename.
export const basename: string = base === "/" || base.startsWith(".") ? "/" : base.replace(/\/$/, "");
/** Remove the base prefix from a browser pathname. */
export function stripBase(pathname: string): string {
  if (basename !== "/" && pathname.startsWith(basename)) return pathname.slice(basename.length) || "/";
  return pathname;
}
/** Prefix an absolute site path (for plain <a>/<img>, not router links). */
export function withBase(path: string): string {
  if (/^(https?:)?\/\//.test(path) || path.startsWith("#") || path.startsWith("mailto:")) return path;
  return (base.replace(/\/$/, "") + (path.startsWith("/") ? path : "/" + path)) || "/";
}
