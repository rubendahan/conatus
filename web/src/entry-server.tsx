// Server entry used only by scripts/prerender.mjs: renders one path to static HTML.
import { StrictMode, type ComponentType } from "react";
import { renderToString } from "react-dom/server";
import { StaticRouter } from "react-router-dom/server";
import App, { type PageMap } from "./App";
import { allRoutes, findRoute, redirects, routes } from "./routes";
import { basename } from "./lib/base";

export { routes, allRoutes, findRoute, redirects };

let pages: PageMap | null = null;
async function loadPages(): Promise<PageMap> {
  if (pages) return pages;
  const entries = await Promise.all(allRoutes.map(async (r) => [r.path, (await r.load()).default] as [string, ComponentType]));
  pages = Object.fromEntries(entries);
  return pages;
}

export async function render(path: string): Promise<string> {
  const p = await loadPages();
  return renderToString(
    <StrictMode>
      <StaticRouter location={basename.replace(/\/$/, "") + path} basename={basename}>
        <App pages={p} />
      </StaticRouter>
    </StrictMode>,
  );
}
