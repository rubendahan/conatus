// Page registry on the client: one lazy chunk per route, plus a synchronous cache. A page whose module
// is already loaded renders without suspending, which lets hydration reuse the prerendered markup and
// lets a navigation run inside a view transition (lib/navigation.ts preloads the next page first, so
// the new snapshot is the new page, never a loading placeholder).
import { lazy, type ComponentType } from "react";
import { allRoutes, findRoute, type RouteDef } from "../routes";

const loaded = new Map<string, ComponentType>();

function page(r: RouteDef): ComponentType {
  const Lazy = lazy(async () => { const m = await r.load(); loaded.set(r.path, m.default); return m; });
  function Page() {
    const C = loaded.get(r.path);
    return C ? <C /> : <Lazy />;
  }
  Page.displayName = `Page(${r.path})`;
  return Page;
}

export const pages: Record<string, ComponentType> = Object.fromEntries(allRoutes.map((r) => [r.path, page(r)]));

/** Load the module of the page at `pathname` (router path, without base) into the cache. */
export async function preload(pathname: string): Promise<void> {
  const r = findRoute(pathname);
  if (loaded.has(r.path)) return;
  loaded.set(r.path, (await r.load()).default);
}
