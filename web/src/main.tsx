// Client entry: hydrate the prerendered HTML. Pages are lazy chunks; the current page's chunk is
// loaded before hydration so the server markup is reused without a flash.
import { StrictMode } from "react";
import { hydrateRoot, createRoot } from "react-dom/client";
import { BrowserRouter, HashRouter } from "react-router";
import App from "./App";
import { findRoute } from "./routes";
import { pages, preload } from "./lib/pages";
import { basename, stripBase } from "./lib/base";
import "./styles/index.css";


// Portable build (HASH=1): routes live in the hash (#/quickstart) so one index.html serves every page;
// in-page anchors (href="#install") would otherwise be read as routes, so they scroll instead.
function portableAnchors() {
  document.addEventListener("click", (e) => {
    const a = (e.target as Element | null)?.closest?.("a[href^='#']") as HTMLAnchorElement | null;
    const id = a?.getAttribute("href")?.slice(1);
    if (!a || !id || id.startsWith("/")) return;
    const el = document.getElementById(decodeURIComponent(id));
    if (!el) return;
    e.preventDefault();
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

async function boot() {
  if (__HASH_ROUTER__) {
    portableAnchors();
    const path = window.location.hash.replace(/^#/, "").split("#")[0] || "/";
    await preload(findRoute(path).path);
    const container = document.getElementById("root")!;
    container.replaceChildren(); // the prerendered home is only a first paint here
    createRoot(container).render(<StrictMode><HashRouter><App pages={pages} /></HashRouter></StrictMode>);
    return;
  }
  const current = findRoute(stripBase(window.location.pathname));
  await preload(current.path); // the page renders synchronously: hydration reuses the prerendered markup
  const container = document.getElementById("root")!;
  const tree = (
    <StrictMode>
      <BrowserRouter basename={basename}>
        <App pages={pages} />
      </BrowserRouter>
    </StrictMode>
  );
  if (container.hasChildNodes()) hydrateRoot(container, tree);
  else createRoot(container).render(tree); // dev server: no prerendered markup
}
boot();
