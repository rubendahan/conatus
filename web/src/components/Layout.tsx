// Page chrome: skip link, header (mark, navigation, search, theme), main, footer.
// Also keeps <title> in sync after client-side navigation and restores scroll / hash targets.
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import { installTransitions } from "../lib/navigation";
import NextStep from "./NextStep";
import { findRoute, routes } from "../routes";
import { PRIMARY } from "../content/nav";
import Mark from "./Mark";
import Search from "./Search";
import ThemeToggle from "./ThemeToggle";
import { SITE } from "../content/site";
import { withBase } from "../lib/base";

// Primary navigation: content/nav.json, the file that also gives the documentation its links (one map
// for the site and the Sphinx pages). "More" lists the remaining pages (routes.ts, group "more").
const primary = PRIMARY.map((n) => ({ path: n.route, label: n.short ?? n.label }));

// Site map by purpose: the footer columns and the groups of the mobile menu.
const COLS: { title: string; links: [string, string][] }[] = [
  { title: "Learn", links: [["/tutorials/getting-started", "Getting started"], ["/tutorials", "Tutorials"], ["/install", "Install"]] },
  { title: "Reference", links: [["/documentation", "Documentation"], ["/capabilities", "Capabilities"], ["/dynamics", "Dynamics"], ["/geometries", "Geometries"], ["/validation", "Validation"]] },
  { title: "Project", links: [["/about", "About"], ["/gallery", "Gallery"], ["/citing", "Citing"], ["/coming-next", "Coming next"]] },
];

const more = routes.filter((r) => r.group === "more");

// Layout effects run inside the view transition's update (flushSync), so the new snapshot is already
// scrolled to the top or to the anchor; on the server they are plain effects (never run).
const useIsoLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;

function useRouteEffects() {
  const { pathname, hash } = useLocation();
  const navigate = useNavigate();
  const first = useRef(true);
  useEffect(() => installTransitions((to) => navigate(to)), [navigate]);
  useIsoLayoutEffect(() => {
    const r = findRoute(pathname);
    document.title = r.path === "/" ? r.title : `${r.title} · Conatus`;
    if (first.current) { first.current = false; return; }
    if (hash) {
      const el = document.getElementById(decodeURIComponent(hash.slice(1)));
      if (el) { el.scrollIntoView(); return; }
    }
    window.scrollTo(0, 0);
    // Move focus to the main landmark so screen readers announce the new page.
    document.getElementById("main")?.focus({ preventScroll: true });
  }, [pathname, hash]);
}

function Header() {
  const [menu, setMenu] = useState(false);
  const { pathname } = useLocation();
  const moreRef = useRef<HTMLDetailsElement>(null);
  // Every menu closes when the page changes (external review 0.1.0, finding 78: "More" stayed open
  // over the new page); "More" also closes on a click outside it and on Escape.
  useEffect(() => { setMenu(false); if (moreRef.current) moreRef.current.open = false; }, [pathname]);
  useEffect(() => {
    const close = (e: Event) => {
      const d = moreRef.current;
      if (d?.open && (e.type === "keydown" ? (e as KeyboardEvent).key === "Escape" : !d.contains(e.target as Node))) d.open = false;
    };
    document.addEventListener("click", close);
    document.addEventListener("keydown", close);
    return () => { document.removeEventListener("click", close); document.removeEventListener("keydown", close); };
  }, []);
  const btn = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!menu) return;
    // Focus the first link of the opened menu; Escape closes it and returns focus to the button.
    (document.querySelector("#mobile-nav a") as HTMLElement | null)?.focus({ preventScroll: true });
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") { setMenu(false); btn.current?.focus(); } };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [menu]);
  return (
    <header className="site-header">
      <div className="wrap header-inner">
        <Link to="/" className="brand" aria-label="Conatus, home">
          <Mark size={30} className="brand-mark" />
          <span className="brand-name">Conatus</span>
        </Link>
        <nav className="nav-primary" aria-label="Primary">
          <ul>
            {primary.map((r) => (
              <li key={r.path}><NavLink to={r.path} end={r.path === "/tutorials"}>{r.label}</NavLink></li>
            ))}
            <li className="nav-more">
              <details ref={moreRef}>
                <summary>More</summary>
                <ul className="nav-more-list">
                  {more.map((r) => (<li key={r.path}><NavLink to={r.path}>{r.nav ?? r.title}</NavLink></li>))}
                </ul>
              </details>
            </li>
          </ul>
        </nav>
        <div className="header-tools">
          <Search />
          <ThemeToggle />
          <button ref={btn} type="button" className="icon-btn menu-btn" aria-expanded={menu} aria-controls="mobile-nav"
            aria-label={menu ? "Close menu" : "Open menu"} onClick={() => setMenu((m) => !m)}>
            <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true">
              {menu ? <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
                : <path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />}
            </svg>
          </button>
        </div>
      </div>
      <nav id="mobile-nav" className={"nav-mobile" + (menu ? " open" : "")} aria-label="Mobile" hidden={!menu}>
        <div className="nav-mobile-inner">
          {COLS.map((c) => (
            <div key={c.title} className="nav-mobile-group">
              <p className="nav-mobile-title">{c.title}</p>
              <ul>{c.links.map(([to, label]) => <li key={to}><NavLink to={to}>{label}</NavLink></li>)}</ul>
            </div>
          ))}
        </div>
      </nav>
    </header>
  );
}

function Footer() {
  return (
    <footer className="site-footer">
      <div className="wrap footer-inner">
        <div className="footer-brand">
          <Mark size={44} />
          <p className="footer-tagline">Experimental research software for fluid instability and dynamical systems.</p>
        </div>
        {COLS.map((c) => (
          <div key={c.title} className="footer-col">
            <h2 className="footer-title">{c.title}</h2>
            <ul>{c.links.map(([to, label]) => <li key={to}><Link to={to}>{label}</Link></li>)}</ul>
          </div>
        ))}
      </div>
      <div className="wrap footer-base">
        <span>{SITE.license} · Ruben Dahan, {SITE.year} · v{SITE.version}{SITE.docs && <> · <a href={withBase("/docs/")}>Reference documentation</a></>}</span>
        <span>{!__HASH_ROUTER__ && <a href={withBase("/sitemap.xml")}>Sitemap</a>}</span>
      </div>
    </footer>
  );
}

export default function Layout({ children }: { children: ReactNode }) {
  useRouteEffects();
  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <Header />
      <main id="main" tabIndex={-1}>{children}<NextStep /></main>
      <Footer />
    </>
  );
}
