// The navigation of the site, from nav.json: the same file gives the documentation its sidebar links
// and its "Start here" list (docs/site/conf.py), so the website and the Sphinx pages share one map.
import nav from "./nav.json";

export interface NavItem { label: string; short?: string; route: string }
export interface StartStep { title: string; text: string; to?: string; docs?: string }

export const PRIMARY: NavItem[] = nav.primary;
/** The getting-started route, numbered 1 to 4 wherever it is shown. */
export const GETTING_STARTED: StartStep[] = nav.start.map((s) => ({
  title: s.title, text: s.text, to: "route" in s ? s.route : undefined, docs: "docs" in s ? s.docs : undefined,
}));
