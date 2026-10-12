// Theme: follows the system (prefers-color-scheme) unless the reader chose one; the choice is a
// per-viewer convenience kept in localStorage (wrapped in try/catch: storage may be blocked).
export type ThemeChoice = "system" | "light" | "dark";
const KEY = "cn-theme";

export function readChoice(): ThemeChoice {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch { return "system"; }
}
export function effectiveTheme(choice: ThemeChoice): "light" | "dark" {
  if (choice !== "system") return choice;
  return typeof matchMedia !== "undefined" && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
export function applyChoice(choice: ThemeChoice) {
  const root = document.documentElement;
  if (choice === "system") delete root.dataset.theme; else root.dataset.theme = choice;
  try { if (choice === "system") localStorage.removeItem(KEY); else localStorage.setItem(KEY, choice); } catch { /* ignore */ }
}
/** Inline, render-blocking snippet placed in <head> so the chosen theme applies before first paint. */
export const themeBootScript =
  `try{var t=localStorage.getItem("${KEY}");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;
