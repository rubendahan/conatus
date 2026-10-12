// Light (paper) / dark (lab black) switch. Starts from the system preference.
import { useEffect, useState } from "react";
import { applyChoice, effectiveTheme, readChoice, type ThemeChoice } from "../lib/theme";

export default function ThemeToggle() {
  const [choice, setChoice] = useState<ThemeChoice>("system");
  const [eff, setEff] = useState<"light" | "dark">("light");
  useEffect(() => {
    const c = readChoice();
    setChoice(c); setEff(effectiveTheme(c));
    const mq = matchMedia("(prefers-color-scheme: dark)");
    const on = () => setEff(effectiveTheme(readChoice()));
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  const next: ThemeChoice = eff === "dark" ? "light" : "dark";
  const label = `Switch to the ${next} theme`;
  return (
    <span className="theme-toggle">
      <button type="button" className="icon-btn" aria-label={label} title={label}
        onClick={() => { applyChoice(next); setChoice(next); setEff(next); }}>
        {eff === "dark" ? (
          <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4.5" fill="none" stroke="currentColor" strokeWidth="1.6"/><g stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">{[0,45,90,135,180,225,270,315].map((a)=>(<line key={a} x1="12" y1="2.5" x2="12" y2="4.5" transform={`rotate(${a} 12 12)`}/>))}</g></svg>
        ) : (
          <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round"/></svg>
        )}
      </button>
      {choice !== "system" && (
        <button type="button" className="linklike theme-reset" onClick={() => { applyChoice("system"); setChoice("system"); setEff(effectiveTheme("system")); }}>
          <span className="visually-hidden">Follow the system theme</span><span aria-hidden="true">auto</span>
        </button>
      )}
    </span>
  );
}
