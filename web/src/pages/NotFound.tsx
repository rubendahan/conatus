// 404 page: one lens, and a message written like a Conatus error (what, why, what to do).
// The requested path is filled in after hydration only, so the prerendered 404.html matches the client render.
import { useEffect, useState } from "react";
import { useLocation } from "react-router";
import Mark from "../components/Mark";
import { ButtonLink } from "../components/ui";
import "../styles/info.css";

export default function NotFound() {
  const { pathname } = useLocation();
  const [path, setPath] = useState<string | null>(null);
  useEffect(() => setPath(pathname), [pathname]);
  return (
    <div className="wrap">
      <div className="info-404">
        <div className="info-404-mark"><Mark size={160} /></div>
        <div>
          <p className="eyebrow">Error 404</p>
          <h1>Page not found</h1>
          <div className="info-err" role="note" aria-label="What happened">
            <p><span className="info-err-kind">PageNotFoundError</span>: no page at <code>{path ?? "this address"}</code>.</p>
            <p><span className="info-err-key">why:</span> the address may be mistyped, or the page may have moved.</p>
            <p><span className="info-err-key">fix:</span> start from the home page, the quickstart or the documentation. Nothing
              was redirected for you.</p>
          </div>
          <div className="row">
            <ButtonLink to="/">Home</ButtonLink>
            <ButtonLink to="/tutorials" variant="ghost">Tutorials</ButtonLink>
            <ButtonLink to="/documentation" variant="ghost">Documentation</ButtonLink>
          </div>
        </div>
      </div>
    </div>
  );
}
