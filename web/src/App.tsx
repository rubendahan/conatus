// Application shell: layout + routes. The same tree is rendered by the server (prerender, with
// resolved page components) and hydrated by the client (with lazy components per route), so
// every page ships only its own code.
import { Suspense, type ComponentType } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { routes, notFound, redirects } from "./routes";
import Layout from "./components/Layout";

export type PageMap = Record<string, ComponentType>;

export default function App({ pages }: { pages: PageMap }) {
  const NotFound = pages[notFound.path];
  return (
    <Layout>
      <Suspense fallback={<div className="page-loading" aria-busy="true" />}>
        <Routes>
          {routes.map((r) => {
            const Page = pages[r.path];
            return <Route key={r.path} path={r.path} element={<Page />} />;
          })}
          {redirects.map((r) => <Route key={r.from} path={r.from} element={<Navigate to={r.to} replace />} />)}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Suspense>
    </Layout>
  );
}
