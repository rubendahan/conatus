// Release status: the optional extras (pyproject.toml, with the purposes of src/content/status.json)
// and the status of each part of the release, written by scripts/gen-data.mjs into
// src/generated/status.json. The documentation builds the same tables from the same files
// (docs/site/generate.py, release_status).
export interface Extra { name: string; adds: string; for: string }
export interface FeatureStatus { feature: string; status: string; note: string }

const found = import.meta.glob<{ extras: Extra[]; features: FeatureStatus[] }>("../generated/status.json", { eager: true, import: "default" });
const data = Object.values(found)[0];
export const extras: Extra[] = data?.extras ?? [];
export const features: FeatureStatus[] = data?.features ?? [];
