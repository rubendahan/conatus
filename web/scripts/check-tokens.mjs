// Check that the website's tokens equal the brand tokens (docs/brand/tokens.css, brand charter),
// apart from the Google Fonts @import (the site self-hosts its fonts) and comments.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const here = path.dirname(fileURLToPath(import.meta.url));
const norm = (s) => s.replace(/\/\*[\s\S]*?\*\//g, "").replace(/@import[^\n]*\n/g, "").replace(/\s+/g, " ").trim();
const brand = norm(fs.readFileSync(path.join(here, "../../docs/brand/tokens.css"), "utf8"));
const site = norm(fs.readFileSync(path.join(here, "../src/styles/tokens.css"), "utf8"));
if (brand !== site) { console.error("web/src/styles/tokens.css differs from docs/brand/tokens.css: copy the brand tokens again."); process.exit(1); }
console.log("tokens: identical to docs/brand/tokens.css");
