// MathJax 4 configuration of the documentation (docs/site/conf.py: mathjax_config_path).
// - Self-hosted: the font is loaded from the vendored package next to MathJax ([mathjax] = the
//   folder of tex-chtml.js under _static/vendor), never from the CDN default.
// - Semantic enrichment, speech and Braille (the speech-rule engine) are off by default: they ran on
//   the main thread for every formula; the reader turns them on in MathJax's menu. Screen readers
//   get the hidden MathML copy (assistiveMml).
// - Typesetting is spread over the page's idle time, one formula per task, instead of one long
//   task at load: Lighthouse measured MathJax as the main-thread cost of the long tutorial pages
//   (performance 41 on the getting-started tutorial). Every formula is typeset,
//   on- or off-screen (so search in the page and printing see them), in document order.
window.MathJax = {
  loader: { paths: { fonts: "[mathjax]/../@mathjax" } },
  options: { menuOptions: { settings: { enrich: false, speech: false, braille: false, assistiveMml: true } } },
  startup: {
    typeset: false,
    pageReady() {
      return MathJax.startup.defaultPageReady().then(async () => {
        const items = Array.from(document.querySelectorAll(".math"));
        for (const el of items) {
          await new Promise((r) => setTimeout(r, 0));     // yield: input stays responsive
          await MathJax.typesetPromise([el]);
        }
      });
    },
  },
};
