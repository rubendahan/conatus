// Citing: BibTeX generated from CITATION.cff at build time (generated/citation.json) and the
// file itself, imported raw. Nothing is retyped.
import { Code, CopyButton, PageHead, Section } from "../components/ui";
import { citation } from "../data";
import "../styles/tutorials.css";

function Bib({ title, text }: { title: string; text: string }) {
  return (
    <div className="bib">
      <div className="bib-head"><h2 className="bib-title">{title}</h2><CopyButton text={text} label="Copy BibTeX" /></div>
      <Code lang="bibtex">{text}</Code>
    </div>
  );
}

export default function Citing() {
  return (
    <>
      <PageHead eyebrow="Citing" title="How to cite Conatus">
        <p>If Conatus contributes to your work, please cite the software.</p>
      </PageHead>
      <Section id="bibtex" className="tut-section">
        <div className="grid grid-2">
          <Bib title="Software" text={citation.bibtex.software} />
        </div>
      </Section>
      <Section id="methods" title="Cite what you used" intro={<p>
        <code>sim.methods()</code> drafts a “Methods” paragraph from the recorded numerical settings (resolution, boundary
        treatment, time stepper, tolerances), with the BibTeX of the methods used. Add your study's validation and
        interpretation. The symbolic language and the sparse spectral bases build on Dedalus (Burns et al. 2020);
        please cite it as well when your work relies on them.</p>} />
    </>
  );
}
