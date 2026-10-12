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
        Every simulation can write the “Methods” paragraph of your paper from the objects themselves (resolution, boundary
        treatment, time stepper, tolerances), with its BibTeX, so the methods you used are cited as they were run. Conatus
        follows Dedalus
        (Burns et al. 2020) for its symbolic language and spectral methods; please cite it too when you rely on that design.</p>} />
    </>
  );
}
