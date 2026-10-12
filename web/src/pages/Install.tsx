// Install page (public 0.1.0): the wheels first (CPU on PyPI, CUDA 12 beta on the GitHub release), then the
// builds from source, extras, how to check an installation.
// Sources: pyproject.toml (build options, extras) and docs/site/install.md.
import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { Code, PageHead, Section } from "../components/ui";
import "../styles/info.css";

const PIP_CPU = `$ python3.12 -m venv venv && . venv/bin/activate
$ pip install "conatus[tutorials]"           # CPU wheel from PyPI + matplotlib for the tutorials
$ python -m conatus doctor                   # what this machine can run`;

const PIP_GPU = `# download the conatus-0.1.0+cu12-cp3XX-…whl of your Python from the GitHub release, then
$ python3.12 -m venv venv && . venv/bin/activate
$ pip install "./conatus-0.1.0+cu12-cp312-cp312-<platform>.whl[gpu,tutorials]"
$ python -m conatus doctor                   # on a GPU node: "GPU (backend='cuda')": ok`;

const CPU_ENV = `# LAPACK and BLAS come from the active conda env only (no system fallback)
$ micromamba create -n conatus -c conda-forge python=3.12 gxx_linux-64=13 liblapack libblas
$ micromamba activate conatus`;

const CPU_INSTALL = `$ git clone https://github.com/rubendahan/conatus && cd conatus
$ pip install ".[tutorials]"                      # CPU (the default) + matplotlib for the tutorials`;

const GPU_ENV = `# in the same env: nvcc, cuFFT, cuBLAS and CuPy from conda-forge (CUDA 12.9 tested)
$ micromamba install -c conda-forge cuda-version=12.9 cuda-nvcc cuda-cudart-dev libcufft-dev libcublas-dev cupy-core`;

const GPU_INSTALL = `$ pip install . -Ccmake.define.CONATUS_CUDA=ON \\
      -Ccmake.define.CMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc \\
      -Ccmake.define.CMAKE_CUDA_HOST_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ \\
      -Ccmake.define.CUDAToolkit_ROOT=$CONDA_PREFIX
# to target your GPU architecture, add
#   -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=<your arch>`;

const CHECK = `$ python -m conatus doctor
# what this machine can run (compiled core, GPU), and what to do about what is missing
$ python -c "import conatus.expert as ex; print(ex.explain())"
# what the single import contains, and any name clash
$ python -c "import conatus.runtime as r; print(r.CORE_SOURCE)"
installed
# CUDA wheel or build only: True on a node with a visible GPU
$ python -c "from conatus.runtime import cuda; print(cuda.available())"`;


type Tab = { id: string; label: string; body: ReactNode };

/** Accessible tabs (WAI-ARIA tabs pattern). Before hydration every panel is shown, one after the other,
 *  so the prerendered page is complete without JavaScript. */
function Tabs({ tabs, label }: { tabs: Tab[]; label: string }) {
  const [ready, setReady] = useState(false);
  const [sel, setSel] = useState(0);
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  useEffect(() => setReady(true), []);
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const n = tabs.length;
    let i = sel;
    if (e.key === "ArrowRight") i = (sel + 1) % n;
    else if (e.key === "ArrowLeft") i = (sel - 1 + n) % n;
    else if (e.key === "Home") i = 0;
    else if (e.key === "End") i = n - 1;
    else return;
    e.preventDefault();
    setSel(i);
    refs.current[i]?.focus();
  };
  return (
    <div className={"info-tabs" + (ready ? " is-ready" : "")}>
      {ready && (
        <div className="info-tablist" role="tablist" aria-label={label} onKeyDown={onKey}>
          {tabs.map((t, i) => (
            <button key={t.id} ref={(el) => { refs.current[i] = el; }} type="button" role="tab" className="info-tab"
              id={`tab-${t.id}`} aria-controls={`panel-${t.id}`} aria-selected={i === sel} tabIndex={i === sel ? 0 : -1}
              onClick={() => setSel(i)}>{t.label}</button>
          ))}
        </div>
      )}
      {tabs.map((t, i) => (
        <div key={t.id} className="info-panel" id={`panel-${t.id}`}
          {...(ready ? { role: "tabpanel", "aria-labelledby": `tab-${t.id}`, hidden: i !== sel, tabIndex: 0 } : {})}>
          {!ready && <h3>{t.label}</h3>}
          {t.body}
        </div>
      ))}
    </div>
  );
}

const cpuSteps = (
  <ol className="info-steps">
    <li>
      <h3 className="rule-h">Create a conda environment with the compiled dependencies</h3>
      <Code lang="bash">{CPU_ENV}</Code>
    </li>
    <li>
      <h3 className="rule-h">Build and install</h3>
      <p><code>pip install ".[tutorials]"</code> compiles the C++20 core (<code>conatus._core</code>, via scikit-build-core, CMake and
        nanobind), installs it inside the package with its dependencies (h5py included: outputs and checkpoints) and
        adds matplotlib, which the tutorials use for their figures. <code>pip install .</code> alone skips matplotlib.</p>
      <Code lang="bash">{CPU_INSTALL}</Code>
    </li>
    <li>
      <h3 className="rule-h">Check</h3>
      <p>See <a href="#check">Check your installation</a> below.</p>
    </li>
  </ol>
);

const gpuSteps = (
  <ol className="info-steps">
    <li>
      <h3 className="rule-h">Start from the CPU environment and add the CUDA toolkit</h3>
      <Code lang="bash">{GPU_ENV}</Code>
    </li>
    <li>
      <h3 className="rule-h">Build with the CUDA backend</h3>
      <p>The CUDA backend is switched on at build time. CuPy comes from conda here: do not add the <code>gpu</code>
        extra, which would install a second CuPy from pip.</p>
      <Code lang="bash">{GPU_INSTALL}</Code>
    </li>
    <li>
      <h3 className="rule-h">Check on a GPU node</h3>
      <p>If the GPU cannot be used, Conatus says why: the core was built without CUDA, or no device is visible.</p>
    </li>
  </ol>
);

const extras: [string, string, string][] = [
  ["tutorials", "matplotlib ≥ 3.7", "The figures of the tutorials (the optional Dedalus check of tutorial 14 needs Dedalus)."],
  ["gpu", "cupy-cuda12x ≥ 13", "Device arrays, with the CUDA wheel (a source build also needs CONATUS_CUDA=ON)."],
  ["notebook", "ipywidgets ≥ 8", "Rich displays of explain() in Jupyter."],
  ["dedalus-compare", "dedalus ≥ 3.0", "Side-by-side comparisons with Dedalus."],
  ["docs", "sphinx ≥ 7, myst-parser ≥ 2", "Building the documentation site."],
  ["dev", "pytest ≥ 8, ruff ≥ 0.5, mpmath ≥ 1.3", "Tests and lint."],
  ["all", "tutorials, notebook, gpu, docs, dev", "Everything above except dedalus-compare."],
];

export default function Install() {
  return (
    <>
      <PageHead eyebrow="Install" title="Install">
        <p>Conatus 0.1.0 installs with pip from a binary wheel, with its compiled core inside: no compiler needed. The CPU
          wheel is on PyPI; a CUDA 12 wheel for NVIDIA GPUs, in beta, is attached to the{" "}
          <a href="https://github.com/rubendahan/conatus/releases/tag/v0.1.0">GitHub release</a>. It also builds from source.</p>
      </PageHead>

      <Section id="requirements" title="Requirements">
        <dl className="info-kv">
          <dt>System</dt>
          <dd>Linux x86-64. Python 3.11, 3.12 or 3.13: 3.12 runs the whole test suite; on all three the wheel installs,
            imports, solves a problem and passes <code>doctor</code> in a clean manylinux_2_28 image; 3.12 is recommended. A newer Python is refused until it is tested.</dd>
          <dt>Python dependencies</dt>
          <dd>NumPy ≥ 1.26, SciPy ≥ 1.11, SymPy ≥ 1.12, h5py ≥ 3.8, installed by pip; matplotlib ≥ 3.7 with the{" "}
            <code>tutorials</code> extra.</dd>
          <dt>Wheels</dt>
          <dd>Linux x86-64 with glibc 2.28 or newer (RHEL 8, Ubuntu 20.04 and later). The CUDA wheel needs an NVIDIA driver
            for CUDA 12.9 (≥ 575, or an older 12.x driver through minor-version compatibility); it pulls the CUDA runtime,
            cuFFT and cuBLAS as pip wheels (about 700 MB). Built for A100 (sm_80) and L40S (sm_89); beta.</dd>
          <dt>Build tools (from source)</dt>
          <dd>A C++20 compiler (GCC 13 from conda-forge), CMake ≥ 3.24, scikit-build-core and nanobind (fetched by pip).</dd>
          <dt>LAPACK, BLAS</dt>
          <dd>From conda-forge, in the active environment; the build looks nowhere else, so you always know which libraries you
            use. The core has no FFT library of its own: CPU transforms are NumPy/SciPy, GPU transforms cuFFT.</dd>
          <dt>GPU (optional)</dt>
          <dd>An NVIDIA GPU; to build from source, the CUDA 12 toolkit (nvcc, cuFFT, cuBLAS) from conda-forge.</dd>
        </dl>
      </Section>

      <Section id="pip" title="Install with pip">
        <Tabs label="Wheel" tabs={[
          { id: "pip-cpu", label: "CPU", body: <Code lang="bash">{PIP_CPU}</Code> },
          { id: "pip-gpu", label: "GPU (CUDA 12, beta)", body: <Code lang="bash">{PIP_GPU}</Code> },
        ]} />
      </Section>

      <Section id="install" title="Install from source">
        <Tabs label="Build variant" tabs={[
          { id: "cpu", label: "CPU", body: cpuSteps },
          { id: "gpu", label: "GPU (CUDA)", body: gpuSteps },
        ]} />
      </Section>

      <Section id="extras" title="Optional extras" intro={<p>Extras add optional Python packages; they are combined as usual,
        e.g. <code>pip install "conatus[tutorials,notebook]"</code>.</p>}>
        <div className="table-wrap">
          <table className="data">
            <thead><tr><th scope="col">Extra</th><th scope="col">Adds</th><th scope="col">For</th></tr></thead>
            <tbody>
              {extras.map(([k, pkgs, what]) => (
                <tr key={k}><td><code>{k}</code></td><td>{pkgs}</td><td>{what}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section id="check" title="Check your installation">
        <div className="info-cols">
          <div>
            <Code lang="bash">{CHECK}</Code>
          </div>
          <div className="prose">
            <p><code>doctor</code> tells you what this machine can run and how to fix what is missing. Conatus always says which
              compiled core it loaded; it never switches to another copy behind your back.</p>
            <p>An installed package writes nothing inside its environment: live-monitoring data go to{" "}
              <code>~/.cache/conatus</code> (or <code>$XDG_CACHE_HOME</code>, <code>CN_CACHE</code>, <code>CN_TMP</code>), and{" "}
              <code>doctor</code> prints these directories.</p>
            <p>Then start the <Link to="/tutorials">tutorials</Link> with tutorial 0, getting started.</p>
          </div>
        </div>
      </Section>

    </>
  );
}
