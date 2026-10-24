# Citing Conatus

If Conatus contributes to your work, please cite the software. The repository carries a
`CITATION.cff` file (GitHub shows it as "Cite this repository"); the same entry in BibTeX:

```bibtex
@software{dahan2026conatus,
  author  = {Dahan, Ruben},
  title   = {{Conatus: spectral PDEs and dynamical systems}},
  year    = {2026},
  version = {0.2.1},
  license = {BSD-3-Clause},
  url     = {https://github.com/rubendahan/conatus},
}
```

A paper describing the design and the validation tests is in preparation.

## Cite the methods too

Conatus implements published methods; please cite them along with Conatus. `plan.methods()`
drafts a *Methods* paragraph for your run and returns the BibTeX entries of the methods it cites
(for example Olver & Townsend 2013 for the
ultraspherical operators, Burns et al. 2020 for the generalised tau method, Vasil et al. 2016 and
2019 for polar and spherical bases, Ascher, Ruuth & Spiteri 1997 for the IMEX Runge–Kutta
schemes). See {doc}`guide/explain`.

## Validation references

The published values reproduced by the tutorials and the gallery are listed on each page with
their source. When you use Conatus to reproduce or extend one of them, cite the original work.
