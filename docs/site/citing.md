# Citing Conatus

Conatus is described in a *living paper*, modelled on the founding Dedalus article (Burns et al.
2020): it details the design and **every** validation test, and each figure or number in it points
to an archived run. It is updated with the code; the version in the repository is the reference.

> R. Dahan, *Conatus: a GPU-native spectral framework for partial differential equations and
> the dynamical systems they define*, in preparation (living document: `paper/conatus.pdf` in the repository).

```bibtex
@unpublished{dahan_conatus,
  author = {Dahan, Ruben},
  title  = {Conatus: a {GPU}-native spectral framework for partial differential equations
            and the dynamical systems they define},
  note   = {In preparation. Living document distributed with the source code},
  year   = {2026}
}
```

The repository carries a `CITATION.cff` file; a software DOI will accompany the 0.1 release.

## Cite the methods too

Conatus implements published methods; please cite them along with Conatus. You do not need to
assemble the list by hand: `plan.methods()` writes a *Methods* paragraph for your run and returns
the BibTeX entries of exactly the methods it used (for example Olver & Townsend 2013 for the
ultraspherical operators, Burns et al. 2020 for the generalised tau method, Vasil et al. 2016 and
2019 for polar and spherical bases, Ascher, Ruuth & Spiteri 1997 for the IMEX Runge–Kutta
schemes). See {doc}`guide/explain`.

## Validation references

The published values reproduced by the tutorials and the gallery are listed on each page with
their source. When you use Conatus to reproduce or extend one of them, cite the original work.
