# Mathematics

Short notes on the methods behind Conatus: what is computed, in which form, and why. They are
written for a reader who uses the library and wants to know what happens underneath; each one ends
with the references it follows.

The notes describe the discrete problem Conatus actually solves. Where the native layer makes a
choice for you (a tau placement, a gauge, the implicit/explicit split), the note says which rule
it applies; `explain()` prints the same decisions for your own model.

```{toctree}
:maxdepth: 1

spectral
curvilinear
timestepping
eigenproblems
invariant_solutions
continuation
floquet
nonmodal
chaos
symmetry
```
