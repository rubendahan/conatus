# Dynamics

Everything Conatus does with a model once it is written: stability, growth, invariant solutions, their branches, their symmetries and their sensitivities.

The model you simulate is the model you analyse. A native model (`cn.Model`) gives its eigenvalue
problem with `m.eigenvalues(...)` or `m.linearize(about=...)`, its time stepper with
`m.simulate(...)` (the flow map that every search for an invariant solution uses), and its
compiled problem with `m.compile(N).problem` for the tools of `conatus.expert`. Every result
carries the residual that validates it and an `explain()` that lists the choices made.

| Page | Question it answers |
|---|---|
| {doc}`eigenvalues` | Is this state stable? Where does it become unstable? |
| {doc}`onset` | The critical Rayleigh number of convection and its mode, from a ready model or from the equations of a normal mode |
| {doc}`nonmodal` | How much can a stable flow amplify a perturbation or a forcing? |
| {doc}`invariant_solutions` | Which equilibria, travelling waves and periodic orbits exist? |
| {doc}`continuation` | How do they change with a parameter? Where are the folds and bifurcations? |
| {doc}`floquet` | Is a time-periodic flow stable? |
| {doc}`contour` | Which eigenvalues lie in this region of the complex plane, and are there any others? |
| {doc}`chaos` | How chaotic is the flow? Where do trajectories end up? |
| {doc}`manifolds` | How are invariant solutions connected? How to remove a continuous symmetry? |
| {doc}`symmetry` | How to compute in a symmetry subspace, and what that hides |
| {doc}`adjoint` | Gradients, adjoint modes, the wavemaker and base-flow sensitivity |
| {doc}`optimal_perturbations` | Which initial perturbation grows most, linearly or not (Cartesian and spherical shell)? |
| {doc}`data` | What do snapshots of a flow say? DMD and SPOD in the exact norm |

The snippets on these pages are executed as they are shown, at small resolutions so that they run
in seconds; the tutorials carry each method to a published value.

```{toctree}
:caption: Dynamics
:maxdepth: 1

eigenvalues
onset
nonmodal
invariant_solutions
continuation
floquet
contour
chaos
manifolds
symmetry
adjoint
optimal_perturbations
data
```
