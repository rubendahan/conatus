# Symmetry subspaces

Computing in an invariant subspace of a symmetry group: fewer unknowns and isolated solutions, at the price of hiding what breaks the symmetry.

When the equations are equivariant under a group (reflections, shifts, rotations by $2\pi/m$), the
states with a given symmetry form an invariant subspace. Computing in it divides the size of the
problem and, for continuous symmetries, removes the neutral directions that make solutions
non-isolated.

Conatus uses symmetry subspaces in three places:

- **Invariant solutions.** `find_equilibrium` works in the odd subspace $u(-x) = -u(x)$ of a
  periodic problem, where translations are broken ({doc}`invariant_solutions`). Exact coherent
  states of 3D shear flows are searched in their shift–reflect and shift–rotate subspaces.
- **Bifurcations.** A symmetry given as a matrix (`branch_switch(..., symmetry=S)`,
  `normal_form(..., symmetry=S)`) classifies symmetry-breaking pitchforks ({doc}`continuation`).
- **Eigenvalue problems.** A linearized problem restricted to one parity class about a reflection,
  or to the azimuthal classes of an $m$-fold rotation in curvilinear geometry, has its own, smaller
  spectrum. Parities incompatible with the equations are refused (`SymmetryBreakingError`). These
  subspaces are declared on problems of the expert layer (tutorial
  {doc}`../../tutorials/03_symmetries`).

## What a subspace hides

An instability that breaks the symmetry does not exist inside the subspace. A solution found there
is not declared stable until its spectrum in the full space has been computed:
`equilibrium_eigenvalues` and `floquet_multipliers` work in the full space, and
`symmetry_breaking_check` reports the growth rates outside the subspace. `explain()` on a symmetry
subspace states what it hides.

**See also:** {doc}`SymmetrySubspace <../../api/_generated/_stubs/conatus.expert.SymmetrySubspace>`, {doc}`Reflection <../../api/_generated/_stubs/conatus.expert.Reflection>`,
{doc}`symmetry_breaking_check <../../api/_generated/_stubs/conatus.expert.symmetry_breaking_check>`.
