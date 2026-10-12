"""Data-driven decompositions: POD, DMD, SPOD, EDMD/Koopman.

Every method takes a *declared* inner product (``DataInnerProduct``: grid quadrature, exact Gram
matrix, or Euclidean stated as such) and declared estimation choices (rank, mean, window, blocks,
dictionary...): nothing is picked for you. Arrays may be NumPy or CuPy (the heavy products run on
the data's device; the small reduced problems on the host). Snapshots come from arrays or from
runs of the registry (``snapshots_from_run``).

* ``pod``: method of snapshots or SVD (Sirovich 1987);
* ``dmd``: exact DMD with amplitudes ('first' or 'optimal') and ResDMD residuals; optimized DMD
  by variable projection (Askham & Kutz 2018);
* ``spod``: Welch blocks, window, overlap (Towne, Schmidt & Colonius 2018), χ² confidence;
* ``edmd``: Koopman eigenpairs on a declared dictionary (Williams et al. 2015), ResDMD residuals;
"""
from .dmd import AMPLITUDES, DMD, DMD_METHODS, dmd
from .koopman import EDMD, Dictionary, edmd, monomials, rbf
from .norms import DataInnerProduct
from .pod import MEAN_POLICIES, POD, POD_METHODS, pod
from .sources import Snapshots, snapshots_from_arrays, snapshots_from_run, uniform_dt
from .spod import SPOD, spod, welch_blocks

__all__ = ["AMPLITUDES", "DMD", "DMD_METHODS", "DataInnerProduct", "Dictionary", "EDMD",
           "MEAN_POLICIES", "POD", "POD_METHODS", "SPOD", "Snapshots", "dmd",
           "edmd", "monomials", "pod", "rbf", "snapshots_from_arrays", "snapshots_from_run", "spod",
           "uniform_dt", "welch_blocks"]
