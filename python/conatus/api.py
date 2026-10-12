"""Deprecated: the expert layer is now ``conatus.expert``.

``conatus.api`` keeps working for existing scripts, with a FutureWarning at import:

* every name resolves to the same object as ``conatus.expert.<name>`` (module ``__getattr__``);
* two names keep their former meaning here only, so that no old script changes behaviour:
  ``sensitivity`` is the dense-EVP helper ``conatus.solvers.sensitivity`` (in ``conatus.expert``
  the name is the ``conatus.sensitivity``) and ``Channel`` is the base-flow record
  ``conatus.flows.ChannelBaseFlow`` (``cn.Channel`` is the native geometry).

Replace ``conatus.api`` by ``conatus.expert``; for these two names, see above.
"""
from __future__ import annotations

import warnings

from . import expert as _expert
from .flows import ChannelBaseFlow as Channel  # noqa: F401 -- former meaning (docstring)
from .solvers.evp import sensitivity  # noqa: F401 -- former meaning (docstring)

warnings.warn("conatus.api is deprecated: import the expert layer as conatus.expert (same names; "
              "'sensitivity' and 'Channel' differ, see help(conatus.api)); the user layer is "
              "`import conatus as cn`", FutureWarning, stacklevel=2)

__all__ = sorted(set(_expert.__all__) | {"sensitivity", "Channel"})


def __getattr__(name: str):
    """Every other name is ``conatus.expert.<name>`` (the same object)."""
    return getattr(_expert, name)


def __dir__() -> list[str]:
    return sorted(set(dir(_expert)) | {"sensitivity", "Channel"})
