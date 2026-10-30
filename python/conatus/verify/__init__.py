"""Verification tools: MMS, convergence studies and reports of measurements, thresholds and failed checks."""
from .report import Check, VerificationReport, verify

__all__ = [
    "Check", "VerificationReport", "verify",
]

# The package called is its main function: cn.verify(...) == conatus.expert.verify(...).
from .._callable import make_callable  # noqa: E402

make_callable(__name__, verify)
