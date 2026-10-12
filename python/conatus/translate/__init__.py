"""Assisted translation of scripts written for other codes into Conatus (v0.1 UX).

``from_dedalus(script)``: a Dedalus 3 script -> a Conatus Problem, the declared time stepping,
initial data, and a line-by-line report (translated / approximated and why / unsupported / skipped).
The script is read with ``ast`` and interpreted by a restricted evaluator: nothing is imported
from Dedalus and no statement of the script is executed by Python itself.
"""
from .dedalus import DedalusTranslation, ReportLine, from_dedalus

__all__ = ["from_dedalus", "DedalusTranslation", "ReportLine"]
