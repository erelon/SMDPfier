"""SMDPfier: SMDP-level behavior for Gymnasium environments.

This package provides the SMDPfier wrapper that enables Semi-Markov Decision Process
(SMDP) behavior by allowing users to choose Options (chains of primitive actions)
while attaching wall-time durations (in ticks) per option or per action.
"""

from .errors import SMDPOptionExecutionError, SMDPOptionValidationError
from .option import ListOption, Option, make_option_id, normalize_act_output
from .wrapper import SMDPfier

__version__ = "0.2.0"
__all__ = [
    "SMDPfier",
    "Option",
    "ListOption",
    "make_option_id",
    "normalize_act_output",
    "SMDPOptionExecutionError",
    "SMDPOptionValidationError",
]
