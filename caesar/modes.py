from __future__ import annotations

from enum import Enum
from typing import Any, Dict


class Mode(Enum):
    """Enumeration of high-level CAESAR member_search modes.

    This is a lightweight abstraction over the various ways halos and
    galaxies can be identified.  It is intentionally coarse so that the
    rest of the code can branch on a small, stable set of options instead
    of string literals spread across the codebase.
    """

    FOF_SNAP = "fof_snap"
    AHF = "ahf"
    AHF_FAST = "ahf_fast"
    AHF_SUBHALO = "ahf_subhalo"


def resolve_mode(kwargs: Dict[str, Any]) -> Mode:
    """Resolve a member_search mode from keyword arguments.

    Parameters
    ----------
    kwargs : dict
        The keyword argument dictionary passed into ``CAESAR.member_search``.

    Returns
    -------
    Mode
        The inferred high-level mode.
    """

    haloid = kwargs.get("haloid")
    haloid_file = kwargs.get("haloid_file")
    if isinstance(haloid, str):
        flag = haloid.strip().upper()
        # AHF modes require an accompanying particles file; if it is
        # missing we quietly fall back to the standard pipeline.
        if haloid_file:
            if flag == "AHF":
                return Mode.AHF
            if flag == "AHF-FAST":
                return Mode.AHF_FAST
            if flag == "AHF-SUBHALO":
                return Mode.AHF_SUBHALO

    # For 'fof', 'snap', None, or anything else we fall back to the
    # standard FOF/SNAP-driven pipeline.
    return Mode.FOF_SNAP
