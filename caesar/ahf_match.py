"""AHF-specific galaxy–subhalo matching interface.

This module provides a dedicated entry point for the AHF path so that
changes to AHF matching logic are isolated from the FOF/SNAP and
AHF-FAST pipelines.
"""

from caesar.halo_matching import integrate_ahf_match_prune_inplace

__all__ = ["integrate_ahf_match_prune_inplace"]

