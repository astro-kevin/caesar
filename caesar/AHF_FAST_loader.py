"""Helpers for loading AHF particle catalogues for the AHF-FAST path.

This module centralises the low-level logic for reading an
``AHF_particles`` file and mapping its particle IDs onto the snapshot
particle IDs used by CAESAR.  The goal is to provide halo ID arrays that
are aligned with the CAESAR selection / concatenated index space, so
that higher-level code (e.g. :mod:`AHF_FAST`, :mod:`fof6d`) does not
need to worry about file format details.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from caesar.property_manager import get_property, has_ptype
from caesar.halo_matching import (
    _read_ahf_hierarchy,
    _iter_memberships_stream,
    ParticleMembership,
)
from caesar.ahf_match import _PidLookup  # uses the same efficient mapper as AHF


@dataclass
class AHFHierarchy:
    """Container for AHF hierarchy metadata."""

    parent_of: Dict[int, int]
    children_of: Dict[int, List[int]]


def load_ahf_hierarchy(ahf_particles_file: str) -> AHFHierarchy:
    """Return AHF hierarchy for a given ``AHF_particles`` file."""

    parent_of, children_of = _read_ahf_hierarchy(ahf_particles_file)
    return AHFHierarchy(parent_of=parent_of, children_of=children_of)


def load_ahf_halos_dataframe(ahf_particles_file: str):
    """Load ``AHF_halos`` into a compact DataFrame.

    The halos catalog is inferred from ``ahf_particles_file`` by replacing the
    basename token ``particles`` -> ``halos`` in the same directory.
    Returns columns: ``hid``, ``host_hid``, ``npart``.
    """

    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "AHF-FAST requires pandas to load AHF_halos as a DataFrame."
        ) from exc

    particles_base = os.path.basename(ahf_particles_file)
    if "particles" not in particles_base:
        raise ValueError(
            f"AHF particles filename does not contain 'particles': {ahf_particles_file}"
        )

    halos_file = os.path.join(
        os.path.dirname(ahf_particles_file),
        particles_base.replace("particles", "halos"),
    )
    if not os.path.isfile(halos_file):
        raise FileNotFoundError(f"AHF halos file not found: {halos_file}")

    df = pd.read_csv(
        halos_file,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 1, 4],
        names=["hid", "host_hid", "npart"],
        dtype=np.int64,
    )
    return df


def load_ahf_particle_blocks(
    ahf_particles_file: str,
    *,
    needed_nodes: Optional[Iterable[int]] = None,
    load_dm: bool = True,
) -> Dict[int, np.ndarray]:
    """Load raw AHF particle memberships into a haloID -> array dict.

    Each value is an N×2 array per halo with columns [pid, ptype_code].
    The input file may contain MPI chunk headers and multiple halo
    blocks; this routine collapses them into a single array per haloID.
    """

    # Determine which nodes exist from the hierarchy file
    parent_of, _ = _read_ahf_hierarchy(ahf_particles_file)
    all_nodes = set(parent_of.keys())
    if needed_nodes is None:
        needed: set[int] = all_nodes
    else:
        needed = {int(h) for h in needed_nodes}
        needed.intersection_update(all_nodes)

    out: Dict[int, List[np.ndarray]] = {}

    for pm in _iter_memberships_stream(ahf_particles_file, needed, load_dm=load_dm):
        rows: List[Tuple[int, int]] = []
        hid = int(pm.id)
        # gas (ptype 0)
        rows.extend((int(pid), 0) for pid in pm.parttype0)
        # dm / low-res variants (ptype 1,2,3) when requested
        if load_dm:
            rows.extend((int(pid), 1) for pid in getattr(pm, "parttype1", set()))
            rows.extend((int(pid), 2) for pid in getattr(pm, "parttype2", set()))
            rows.extend((int(pid), 3) for pid in getattr(pm, "parttype3", set()))
        # stars (ptype 4)
        rows.extend((int(pid), 4) for pid in pm.parttype4)
        # black holes (ptype 5)
        rows.extend((int(pid), 5) for pid in pm.parttype5)

        if not rows:
            continue
        arr = np.asarray(rows, dtype=np.int64)
        out.setdefault(hid, []).append(arr)

    # Collapse multiple blocks per halo into a single 2-column array
    final: Dict[int, np.ndarray] = {}
    for hid, blocks in out.items():
        if not blocks:
            continue
        if len(blocks) == 1:
            final[hid] = blocks[0]
        else:
            final[hid] = np.vstack(blocks)

    return final
