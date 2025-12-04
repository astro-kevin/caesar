"""AHF-FAST-specific halo construction.

This module builds CAESAR halos from an AHF catalogue for the
``haloid='AHF-FAST'`` path.  It is intentionally separate from
``fof6d.load_ahf_id`` so that the FAST path can:

- Use the shared :mod:`AHF_FAST_loader` representation.
- Treat only top-level AHF hosts (hostID == 0) as halos.
- Leave the generic FOF/FOF-from-snapshot machinery untouched.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


def build_halos_from_ahf_fast(sim, ahf_particles_file: str):
    """Populate ``sim.halo_list`` from AHF for the AHF-FAST path.

    Halos are built from *top-level* AHF hosts (hostID == 0) by mapping
    their (and their subhalos') particle memberships onto the snapshot
    PIDs.  Galaxies are handled separately in :mod:`ahf_fast_match`.
    """

    from yt.funcs import mylog

    from caesar.fof6d import fof6d, _PidLookup
    from caesar.fubar import get_mean_interparticle_separation
    from caesar.group import get_group_properties
    from caesar.property_manager import get_property, has_ptype, ptype_ints
    from caesar.AHF_FAST_loader import load_ahf_hierarchy, load_ahf_particle_blocks
    from caesar.halo_matching import _update_ahf_halo_maps, _populate_hydrogen_masses

    if not ahf_particles_file:
        raise ValueError("AHF-FAST requires an AHF_particles file.")

    # Load hierarchy and cache it on the simulation for reuse by the
    # galaxy builder.
    hier = load_ahf_hierarchy(ahf_particles_file)
    parent_of = hier.parent_of
    children_of = hier.children_of
    if not parent_of:
        mylog.warning("AHF-FAST: empty AHF hierarchy; no halos will be built")
        sim.halo_list = []
        sim.halos = []
        sim.nhalos = 0
        return None

    # Cache for reuse by the galaxy builder
    sim._ahf_fast_parent_of = parent_of
    sim._ahf_fast_children_of = children_of

    # Build mapping root -> nodes using the same helper as the galaxy path.
    from caesar.halo_matching import _group_nodes_by_root

    host_to_nodes, node_to_root = _group_nodes_by_root(parent_of)
    if not host_to_nodes:
        mylog.warning("AHF-FAST: no host halos resolved from AHF hierarchy")
        sim.halo_list = []
        sim.halos = []
        sim.nhalos = 0
        return None

    # Load memberships for all nodes once.
    memberships = load_ahf_particle_blocks(
        ahf_particles_file,
        needed_nodes=parent_of.keys(),
        load_dm=True,
    )
    sim._ahf_fast_memberships = memberships

    # Aggregate memberships per host root.  Each host halo collects the
    # particles from all nodes in its tree (host + subhalos).
    host_blocks: Dict[int, List[np.ndarray]] = {}
    for node_id, arr in memberships.items():
        root = node_to_root.get(int(node_id))
        if root is None:
            continue
        arr = np.asarray(arr, dtype=np.int64)
        if arr.size == 0:
            continue
        if arr.ndim != 2 or arr.shape[1] != 2:
            arr = arr.reshape(-1, 2)
        host_blocks.setdefault(int(root), []).append(arr)

    # Instantiate fof6d helper for halos.
    halos = fof6d(sim, "halo")
    halos.MIS = get_mean_interparticle_separation(sim).d
    # For AHF-FAST we want to retain even low-mass halos; pruning of
    # DM-poor halos happens later (after galaxies are attached), so that
    # every galaxy host AHF ID can resolve to a CAESAR halo.
    halos.keep_all_groups = True

    # Build per-type PID lookups and haloID arrays.
    halos.haloid = {}
    lookup_map: Dict[int, Tuple[_PidLookup, np.ndarray]] = {}
    tmpp_by_ptype: Dict[str, np.ndarray] = {}

    for p in sim.data_manager.ptypes:
        if has_ptype(sim, p):
            data = get_property(sim, "pid", p).d.astype(np.int64)
            tmpp = np.full(len(data), -1, dtype=np.int64)
            halos.haloid[p] = tmpp
            lookup_map[ptype_ints[p]] = (_PidLookup(data), tmpp)
            tmpp_by_ptype[p] = tmpp
        else:
            tmpp = np.empty(0, dtype=np.int64)
            halos.haloid[p] = tmpp
            tmpp_by_ptype[p] = tmpp

    nhid = 0
    from caesar.utils import memlog

    memlog("AHF-FAST: mapping host+subhalo particle IDs to snapshot")

    for root_id, blocks in host_blocks.items():
        if not blocks:
            continue
        block = blocks[0] if len(blocks) == 1 else np.vstack(blocks)
        if block.size == 0:
            continue
        pid_vals = block[:, 0]
        type_vals = block[:, 1]
        ptypes_present, inv = np.unique(type_vals, return_inverse=True)
        hid_val = int(root_id)
        for code in ptypes_present:
            entry = lookup_map.get(int(code))
            if entry is None:
                continue
            lookup, tmpp = entry
            code_idx = np.where(ptypes_present == code)[0][0]
            mask = inv == code_idx
            if not np.any(mask):
                continue
            indices, matched_mask = lookup.search(pid_vals[mask])
            if indices.size == 0:
                continue
            tmpp[indices] = hid_val
            nhid += indices.size

    memlog("AHF-FAST: total halo particle IDs = %d" % nhid)

    # Initialise member search using the host-only haloid mapping.
    sim.data_manager._member_search_init(select=halos.haloid)

    # Flatten per-type haloid arrays into the concatenated index space so that
    # fof6d.plist_init() can group particles by halo.
    flattened: List[np.ndarray] = []
    from caesar.property_manager import has_ptype as _has_ptype

    for ptype in sim.data_manager.ptypes:
        source = halos.haloid.get(ptype)
        if source is None:
            continue
        source_arr = np.asarray(source, dtype=np.int64).reshape(-1)
        if source_arr.size == 0:
            continue
        mask = source_arr >= 0
        if mask.any():
            flattened.append(source_arr[mask])
    if flattened:
        sim.data_manager.haloid = np.concatenate(flattened).astype(np.int64, copy=False)
    else:
        sim.data_manager.haloid = np.empty(0, dtype=np.int64)

    if not halos.plist_init():
        return None

    halos.keep_all_groups = getattr(halos, "keep_all_groups", False)
    halos.load_lists()
    if hasattr(halos, "keep_all_groups"):
        delattr(halos, "keep_all_groups")
    if len(sim.halo_list) == 0:
        mylog.warning("AHF-FAST: no valid halos found; aborting member search")
        return None

    get_group_properties(halos, sim.halo_list)

    computed_hydrogen, halo_masses = _populate_hydrogen_masses(sim, sim.halo_list)

    if not computed_hydrogen:
        mylog.info("HI/H2 fractions unavailable; running hydrogen_mass_calc() for halos")
        import caesar.hydrogen_mass_calc as hydrogen_mass_calc

        hydrogen_mass_calc.hydrogen_mass_calc(sim)
        computed_hydrogen, halo_masses = _populate_hydrogen_masses(sim, sim.halo_list)
        if not computed_hydrogen:
            mylog.warning(
                "hydrogen_mass_calc() did not produce HI/H2 masses; setting to zero (check snapshot)"
            )
            halo_masses = {}

    setattr(sim, "_ahf_halo_hydrogen_masses", halo_masses)

    _update_ahf_halo_maps(sim)

    if "halo" not in sim.group_types:
        sim.group_types.append("halo")
    sim.halos = sim.halo_list
    sim.nhalos = len(sim.halo_list)

    return halos
