"""AHF-FAST-specific galaxy construction entry point.

This module holds the full implementation of the AHF-FAST galaxy builder
so that the fast path can evolve independently of the generic matching
utilities in :mod:`caesar.halo_matching`.  The :mod:`halo_matching`
module re-exports a thin wrapper for backwards compatibility, but the
source of truth lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


@dataclass
class HostProcessingContext:
    """Immutable context for processing a single AHF host.

    This dataclass bundles all the read-only parameters needed by
    `process_single_host()` so they can be passed as a single object
    and easily serialized for parallel processing with loky.
    """
    # Simulation and data access
    sim: Any  # CAESAR simulation object (read-only during processing)
    pid_maps_sel: Dict[str, Any]  # PID -> index lookup tables per particle type

    # AHF hierarchy
    parent_of: Dict[int, int]
    children_of: Dict[int, List[int]]
    host_to_nodes: Dict[int, Set[int]]

    # Configuration
    min_stars: int
    use_fof: bool
    jobs: int  # Number of parallel jobs for internal FOF

    # FOF parameters (only used if use_fof=True)
    fof_LL: float = 0.0
    vel_LL: float = 0.0
    kerneltab: Optional[np.ndarray] = None
    Lbox: float = 0.0

    # Callable helpers (need to be passed since they're imported inside main func)
    create_new_group: Optional[Callable] = None
    get_property: Optional[Callable] = None
    compute_exclusive_memberships: Optional[Callable] = None
    run_fof6d_direct: Optional[Callable] = None


@dataclass
class HostProcessingResult:
    """Results from processing a single AHF host."""
    host_id: int
    order_idx: int
    galaxies: List[Tuple[int, Any]] = field(default_factory=list)  # (node_id, Group)
    skipped_empty_payloads: int = 0


def build_galaxies_from_ahf_fast(
    sim,
    ahf_particles_file: str,
    *,
    min_stars: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> None:
    """Build galaxies directly from AHF nodes using a star-count gate.

    Streaming host-by-host keeps memory bounded while joblib threads reuse
    the shared DataManager arrays for per-galaxy construction.
    """

    from yt.funcs import mylog
    import os as _os
    from tqdm import tqdm

    from caesar.group import create_new_group
    from caesar.group import get_group_properties as _get_group_properties
    from caesar.property_manager import get_property, has_ptype
    from caesar.AHF_FAST_loader import load_ahf_particle_blocks
    from caesar.halo_matching import (
        ParticleMembership,
        _PidLookup,
        _build_selected_pid_maps,
        _read_ahf_hierarchy,
        _group_nodes_by_root,
        _compute_exclusive_memberships,
        _ensure_missing_ahf_halos,
        _update_ahf_galaxy_maps,
        _prune_halos_after_galaxies,
    )

    # Resolve minimum-star threshold from a single source (kwarg -> default)
    from caesar.group import get_min_stars as _get_min_stars

    min_stars = _get_min_stars(sim, override=min_stars)

    # FOF integration: optionally run 6D FOF within each subhalo
    use_fof = _os.environ.get("CAESAR_AHF_FAST_USE_FOF", "1") == "1"
    if use_fof:
        from caesar.fubar import get_mean_interparticle_separation, get_b
        from caesar.fof6d import kernel_table, fof6d_main, fof6d_halo
        from caesar.property_manager import ptype_ints

        # Threshold for sharded FOF on central halos (configurable via environment variable)
        SHARDED_FOF_THRESHOLD = int(_os.environ.get("CAESAR_AHF_FAST_SHARDED_THRESHOLD", "1000"))

        # FOF parameters (same as regular AHF mode)
        MIS = get_mean_interparticle_separation(sim).d
        fof_LL = MIS * get_b(sim, 'galaxy')  # typically MIS * 0.02
        vel_LL = 1.0
        kerneltab = kernel_table(fof_LL)
        Lbox = sim.simulation.boxsize.d
        nHlim, Tlim = 0.13, 1e5
        mylog.info("AHF-FAST: FOF integration enabled (fof_LL=%.4f, nHlim=%.2f, Tlim=%.0f)", fof_LL, nHlim, Tlim)

        def run_fof6d_direct(pos, vel, minstars, box_size, ll, vel_ll, ktab):
            """Run 6D FOF directly without the sorting pre-pass that fragments small groups.

            The standard fof6d_halo() function sorts particles spatially and splits
            groups at gaps > fof_LL. This fragments small subhalos (barely meeting
            min_stars) into pieces too small to survive. By calling fof6d_main()
            directly with a single group, we skip this fragmentation.
            """
            npart = len(pos)
            if npart < minstars:
                return np.zeros(npart, dtype=np.int64) - 1, 0

            # Call fof6d_main directly with a single group containing all particles
            groups = [[0, npart]]
            result = fof6d_main(
                igrp=0,
                groups=groups,
                poslist=pos,    # [npart, ndim] format
                vellist=vel,
                kerneltab=ktab,
                t0=0.,
                Lbox=box_size,
                mingrp=minstars,
                fof_LL=ll,
                vel_LL=vel_ll,
            )

            n_galaxies = result[0]
            if n_galaxies > 0:
                galind = np.asarray(result[1], dtype=np.int64)
            else:
                galind = np.zeros(npart, dtype=np.int64) - 1
            return galind, n_galaxies

    else:
        mylog.info("AHF-FAST: FOF integration disabled (direct galaxy assignment)")
        fof_LL = vel_LL = kerneltab = Lbox = nHlim = Tlim = None

    # Optional debug controls for the FAST path
    debug_fast = _os.environ.get("CAESAR_AHF_FAST_DEBUG", "0") == "1"
    debug_host: Optional[int]
    try:
        _h = _os.environ.get("CAESAR_AHF_FAST_DEBUG_HOST")
        debug_host = int(_h) if _h not in (None, "") else None
    except Exception:
        debug_host = None

    pid_maps_sel = _build_selected_pid_maps(sim)
    dm_pid_lookup = pid_maps_sel.get("dm")
    dm_full_lookup: Optional[_PidLookup] = None
    ndm_full = 0
    if dm_pid_lookup is not None and has_ptype(sim, "dm"):
        dm_pids_full = get_property(sim, "pid", "dm").d
        dm_pids_full = np.asarray(dm_pids_full, dtype=np.int64)
        ndm_full = dm_pids_full.size
        if ndm_full > 0:
            dm_full_lookup = _PidLookup(dm_pids_full)
        del dm_pids_full

    # Prefer pre-loaded hierarchy/memberships (set by AHF_FAST.run) so that we
    # do not re-read the AHF_particles file. Fall back to on-demand loading if
    # they are not available (e.g. in tests).
    parent_of = getattr(sim, "_ahf_fast_parent_of", None)
    children_of = getattr(sim, "_ahf_fast_children_of", None)
    membership_arrays = getattr(sim, "_ahf_fast_memberships", None)

    if not parent_of:
        parent_of, children_of = _read_ahf_hierarchy(ahf_particles_file)
        membership_arrays = None

    if not parent_of:
        sim.galaxy_list = []
        sim.ngalaxies = 0
        return

    host_to_nodes, node_to_root = _group_nodes_by_root(parent_of)
    if not host_to_nodes:
        sim.galaxy_list = []
        sim.ngalaxies = 0
        return

    all_needed_nodes: Set[int] = set(node_to_root.keys())

    galaxies: List = []
    galaxy_node_ids: List[int] = []

    # Ensure we have memberships for all needed nodes.
    load_dm = dm_pid_lookup is not None
    if membership_arrays is None:
        membership_arrays = load_ahf_particle_blocks(
            ahf_particles_file,
            needed_nodes=all_needed_nodes,
            load_dm=load_dm,
        )

    jobs = n_jobs
    if jobs is None:
        jobs = getattr(sim, "nproc", 1)
    try:
        jobs = int(jobs)
    except Exception:
        jobs = 1
    jobs = max(1, jobs)

    show_progress = bool(getattr(sim, "_show_progress", True))
    total_hosts = len(host_to_nodes)
    host_progress = tqdm(
        total=total_hosts,
        desc="Building galaxies (AHF-FAST)",
        disable=(not show_progress or total_hosts == 0),
        leave=False,
    )

    def map_sel(pids: np.ndarray, key: str) -> np.ndarray:
        """Map particle IDs to indices using PID lookup tables.

        Args:
            pids: numpy array of particle IDs (int64)
            key: particle type key ("star", "gas", "bh", "dm")

        Returns:
            numpy array of unique mapped indices (int32)
        """
        lookup = pid_maps_sel.get(key)
        if lookup is None or pids.size == 0:
            return np.empty(0, dtype=np.int32)
        mapped = lookup.map(pids)  # _PidLookup.map() handles numpy arrays efficiently
        if mapped.size == 0:
            return mapped
        return np.unique(mapped)

    def get_dense_gas_indices(gas_indices: np.ndarray) -> np.ndarray:
        """Return gas indices that pass the dense gas gate (pre-FOF filtering).

        Gate: nH > nHlim and (T < Tlim or SFR > 0)
        Same criteria as standard FOF in fof6d.py setup_indexes().
        """
        if not use_fof or len(gas_indices) == 0:
            return gas_indices
        try:
            dm = sim.data_manager
            gnh = dm.gnh[gas_indices]
            gT = dm.gT[gas_indices]
            gsfr = dm.gsfr[gas_indices]
            mask = (gnh > nHlim) & ((gT < Tlim) | (gsfr > 0))
            return gas_indices[mask]
        except Exception:
            # If gas properties unavailable, return all gas
            return gas_indices

    def _apply_dense_gas_gate(gal) -> None:
        """Optionally restrict galaxy gas to dense/cool/SF gas before computing properties.

        Mirrors the gate used by 6D-FOF:
        nH > 0.13 and (T < 1e5 or SFR > 0).
        """
        try:
            gidx = getattr(gal, "glist", None)
            if gidx is None or len(gidx) == 0:
                return
            dm = getattr(sim, "data_manager", None)
            if dm is None:
                return
            gnh = getattr(dm, "gnh", None)
            gT = getattr(dm, "gT", None)
            gsfr = getattr(dm, "gsfr", None)
            if gnh is None or gT is None or gsfr is None:
                return
            gidx_arr = np.asarray(gidx, dtype=np.int64)
            nh = gnh[gidx_arr]
            temp = gT[gidx_arr]
            sfr = gsfr[gidx_arr]
            nHlim = 0.13
            Tlim = 1.0e5
            mask = (nh > nHlim) & ((temp < Tlim) | (sfr > 0))
            gal.glist = gidx_arr[mask]
        except Exception:
            # If anything goes wrong, leave glist unchanged.
            return

    # Import joblib for parallel FOF within each host
    if use_fof:
        from joblib import Parallel, delayed

    def process_host(order_idx: int, root_id: int, bucket: Dict[int, ParticleMembership]) -> HostProcessingResult:
        """Process a single host, running FOF in parallel across subhalos at each depth level.

        Returns:
            HostProcessingResult with galaxies and skipped payload count
        """
        local_skipped = 0
        nodes_for_host = host_to_nodes.get(root_id, set())
        if not nodes_for_host:
            return HostProcessingResult(host_id=root_id, order_idx=order_idx)

        # Ensure every node has a membership object (possibly empty)
        for node in nodes_for_host:
            bucket.setdefault(node, ParticleMembership(node))

        exclusives = _compute_exclusive_memberships(bucket, children_of, nodes_for_host)

        from collections import defaultdict as _dd

        # Check if host is large enough for FOF (4×min_stars threshold)
        # Host's membership is inclusive (includes all subhalo particles)
        host_pm = bucket.get(root_id, ParticleMembership(root_id))
        host_needs_fof = use_fof and len(host_pm.parttype4) >= 4 * min_stars

        depth_cache: Dict[int, int] = {}

        def node_depth(node: int) -> int:
            if node in depth_cache:
                return depth_cache[node]
            parent = parent_of.get(node, 0)
            if parent in (0, None):
                depth_cache[node] = 0
            else:
                depth_cache[node] = node_depth(int(parent)) + 1
            return depth_cache[node]

        # Carry dicts store numpy arrays of unclaimed PIDs to promote to parent
        carry_star: Dict[int, np.ndarray] = {}
        carry_gas: Dict[int, np.ndarray] = {}
        carry_bh: Dict[int, np.ndarray] = {}
        carry_dm: Dict[int, np.ndarray] = {}

        # Claimed particle tracking (PIDs that have been assigned to galaxies)
        # Use numpy arrays for O(n log n) set operations instead of Python sets
        claimed_star_pids: np.ndarray = np.array([], dtype=np.int64)
        claimed_gas_pids: np.ndarray = np.array([], dtype=np.int64)
        claimed_bh_pids: np.ndarray = np.array([], dtype=np.int64)

        host_galaxies: List[Tuple[int, object]] = []

        # Group nodes by depth for wave-based processing
        depth_groups: Dict[int, List[int]] = _dd(list)
        for node in nodes_for_host:
            depth_groups[node_depth(node)].append(node)

        # Helper function to run FOF for a single task (for joblib.Parallel)
        def _run_single_fof(task):
            """Run FOF on a single task and return results."""
            if task.get('use_sharded', False):
                # Use sharded FOF for very large central halos
                # fof6d_halo expects [ndim, npart] format
                pos_T = task['pos'].T
                vel_T = task['vel'].T
                fof_tags, n_galaxies = fof6d_halo(
                    nparthalo=len(task['pos']),
                    npart=len(task['pos']),
                    pos=pos_T,
                    vel=vel_T,
                    minstars=min_stars,
                    Lbox=Lbox,
                    fof_LL=fof_LL,
                    vel_LL=vel_LL,
                    kerneltab=kerneltab,
                )
            else:
                # Use direct FOF for subhalos (avoids fragmentation)
                fof_tags, n_galaxies = run_fof6d_direct(
                    pos=task['pos'],
                    vel=task['vel'],
                    minstars=min_stars,
                    box_size=Lbox,
                    ll=fof_LL,
                    vel_ll=vel_LL,
                    ktab=kerneltab,
                )
            return {
                'node': task['node'],
                'fof_tags': fof_tags,
                'n_galaxies': n_galaxies,
                'star_arr': task['star_arr'],
                'gas_arr': task['gas_arr'],
                'bh_arr': task['bh_arr'],
                'dm_exc_arr': task['dm_exc_arr'],
                'dm_inclusive': task['dm_inclusive'],
                'star_indices': task['star_indices'],
                'dense_gas_indices': task['dense_gas_indices'],
                'bh_indices': task['bh_indices'],
                'fof_gas_end': task['fof_gas_end'],
                'fof_star_end': task['fof_star_end'],
                'fof_bh_end': task['fof_bh_end'],
            }

        # Process depth levels from deepest to shallowest
        for depth in sorted(depth_groups.keys(), reverse=True):
            nodes_at_depth = depth_groups[depth]

            # Phase 1: Collect FOF tasks and non-FOF nodes for this depth level
            fof_tasks = []
            non_fof_nodes = []  # Nodes that skip FOF or have too few particles

            _empty_arr = np.array([], dtype=np.int64)
            for node in nodes_at_depth:
                # Pop carried particles from parent levels (numpy arrays)
                extras_star = carry_star.pop(node, _empty_arr)
                extras_gas = carry_gas.pop(node, _empty_arr)
                extras_bh = carry_bh.pop(node, _empty_arr)
                extras_dm = carry_dm.pop(node, _empty_arr)

                ex = exclusives.get(node)
                if ex is None:
                    # No exclusive data; use empty arrays
                    ex_pt4 = _empty_arr
                    ex_pt0 = _empty_arr
                    ex_pt5 = _empty_arr
                    ex_pt1 = _empty_arr
                else:
                    ex_pt4 = np.asarray(ex.parttype4, dtype=np.int64)
                    ex_pt0 = np.asarray(ex.parttype0, dtype=np.int64)
                    ex_pt5 = np.asarray(ex.parttype5, dtype=np.int64)
                    ex_pt1 = np.asarray(ex.parttype1, dtype=np.int64)

                # Combine exclusive + carried, then subtract claimed (all numpy ops)
                star_combined = np.union1d(ex_pt4, extras_star) if extras_star.size > 0 else ex_pt4
                star_arr = star_combined[~np.isin(star_combined, claimed_star_pids)] if claimed_star_pids.size > 0 else star_combined

                gas_combined = np.union1d(ex_pt0, extras_gas) if extras_gas.size > 0 else ex_pt0
                gas_arr = gas_combined[~np.isin(gas_combined, claimed_gas_pids)] if claimed_gas_pids.size > 0 else gas_combined

                bh_combined = np.union1d(ex_pt5, extras_bh) if extras_bh.size > 0 else ex_pt5
                bh_arr = bh_combined[~np.isin(bh_combined, claimed_bh_pids)] if claimed_bh_pids.size > 0 else bh_combined

                dm_exc_arr = np.union1d(ex_pt1, extras_dm) if extras_dm.size > 0 else ex_pt1

                # Get DM for later (not used in FOF)
                dm_pm = bucket.get(node)
                dm_inclusive = np.asarray(dm_pm.parttype1, dtype=np.int64) if dm_pm is not None else _empty_arr

                if star_arr.size < min_stars:
                    # Not enough stars, promote to parent
                    non_fof_nodes.append((node, star_arr, gas_arr, bh_arr, dm_exc_arr, dm_inclusive, 'too_few_stars'))
                    continue

                # Map PIDs to indices (map_sel now accepts numpy arrays)
                star_indices = map_sel(star_arr, "star")
                gas_indices = map_sel(gas_arr, "gas")
                bh_indices = map_sel(bh_arr, "bh") if "bh" in pid_maps_sel else np.array([], dtype=np.int32)

                if host_needs_fof and len(star_indices) >= min_stars:
                    # FOF path: prepare task (only for large hosts)
                    dense_gas_indices = get_dense_gas_indices(gas_indices)

                    # Combine eligible particles for FOF
                    fof_parts = [dense_gas_indices, star_indices]
                    if len(bh_indices) > 0:
                        fof_parts.append(bh_indices)
                    fof_indices = np.concatenate(fof_parts) if any(len(p) > 0 for p in fof_parts) else np.array([], dtype=np.int32)

                    if len(fof_indices) < min_stars:
                        non_fof_nodes.append((node, star_arr, gas_arr, bh_arr, dm_exc_arr, dm_inclusive, 'too_few_fof'))
                        continue

                    # Get positions and velocities
                    pos_parts = []
                    vel_parts = []

                    n_dense_gas = len(dense_gas_indices)
                    n_star = len(star_indices)
                    n_bh = len(bh_indices)

                    if n_dense_gas > 0:
                        pos_parts.append(get_property(sim, "pos", "gas").d[dense_gas_indices])
                        vel_parts.append(get_property(sim, "vel", "gas").d[dense_gas_indices])
                    if n_star > 0:
                        pos_parts.append(get_property(sim, "pos", "star").d[star_indices])
                        vel_parts.append(get_property(sim, "vel", "star").d[star_indices])
                    if n_bh > 0:
                        pos_parts.append(get_property(sim, "pos", "bh").d[bh_indices])
                        vel_parts.append(get_property(sim, "vel", "bh").d[bh_indices])

                    pos = np.concatenate(pos_parts) if pos_parts else np.empty((0, 3))
                    vel = np.concatenate(vel_parts) if vel_parts else np.empty((0, 3))

                    # Use sharded FOF for very large central halos (depth 0 with many stars)
                    use_sharded = (depth == 0 and len(star_indices) >= SHARDED_FOF_THRESHOLD)

                    fof_tasks.append({
                        'node': node,
                        'pos': pos,
                        'vel': vel,
                        'use_sharded': use_sharded,
                        'star_arr': star_arr,
                        'gas_arr': gas_arr,
                        'bh_arr': bh_arr,
                        'dm_exc_arr': dm_exc_arr,
                        'dm_inclusive': dm_inclusive,
                        'star_indices': star_indices,
                        'dense_gas_indices': dense_gas_indices,
                        'bh_indices': bh_indices,
                        'fof_gas_end': n_dense_gas,
                        'fof_star_end': n_dense_gas + n_star,
                        'fof_bh_end': n_dense_gas + n_star + n_bh,
                    })

                else:
                    # Non-FOF path: direct galaxy assignment
                    grp = create_new_group(sim, "galaxy")
                    grp.AHF_haloID = int(node)
                    grp.slist = star_indices
                    grp.glist = gas_indices
                    _apply_dense_gas_gate(grp)
                    grp.bhlist = bh_indices
                    dm_selected = map_sel(dm_inclusive, "dm") if "dm" in pid_maps_sel else np.array([], dtype=np.int32)
                    grp.dmlist = dm_selected
                    grp.global_indexes = np.array([], dtype=np.int64)
                    # Store DM exclusive PIDs as numpy array (converted to set later for map())
                    if dm_exc_arr.size > 0:
                        grp.__dict__["_dm_exclusive_pids"] = dm_exc_arr
                    else:
                        grp.__dict__["_dm_exclusive_pids"] = np.array([], dtype=np.int64)

                    mapped_star = len(grp.slist) if hasattr(grp, "slist") else 0
                    mapped_gas = len(grp.glist) if hasattr(grp, "glist") else 0
                    mapped_bh = len(grp.bhlist) if hasattr(grp, "bhlist") else 0
                    mapped_dm = len(dm_selected)
                    particle_total = mapped_star + mapped_gas + mapped_bh + mapped_dm
                    if particle_total == 0:
                        local_skipped += 1
                        if local_skipped <= 3:  # Limit warnings per host
                            mylog.warning(
                                "AHF-FAST: node %d had %d star / %d gas / %d bh / %d dm particles "
                                "from AHF but none mapped into CAESAR selection",
                                node, star_arr.size, gas_arr.size, bh_arr.size, dm_inclusive.size,
                            )
                        continue
                    host_galaxies.append((int(node), grp))

                    # Mark particles as claimed (non-FOF path) using numpy union
                    claimed_star_pids = np.union1d(claimed_star_pids, star_arr)
                    claimed_gas_pids = np.union1d(claimed_gas_pids, gas_arr)
                    claimed_bh_pids = np.union1d(claimed_bh_pids, bh_arr)

            # Phase 2: Run FOF in parallel for all tasks at this depth level
            if fof_tasks:
                # Use threading for parallelism (avoids pickle issues with sim)
                # KD-tree operations release GIL, so threading is effective
                results = Parallel(n_jobs=jobs, backend='threading')(
                    delayed(_run_single_fof)(task) for task in fof_tasks
                )

                # Phase 3: Process FOF results sequentially and update claimed state
                for result in results:
                    node = result['node']
                    fof_tags = result['fof_tags']
                    n_galaxies = result['n_galaxies']
                    star_arr = result['star_arr']
                    gas_arr = result['gas_arr']
                    bh_arr = result['bh_arr']
                    dm_exc_arr = result['dm_exc_arr']
                    dm_inclusive = result['dm_inclusive']
                    star_indices = result['star_indices']
                    dense_gas_indices = result['dense_gas_indices']
                    bh_indices = result['bh_indices']
                    fof_gas_end = result['fof_gas_end']
                    fof_star_end = result['fof_star_end']
                    fof_bh_end = result['fof_bh_end']

                    n_dense_gas = len(dense_gas_indices)
                    n_star = len(star_indices)
                    n_bh = len(bh_indices)

                    if n_galaxies == 0:
                        # No galaxies found, promote to parent
                        non_fof_nodes.append((node, star_arr, gas_arr, bh_arr, dm_exc_arr, dm_inclusive, 'no_galaxies'))
                        continue

                    # Create galaxies from FOF groups
                    for gal_id in range(n_galaxies):
                        gal_mask = fof_tags == gal_id

                        # Extract per-type indices using tracked slices
                        gal_gas_mask = gal_mask[:fof_gas_end]
                        gal_star_mask = gal_mask[fof_gas_end:fof_star_end]
                        gal_bh_mask = gal_mask[fof_star_end:fof_bh_end] if n_bh > 0 else np.array([], dtype=bool)

                        # Get the original per-type indices for particles in this galaxy
                        gal_gas = dense_gas_indices[gal_gas_mask] if n_dense_gas > 0 else np.array([], dtype=np.int32)
                        gal_star = star_indices[gal_star_mask] if n_star > 0 else np.array([], dtype=np.int32)
                        gal_bh = bh_indices[gal_bh_mask] if n_bh > 0 else np.array([], dtype=np.int32)

                        if len(gal_star) < min_stars:
                            continue

                        # Create galaxy
                        grp = create_new_group(sim, "galaxy")
                        grp.AHF_haloID = int(node)
                        grp.slist = gal_star
                        grp.glist = gal_gas
                        grp.bhlist = gal_bh if len(gal_bh) > 0 else np.array([], dtype=np.int32)
                        dm_selected = map_sel(dm_inclusive, "dm") if "dm" in pid_maps_sel else np.array([], dtype=np.int32)
                        grp.dmlist = dm_selected
                        grp.global_indexes = np.array([], dtype=np.int64)
                        # Store DM exclusive PIDs as numpy array
                        if dm_exc_arr.size > 0:
                            grp.__dict__["_dm_exclusive_pids"] = dm_exc_arr
                        else:
                            grp.__dict__["_dm_exclusive_pids"] = np.array([], dtype=np.int64)

                        host_galaxies.append((int(node), grp))

                    # Mark ALL particles from this node as claimed using numpy union
                    claimed_star_pids = np.union1d(claimed_star_pids, star_arr)
                    claimed_gas_pids = np.union1d(claimed_gas_pids, gas_arr)
                    claimed_bh_pids = np.union1d(claimed_bh_pids, bh_arr)

            # Phase 4: Handle nodes that skipped FOF or had no galaxies (promote to parent)
            for item in non_fof_nodes:
                node, star_arr, gas_arr, bh_arr, dm_exc_arr, dm_inclusive, reason = item
                parent = parent_of.get(node, 0)
                if parent not in (0, None):
                    # Merge numpy arrays into carry dicts
                    if parent in carry_star:
                        carry_star[parent] = np.union1d(carry_star[parent], star_arr)
                    else:
                        carry_star[parent] = star_arr
                    if parent in carry_gas:
                        carry_gas[parent] = np.union1d(carry_gas[parent], gas_arr)
                    else:
                        carry_gas[parent] = gas_arr
                    if parent in carry_bh:
                        carry_bh[parent] = np.union1d(carry_bh[parent], bh_arr)
                    else:
                        carry_bh[parent] = bh_arr
                    if parent in carry_dm:
                        carry_dm[parent] = np.union1d(carry_dm[parent], dm_exc_arr)
                    else:
                        carry_dm[parent] = dm_exc_arr

        return HostProcessingResult(
            host_id=root_id,
            order_idx=order_idx,
            galaxies=host_galaxies,
            skipped_empty_payloads=local_skipped,
        )

    # Helper to build membership bucket for a host
    def build_bucket(nodes: Set[int]) -> Dict[int, ParticleMembership]:
        bucket: Dict[int, ParticleMembership] = {}
        if not membership_arrays:
            return bucket
        for node_id in nodes:
            arr = membership_arrays.get(int(node_id))
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.int64)
            if arr.size == 0:
                continue
            if arr.ndim != 2 or arr.shape[1] != 2:
                arr = arr.reshape(-1, 2)
            pids = arr[:, 0]
            ptypes = arr[:, 1]
            # Keep as numpy arrays - no Python set conversion
            mask0 = ptypes == 0
            pt0 = pids[mask0] if np.any(mask0) else np.array([], dtype=np.int64)
            mask4 = ptypes == 4
            pt4 = pids[mask4] if np.any(mask4) else np.array([], dtype=np.int64)
            mask5 = ptypes == 5
            pt5 = pids[mask5] if np.any(mask5) else np.array([], dtype=np.int64)
            if load_dm:
                mask1 = ptypes == 1
                pt1 = pids[mask1] if np.any(mask1) else np.array([], dtype=np.int64)
                mask2 = ptypes == 2
                pt2 = pids[mask2] if np.any(mask2) else np.array([], dtype=np.int64)
                mask3 = ptypes == 3
                pt3 = pids[mask3] if np.any(mask3) else np.array([], dtype=np.int64)
            else:
                pt1 = np.array([], dtype=np.int64)
                pt2 = np.array([], dtype=np.int64)
                pt3 = np.array([], dtype=np.int64)
            pm = ParticleMembership(
                id=int(node_id),
                parttype0=pt0,
                parttype1=pt1,
                parttype2=pt2,
                parttype3=pt3,
                parttype4=pt4,
                parttype5=pt5,
            )
            bucket[int(node_id)] = pm
        return bucket

    # Estimate host size from membership arrays (star particles only for threshold)
    def estimate_host_stars(nodes: Set[int]) -> int:
        total = 0
        for node_id in nodes:
            arr = membership_arrays.get(int(node_id))
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.int64)
            if arr.size == 0:
                continue
            if arr.ndim != 2:
                arr = arr.reshape(-1, 2)
            ptypes = arr[:, 1]
            total += np.sum(ptypes == 4)  # Count star particles (type 4)
        return total

    # Categorize hosts into small (parallel) and large (sequential)
    size_threshold = 4 * min_stars  # Same threshold used for FOF decision
    small_hosts: List[Tuple[int, Set[int]]] = []
    large_hosts: List[Tuple[int, Set[int]]] = []

    for root_id, nodes in host_to_nodes.items():
        if estimate_host_stars(nodes) < size_threshold:
            small_hosts.append((root_id, nodes))
        else:
            large_hosts.append((root_id, nodes))

    mylog.info(
        "AHF-FAST: categorized %d small hosts (parallel) and %d large hosts (sequential)",
        len(small_hosts), len(large_hosts)
    )

    total_skipped = 0
    host_order = 0

    # Process small hosts in parallel batches
    if small_hosts and use_fof:
        from joblib import Parallel, delayed

        def process_small_host(order_idx: int, root_id: int, nodes: Set[int]) -> HostProcessingResult:
            bucket = build_bucket(nodes)
            return process_host(order_idx, root_id, bucket)

        mylog.info("AHF-FAST: processing %d small hosts in parallel", len(small_hosts))
        small_results = Parallel(n_jobs=jobs, backend='threading')(
            delayed(process_small_host)(host_order + i, root_id, nodes)
            for i, (root_id, nodes) in enumerate(small_hosts)
        )

        for result in small_results:
            if result.galaxies:
                for node_id, grp in result.galaxies:
                    galaxies.append(grp)
                    galaxy_node_ids.append(int(node_id))
            total_skipped += result.skipped_empty_payloads

        host_order += len(small_hosts)
        if host_progress is not None:
            host_progress.update(len(small_hosts))

    elif small_hosts:
        # No FOF, process small hosts sequentially
        for root_id, nodes in small_hosts:
            bucket = build_bucket(nodes)
            result = process_host(host_order, root_id, bucket)
            if result.galaxies:
                for node_id, grp in result.galaxies:
                    galaxies.append(grp)
                    galaxy_node_ids.append(int(node_id))
            total_skipped += result.skipped_empty_payloads
            host_order += 1
            if host_progress is not None:
                host_progress.update(1)

    # Process large hosts sequentially (FOF parallelized within each)
    for root_id, nodes in large_hosts:
        bucket = build_bucket(nodes)
        result = process_host(host_order, root_id, bucket)
        if result.galaxies:
            for node_id, grp in result.galaxies:
                galaxies.append(grp)
                galaxy_node_ids.append(int(node_id))
        total_skipped += result.skipped_empty_payloads
        host_order += 1
        if host_progress is not None:
            host_progress.update(1)

    if host_progress is not None:
        host_progress.close()

    if total_skipped > 0:
        mylog.warning(
            "AHF-FAST: skipped %d galaxy payload(s) with no mapped particles"
            % total_skipped
        )

    sim.galaxy_list = galaxies
    sim.ngalaxies = len(galaxies)
    sim.galaxies = galaxies
    setattr(sim, "_ahf_matched", True)
    setattr(sim, "_include_dm_in_galaxies", True)
    if sim.ngalaxies == 0:
        sim._ahf_galaxy_ahf_ids = []
        _update_ahf_galaxy_maps(sim, [])
        sim._ahf_galaxy_hosts = []
        return

    def _compute_global_indexes(gal) -> np.ndarray:
        blocks = []
        try:
            if hasattr(gal, "glist") and gal.glist is not None and len(gal.glist) > 0:
                blocks.append(sim.data_manager.glist[gal.glist])
        except Exception:
            pass
        try:
            if hasattr(gal, "slist") and gal.slist is not None and len(gal.slist) > 0:
                blocks.append(sim.data_manager.slist[gal.slist])
        except Exception:
            pass
        try:
            if hasattr(gal, "dmlist") and gal.dmlist is not None and len(gal.dmlist) > 0 and has_ptype(sim, "dm"):
                blocks.append(sim.data_manager.dmlist[gal.dmlist])
        except Exception:
            pass
        try:
            if hasattr(gal, "bhlist") and gal.bhlist is not None and len(gal.bhlist) > 0:
                blocks.append(sim.data_manager.bhlist[gal.bhlist])
        except Exception:
            pass
        if blocks:
            return np.concatenate(blocks).astype(np.int64)
        return np.array([], dtype=np.int64)

    def _refresh_global_indexes(gal) -> None:
        gal.global_indexes = _compute_global_indexes(gal)

    for gal in sim.galaxy_list:
        _refresh_global_indexes(gal)

    host_indices: List[int] = []
    if galaxy_node_ids and len(galaxy_node_ids) == sim.ngalaxies:
        galaxy_node_ids = [int(nid) if nid is not None else -1 for nid in galaxy_node_ids]
        sim._ahf_galaxy_ahf_ids = list(galaxy_node_ids)
        _update_ahf_galaxy_maps(sim, sim._ahf_galaxy_ahf_ids)

        ahf_to_halo_index: Dict[int, int] = {}
        for halo_index, halo in enumerate(sim.halo_list):
            ahf_hid = getattr(halo, "AHF_haloID", None)
            if ahf_hid is None:
                continue
            try:
                ahf_to_halo_index[int(ahf_hid)] = halo_index
            except Exception:
                continue

        if debug_fast:
            mylog.info(
                "AHF-FAST debug: nhalos=%d, ngalaxies=%d",
                len(sim.halo_list),
                sim.ngalaxies,
            )
            if debug_host is not None:
                present = any(
                    int(getattr(h, "AHF_haloID", -1)) == debug_host
                    for h in sim.halo_list
                )
                mylog.info(
                    "AHF-FAST debug: host %d present in halo_list? %s",
                    debug_host,
                    present,
                )

        def _resolve_halo_index(node_id: int) -> int:
            cur = int(node_id)
            visited: Set[int] = set()
            while True:
                candidate = ahf_to_halo_index.get(cur)
                if candidate is not None:
                    return candidate
                parent = parent_of.get(cur, 0)
                if parent in (0, None):
                    break
                cur = int(parent)
                if cur in visited:
                    break
                visited.add(cur)
            return -1

        missing_hosts: Set[int] = set()
        preliminary_indices: List[int] = []
        for node_id in galaxy_node_ids:
            if node_id is None or node_id == -1:
                preliminary_indices.append(-1)
                continue
            resolved = _resolve_halo_index(int(node_id))
            if resolved < 0:
                missing_hosts.add(int(node_id))
            preliminary_indices.append(resolved)
        if missing_hosts:
            # Exclude stolen hosts (MPI boundary artifacts) from the error
            stolen = getattr(sim, '_ahf_stolen_hosts', set())
            genuine_missing = missing_hosts - stolen
            stolen_missing = missing_hosts & stolen

            if stolen_missing:
                mylog.warning(
                    "AHF-FAST: %d galaxy host(s) were stolen by other hosts (MPI artifacts). "
                    "Example IDs: %s",
                    len(stolen_missing),
                    list(sorted(stolen_missing))[:5]
                )
                # Remove orphan galaxies whose host was stolen (MPI artifacts)
                orphan_indices = [i for i, nid in enumerate(galaxy_node_ids)
                                  if nid in stolen_missing]
                if orphan_indices:
                    # Remove in reverse order to preserve indices
                    for i in sorted(orphan_indices, reverse=True):
                        del sim.galaxy_list[i]
                        del preliminary_indices[i]
                        del galaxy_node_ids[i]
                    sim.ngalaxies = len(sim.galaxy_list)
                    sim._ahf_galaxy_ahf_ids = list(galaxy_node_ids)
                    mylog.warning(
                        "AHF-FAST: Removed %d orphan galaxies whose hosts were stolen",
                        len(orphan_indices)
                    )

            if genuine_missing:
                sample = list(sorted(genuine_missing))[:5]
                if debug_fast:
                    mylog.info(
                        "AHF-FAST debug: missing_hosts=%d example=%s",
                        len(genuine_missing),
                        sample,
                    )
                # Invariants for the AHF-FAST path: every galaxy host AHF ID
                # must correspond to an existing CAESAR halo built from the
                # top-level hosts.  If we ever violate this, we want an
                # immediate, loud failure rather than silently synthesizing
                # halos from the particles file.
                raise AssertionError(
                    f"AHF-FAST invariant violated: {len(genuine_missing)} galaxy host halo(s) "
                    f"were not resolved; example IDs {sample}"
                )
        host_indices = preliminary_indices
    else:
        sim._ahf_galaxy_ahf_ids = []
        _update_ahf_galaxy_maps(sim, [])
        host_indices = [-1 for _ in range(sim.ngalaxies)]

    for halo in sim.halo_list:
        halo.galaxy_index_list = []

    for gi, host_idx in enumerate(host_indices):
        idx = int(host_idx) if host_idx is not None else -1
        sim.galaxy_list[gi].parent_halo_index = idx
        if idx >= 0 and idx < len(sim.halo_list):
            sim.halo_list[idx].galaxy_index_list.append(gi)

    sim._ahf_galaxy_hosts = [int(h) if h is not None else -1 for h in host_indices]

    _prune_halos_after_galaxies(sim)

    class _Ctx:
        def __init__(self, sim):
            self.obj = sim
            self.obj_type = "galaxy"
            self.nproc = getattr(sim, "nproc", 1)
            self.load_pot = getattr(sim, "load_pot", True)
            self.nparttot = sum(len(getattr(g, "global_indexes", [])) for g in sim.galaxy_list)
            mapping = {
                "gas": "glist",
                "star": "slist",
                "bh": "bhlist",
                "dm": "dmlist",
                "dm2": "dm2list",
                "dm3": "dm3list",
            }
            present = set(getattr(sim.data_manager, "ptypes", []))
            counts: Dict[str, int] = {}
            for p, attr in mapping.items():
                if present and p not in present:
                    continue
                data = getattr(sim.data_manager, attr, None)
                if data is None:
                    continue
                try:
                    counts[p] = len(data)
                except TypeError:
                    continue
            self.nparttype = counts
            self.counts = {"galaxy": len(sim.galaxy_list)}

    ctx = _Ctx(sim)
    prop_bar = None
    if show_progress and sim.ngalaxies > 0:
        prop_bar = tqdm(total=1, desc="Computing galaxy properties", leave=False)
    try:
        _get_group_properties(ctx, sim.galaxy_list)
    finally:
        if prop_bar is not None:
            prop_bar.update(1)
            prop_bar.close()

    try:
        if "galaxy" not in sim.group_types:
            sim.group_types.append("galaxy")
    except Exception:
        pass

    if ndm_full > 0 and dm_full_lookup is not None:
        exclusive_gal_dm = np.full(ndm_full, -1, dtype=np.int32)
        for gal in sim.galaxy_list:
            dm_exc = getattr(gal, "_dm_exclusive_pids", None)
            # dm_exc is now a numpy array; check for None and empty
            if dm_exc is None or (hasattr(dm_exc, 'size') and dm_exc.size == 0):
                continue
            mapped = dm_full_lookup.map(dm_exc)
            if mapped.size == 0:
                try:
                    del gal.__dict__["_dm_exclusive_pids"]
                except KeyError:
                    pass
                continue
            exclusive_gal_dm[mapped] = int(getattr(gal, "GroupID", -1))
            # drop the cached array to free memory
            try:
                del gal.__dict__["_dm_exclusive_pids"]
            except KeyError:
                pass
        setattr(sim, "_exclusive_galaxy_dmlist", exclusive_gal_dm)
    else:
        for gal in sim.galaxy_list:
            if "_dm_exclusive_pids" in gal.__dict__:
                del gal.__dict__["_dm_exclusive_pids"]
        if hasattr(sim, "_exclusive_galaxy_dmlist"):
            delattr(sim, "_exclusive_galaxy_dmlist")
