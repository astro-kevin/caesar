"""AHF-FAST-specific galaxy construction entry point.

This module holds the full implementation of the AHF-FAST galaxy builder
so that the fast path can evolve independently of the generic matching
utilities in :mod:`caesar.halo_matching`.  The :mod:`halo_matching`
module re-exports a thin wrapper for backwards compatibility, but the
source of truth lives here.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


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
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED

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

    def map_sel(pidset: Set[int], key: str) -> np.ndarray:
        lookup = pid_maps_sel.get(key)
        if lookup is None or not pidset:
            return np.empty(0, dtype=np.int32)
        mapped = lookup.map(pidset)
        if mapped.size == 0:
            return mapped
        return np.unique(mapped)

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

    skipped_empty_payloads = 0

    def process_host(order_idx: int, root_id: int, bucket: Dict[int, ParticleMembership]):
        nonlocal skipped_empty_payloads
        nodes_for_host = host_to_nodes.get(root_id, set())
        if not nodes_for_host:
            return order_idx, []

        # Ensure every node has a membership object (possibly empty)
        for node in nodes_for_host:
            bucket.setdefault(node, ParticleMembership(node))

        exclusives = _compute_exclusive_memberships(bucket, children_of, nodes_for_host)

        from collections import defaultdict as _dd

        payloads: List[Tuple[int, Set[int], Set[int], Set[int], Set[int]]] = []

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

        carry_star = _dd(set)
        carry_gas = _dd(set)
        carry_bh = _dd(set)
        carry_dm = _dd(set)

        nodes_sorted = sorted(nodes_for_host, key=node_depth, reverse=True)

        for node in nodes_sorted:
            extras_star = carry_star.pop(node, set())
            extras_gas = carry_gas.pop(node, set())
            extras_bh = carry_bh.pop(node, set())
            extras_dm = carry_dm.pop(node, set())

            ex = exclusives.get(node, ParticleMembership(node))
            star_set = set(ex.parttype4) | extras_star
            gas_set = set(ex.parttype0) | extras_gas
            bh_set = set(ex.parttype5) | extras_bh
            dm_exc_set = set(ex.parttype1) | extras_dm

            if len(star_set) >= min_stars:
                payloads.append((node, star_set, gas_set, bh_set, dm_exc_set))
            else:
                parent = parent_of.get(node, 0)
                if parent not in (0, None):
                    carry_star[parent].update(star_set)
                    carry_gas[parent].update(gas_set)
                    carry_bh[parent].update(bh_set)
                    carry_dm[parent].update(dm_exc_set)

        if not payloads:
            return order_idx, []

        host_galaxies: List[Tuple[int, object]] = []

        for payload in payloads:
            node_id, star_set, gas_set, bh_set, dm_exc = payload
            dm_pm = bucket.get(node_id)
            dm_inclusive = dm_pm.parttype1 if dm_pm is not None else set()
            grp = create_new_group(sim, "galaxy")
            grp.AHF_haloID = int(node_id)
            grp.slist = map_sel(star_set, "star")
            grp.glist = map_sel(gas_set, "gas")
            _apply_dense_gas_gate(grp)
            if "bh" in pid_maps_sel:
                grp.bhlist = map_sel(bh_set, "bh")
            if "dm" in pid_maps_sel:
                dm_selected = map_sel(dm_inclusive, "dm")
            else:
                dm_selected = np.array([], dtype=np.int32)
            grp.dmlist = dm_selected
            grp.global_indexes = np.array([], dtype=np.int64)
            if dm_exc:
                grp.__dict__["_dm_exclusive_pids"] = set(dm_exc)
            else:
                grp.__dict__["_dm_exclusive_pids"] = set()

            mapped_star = len(grp.slist) if hasattr(grp, "slist") else 0
            mapped_gas = len(grp.glist) if hasattr(grp, "glist") else 0
            mapped_bh = len(grp.bhlist) if hasattr(grp, "bhlist") else 0
            mapped_dm = len(dm_selected)
            particle_total = mapped_star + mapped_gas + mapped_bh + mapped_dm
            if particle_total == 0:
                skipped_empty_payloads += 1
                if skipped_empty_payloads <= 10:
                    mylog.warning(
                        "AHF-FAST: node %d had %d star / %d gas / %d bh / %d dm particles "
                        "from AHF but none mapped into CAESAR selection",
                        node_id,
                        len(star_set),
                        len(gas_set),
                        len(bh_set),
                        len(dm_inclusive),
                    )
                continue
            host_galaxies.append((int(node_id), grp))

        return order_idx, host_galaxies

    pending_results: Dict[int, List] = {}
    next_to_emit = 0

    def flush_completed(futures, block: bool = False):
        nonlocal next_to_emit
        if not futures:
            return
        timeout = None if block else 0
        return_when = ALL_COMPLETED if block else FIRST_COMPLETED
        done, _ = wait(list(futures.keys()), timeout=timeout, return_when=return_when)
        if not done:
            return
        for fut in done:
            order_idx, host_gals = fut.result()
            futures.pop(fut, None)
            pending_results[order_idx] = host_gals
        while next_to_emit in pending_results:
            host_gals = pending_results.pop(next_to_emit)
            if host_gals:
                for node_id, grp in host_gals:
                    galaxies.append(grp)
                    galaxy_node_ids.append(int(node_id))
            if host_progress is not None:
                host_progress.update(1)
            next_to_emit += 1

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        pending_futures: Dict = {}
        host_order = 0

        for root_id, nodes_for_host in host_to_nodes.items():
            # Build a membership bucket for this host from the loader arrays
            bucket: Dict[int, ParticleMembership] = {}
            if membership_arrays:
                for node_id in nodes_for_host:
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
                    pm = ParticleMembership(int(node_id))
                    mask0 = ptypes == 0
                    if np.any(mask0):
                        pm.parttype0 = set(int(v) for v in pids[mask0])
                    if load_dm:
                        mask1 = ptypes == 1
                        if np.any(mask1):
                            pm.parttype1 = set(int(v) for v in pids[mask1])
                        mask2 = ptypes == 2
                        if np.any(mask2):
                            pm.parttype2 = set(int(v) for v in pids[mask2])
                        mask3 = ptypes == 3
                        if np.any(mask3):
                            pm.parttype3 = set(int(v) for v in pids[mask3])
                    mask4 = ptypes == 4
                    if np.any(mask4):
                        pm.parttype4 = set(int(v) for v in pids[mask4])
                    mask5 = ptypes == 5
                    if np.any(mask5):
                        pm.parttype5 = set(int(v) for v in pids[mask5])
                    bucket[int(node_id)] = pm

            bucket_copy = dict(bucket)
            future = executor.submit(process_host, host_order, root_id, bucket_copy)
            pending_futures[future] = host_order
            host_order += 1
            flush_completed(pending_futures, block=False)

        flush_completed(pending_futures, block=True)

    if host_progress is not None:
        host_progress.close()

    if skipped_empty_payloads > 0:
        mylog.warning(
            "AHF-FAST: skipped %d galaxy payload(s) with no mapped particles"
            % skipped_empty_payloads
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
            sample = list(sorted(missing_hosts))[:5]
            if debug_fast:
                mylog.info(
                    "AHF-FAST debug: missing_hosts=%d example=%s",
                    len(missing_hosts),
                    sample,
                )
            # Invariants for the AHF-FAST path: every galaxy host AHF ID
            # must correspond to an existing CAESAR halo built from the
            # top-level hosts.  If we ever violate this, we want an
            # immediate, loud failure rather than silently synthesizing
            # halos from the particles file.
            raise AssertionError(
                f"AHF-FAST invariant violated: {len(missing_hosts)} galaxy host halo(s) "
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
            if not dm_exc:
                continue
            mapped = dm_full_lookup.map(dm_exc)
            if mapped.size == 0:
                try:
                    del gal.__dict__["_dm_exclusive_pids"]
                except KeyError:
                    pass
                continue
            exclusive_gal_dm[mapped] = int(getattr(gal, "GroupID", -1))
            # drop the cached set to free memory
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
