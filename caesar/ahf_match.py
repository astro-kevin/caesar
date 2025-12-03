"""AHF-specific galaxy–subhalo matching logic.

This module contains a self-contained implementation of the AHF matching
and pruning routine used by the ``haloid='AHF'`` path.  It is intended to
be isolated from the generic matching utilities in :mod:`halo_matching`
so that changes to the AHF path do not impact the FOF/SNAP or AHF-FAST
pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional, Dict, Iterable
import os
import gzip

import numpy as np

from bidict import bidict
from yt.funcs import mylog

from numba import njit, prange, set_num_threads
from numba.typed import List as NumbaList

from caesar.property_manager import get_property, has_ptype


def _compute_mass_quantity(sim, value: float):
    yt_dataset = getattr(sim, 'yt_dataset', None)
    units = getattr(sim, 'units', None)
    if yt_dataset is not None and units is not None and isinstance(units, dict):
        mass_unit = units.get('mass')
        if mass_unit is not None:
            try:
                return yt_dataset.quan(value, mass_unit)
            except Exception:
                pass
    return value


def _populate_hydrogen_masses(sim, halos: Iterable) -> Tuple[bool, Dict[int, Tuple[float, float]]]:
    dm = getattr(sim, 'data_manager', None)
    if dm is None:
        return False, {}

    if 'gas' not in getattr(dm, 'ptypes', []):
        return False, {}

    gas_index = getattr(dm, 'glist', None)
    mass_arr = getattr(dm, 'mass', None)
    gfHI_arr = getattr(dm, 'gfHI', None)
    gfH2_arr = getattr(dm, 'gfH2', None)

    if (
        gas_index is None
        or mass_arr is None
        or gfHI_arr is None
        or gfH2_arr is None
        or len(gas_index) == 0
    ):
        return False, {}

    gas_index = np.asarray(gas_index, dtype=np.int64)
    hydrogen_fraction = getattr(getattr(sim, 'simulation', None), 'XH', None)
    if hydrogen_fraction is None:
        hydrogen_fraction = 0.76

    def _values(array, indices):
        subset = array[indices]
        if hasattr(subset, 'd'):
            return np.asarray(subset.d, dtype=np.float64)
        return np.asarray(subset, dtype=np.float64)

    per_halo = {}
    for halo in halos:
        masses = getattr(halo, 'masses', None)
        if not isinstance(masses, dict):
            continue

        gl = getattr(halo, 'glist', None)
        if gl is None or len(gl) == 0:
            hi_mass = 0.0
            h2_mass = 0.0
        else:
            gas_sel = np.asarray(gl, dtype=np.int64)
            valid = (gas_sel >= 0) & (gas_sel < gas_index.size)
            gas_sel = gas_sel[valid]
            if gas_sel.size == 0:
                hi_mass = 0.0
                h2_mass = 0.0
                masses['HI'] = _compute_mass_quantity(sim, hi_mass)
                masses['H2'] = _compute_mass_quantity(sim, h2_mass)
                continue

            concat_idx = gas_index[gas_sel]
            valid_concat = (concat_idx >= 0) & (concat_idx < mass_arr.size)
            concat_idx = concat_idx[valid_concat]
            if concat_idx.size == 0:
                hi_mass = 0.0
                h2_mass = 0.0
                masses['HI'] = _compute_mass_quantity(sim, hi_mass)
                masses['H2'] = _compute_mass_quantity(sim, h2_mass)
                continue

            gas_mass = _values(mass_arr, concat_idx)
            hi_frac = _values(gfHI_arr, gas_sel)
            h2_frac = _values(gfH2_arr, gas_sel)
            hi_mass = float(np.sum(gas_mass * hi_frac) * hydrogen_fraction)
            h2_mass = float(np.sum(gas_mass * h2_frac) * hydrogen_fraction)

        masses['HI'] = _compute_mass_quantity(sim, hi_mass)
        masses['H2'] = _compute_mass_quantity(sim, h2_mass)
        ahf_id = getattr(halo, 'AHF_haloID', None)
        if ahf_id is not None:
            per_halo[int(ahf_id)] = (hi_mass, h2_mass)

    return True, per_halo


def _open_ahf_particles(path: str):
    """Open an AHF particles/halos text file, transparently handling gzip."""
    if path.endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path, 'r')


def _ahf_halos_path(ahf_particles_file: str) -> str:
    """Return matching AHF halos filename for a given particles file (supports .gz)."""
    if ahf_particles_file.endswith('particles.gz'):
        return ahf_particles_file.replace('particles.gz', 'halos')
    return ahf_particles_file.replace('particles', 'halos')


def _read_ahf_hierarchy(ahf_particles_file: str) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """Read AHF halos file and return parent and children mappings.

    Returns (parent_of, children_of) where parent_of[h] -> host_hid, and
    children_of[parent] -> list of child IDs.
    """
    halos_file = _ahf_halos_path(ahf_particles_file)
    if not os.path.isfile(halos_file):
        raise FileNotFoundError(f"AHF halos file not found: {halos_file}")

    # Load two integer columns: halo ID, host ID
    if halos_file.endswith('.gz'):
        with gzip.open(halos_file, 'rt') as f:
            data = np.loadtxt(f, usecols=(0, 1), dtype=np.int64)
    else:
        data = np.loadtxt(halos_file, usecols=(0, 1), dtype=np.int64)

    if data.ndim == 1 and data.size == 2:  # single halo edge case
        data = data.reshape((1, 2))

    parent_of: Dict[int, int] = {}
    children_of: Dict[int, List[int]] = {}
    for hid, host in data:
        parent_of[int(hid)] = int(host)
        children_of.setdefault(int(host), []).append(int(hid))
    # ensure every node has a children list entry
    for hid in parent_of.keys():
        children_of.setdefault(hid, [])
    return parent_of, children_of


class _PidLookup:
    """Memory-efficient PID -> index mapper backed by sorted NumPy arrays."""

    __slots__ = ('sorted_pids', 'index_order')

    def __init__(self, values: np.ndarray):
        arr = np.asarray(values, dtype=np.int64)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if arr.size == 0:
            self.sorted_pids = arr
            self.index_order = np.empty(0, dtype=np.int32)
            return
        order = np.argsort(arr, kind='mergesort')
        self.sorted_pids = arr[order]
        self.index_order = order.astype(np.int32, copy=False)

    @property
    def size(self) -> int:
        return int(self.sorted_pids.size)

    def map(self, pidset: Iterable[int]) -> np.ndarray:
        if isinstance(pidset, np.ndarray):
            arr = np.asarray(pidset, dtype=np.int64)
        else:
            length = len(pidset) if hasattr(pidset, '__len__') else -1
            if length == 0:
                return np.empty(0, dtype=np.int32)
            arr = np.fromiter((int(pid) for pid in pidset), dtype=np.int64, count=length)
        if arr.size == 0:
            return np.empty(0, dtype=np.int32)
        idx = np.searchsorted(self.sorted_pids, arr)
        mask = (idx < self.sorted_pids.size) & (self.sorted_pids[idx] == arr)
        if not mask.any():
            return np.empty(0, dtype=np.int32)
        return self.index_order[idx[mask]]


def _build_selected_pid_maps(sim) -> Dict[str, _PidLookup]:
    """Build PID -> selected-index maps for each ptype present in the CAESAR-selected subset."""

    maps: Dict[str, _PidLookup] = {}

    def add_map(ptype: str, list_name: str) -> None:
        if not has_ptype(sim, ptype):
            return
        sel = getattr(sim.data_manager, list_name, None)
        if sel is None or len(sel) == 0:
            return
        try:
            sel_arr = np.asarray(sel, dtype=np.intp)
            full_indices = np.asarray(sim.data_manager.indexes[sel_arr], dtype=np.int64)
            full_pids = get_property(sim, 'pid', ptype).d
            pid_array = np.asarray(full_pids, dtype=np.int64)
            selected = pid_array[full_indices]
            lookup = _PidLookup(selected)
            if lookup.size > 0:
                maps[ptype] = lookup
            del selected
            del pid_array
            del full_indices
        except Exception:
            return
        finally:
            try:
                del full_pids
            except NameError:
                pass

    add_map('star', 'slist')
    add_map('gas', 'glist')
    add_map('bh', 'bhlist')
    add_map('dm', 'dmlist')
    add_map('dm2', 'dm2list')
    add_map('dm3', 'dm3list')
    return maps


def _update_ahf_halo_maps(sim) -> None:
    """Cache halo AHF ID lookups for O(1) access."""

    halo_map = bidict()

    for idx, halo in enumerate(getattr(sim, 'halo_list', [])):
        hid_raw = getattr(halo, 'AHF_haloID', None)
        try:
            hid_val = int(hid_raw) if hid_raw is not None else None
        except Exception:
            hid_val = None
        if hid_val is not None and hid_val < 0:
            hid_val = None
        if hid_val is not None:
            halo_map[hid_val] = idx

    sim._ahf_halo_map = halo_map
    sim._ahf_halo_id_to_index = halo_map
    sim._ahf_halo_index_to_id = halo_map.inv


@njit
def _count_intersection_u64(a, b):
    """Return the size of the intersection of two sorted, unique uint64 arrays."""
    i = 0
    j = 0
    na = a.size
    nb = b.size
    count = 0
    while i < na and j < nb:
        av = a[i]
        bv = b[j]
        if av == bv:
            count += 1
            i += 1
            j += 1
        elif av < bv:
            i += 1
        else:
            j += 1
    return count


@njit(parallel=True)
def _numba_select_ahf_for_gals(
    gal_pid_lists,
    node_pid_lists,
    node_depth,
    node_ids,
    cand_node_lists,
    galaxy_star_counts,
):
    """Numba-parallel selection of AHF nodes per galaxy using typed lists."""
    ngal = len(gal_pid_lists)
    primaries = np.empty(ngal, dtype=np.int64)

    for gi in prange(ngal):
        gal_pids = gal_pid_lists[gi]
        if gal_pids.size == 0:
            primaries[gi] = -1
            continue

        total = int(galaxy_star_counts[gi])
        if total <= 0:
            primaries[gi] = -1
            continue
        thresh = 0.5 * total

        cand_idx = cand_node_lists[gi]
        if cand_idx.size == 0:
            primaries[gi] = -1
            continue

        best_count = -1
        best_hid = -1
        best_major_depth = -1
        best_major_count = -1
        best_major_hid = -1

        for ci in range(cand_idx.size):
            node_index = int(cand_idx[ci])
            node_pids = node_pid_lists[node_index]
            if node_pids.size == 0:
                continue
            # Both gal_pids and node_pids are sorted/unique uint64 arrays.
            c = _count_intersection_u64(gal_pids, node_pids)
            if c <= 0:
                continue

            hid_val = int(node_ids[node_index])
            if c > best_count or (c == best_count and hid_val > best_hid):
                best_count = c
                best_hid = hid_val

            if c > thresh:
                d = int(node_depth[node_index])
                if (
                    d > best_major_depth
                    or (
                        d == best_major_depth
                        and (
                            c > best_major_count
                            or (c == best_major_count and hid_val > best_major_hid)
                        )
                    )
                ):
                    best_major_depth = d
                    best_major_count = c
                    best_major_hid = hid_val

        primary = -1
        if best_major_hid != -1:
            primary = best_major_hid
        elif best_count > 0:
            primary = best_hid
        primaries[gi] = primary

    return primaries


def _pid_to_index_map(arr: np.ndarray) -> Dict[int, int]:
    """Build a simple PID -> index mapping for 1D arrays."""
    return {int(pid): int(i) for i, pid in enumerate(np.asarray(arr, dtype=np.int64).reshape(-1).tolist())}


def integrate_ahf_match_prune_inplace(sim, ahf_particles_file: str, fof_helper=None) -> None:
    """Integrate AHF matching per your spec, then prune and annotate."""

    do_ahf_check = os.environ.get('CAESAR_AHF_CHECK', '0') == '1'
    do_ahf_assert = os.environ.get('CAESAR_ASSERT_AHF', '0') == '1'
    pre_ngas = None
    if do_ahf_check:
        try:
            pre_ngas = [len(getattr(g, 'glist', [])) if getattr(g, 'glist', None) is not None else 0 for g in getattr(sim, 'galaxy_list', [])]
            mylog.info('AHF match: pre-collapse galaxies with gas=%d (total=%d)', sum(1 for v in pre_ngas if v > 0), len(pre_ngas))
        except Exception:
            pre_ngas = None

    if not hasattr(sim, 'galaxy_list') or len(sim.galaxy_list) == 0:
        return

    # Star PID array and PID->index map
    star_ids = get_property(sim, 'pid', 'star').d.astype(np.uint64)
    pid_to_star_index: Dict[int, int] = _pid_to_index_map(star_ids)

    # Build star_index -> galaxy_index map and per-galaxy star PID arrays.
    nstar = len(star_ids)
    staridx_to_galidx = np.full(nstar, -1, dtype=np.int32)
    galaxy_star_counts = np.zeros(len(sim.galaxy_list), dtype=np.int64)
    gal_star_pids: List[np.ndarray] = [np.empty(0, dtype=np.uint64) for _ in range(len(sim.galaxy_list))]

    dm_slist = getattr(sim.data_manager, 'slist', None)
    if dm_slist is None:
        dm_slist = np.array([], dtype=np.int64)
    for gi, gal in enumerate(sim.galaxy_list):
        sl = getattr(gal, 'slist', [])
        if sl is None:
            continue
        if not isinstance(sl, np.ndarray):
            sl = np.array(list(sl), dtype=np.int64)
        if sl.size == 0:
            continue
        try:
            concat_idx = dm_slist[sl]
            full_idx = sim.data_manager.indexes[concat_idx]
            staridx_to_galidx[full_idx] = gi
            gal_pids = star_ids[full_idx]
            if gal_pids.size > 0:
                gal_pids = np.unique(np.asarray(gal_pids, dtype=np.uint64))
            else:
                gal_pids = np.empty(0, dtype=np.uint64)
            gal_star_pids[gi] = gal_pids
            galaxy_star_counts[gi] = gal_pids.size
        except Exception:
            continue

    # Read AHF particles file once and collate per-node memberships.
    node_members: Dict[int, np.ndarray] = {}
    node_write_pos: Dict[int, int] = {}
    star_node_pids: Dict[int, np.ndarray] = {}
    gal_candidate_nodes: List[Set[int]] = [set() for _ in range(len(sim.galaxy_list))]

    with _open_ahf_particles(ahf_particles_file) as f:
        current_hid = None
        remaining = 0
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if remaining == 0 and len(parts) == 2:
                try:
                    remaining = int(parts[0])
                    current_hid = int(parts[1])
                except Exception:
                    current_hid = None
                    remaining = 0
                if current_hid is not None and current_hid not in node_members and remaining > 0:
                    node_members[current_hid] = np.empty((int(remaining), 2), dtype=np.int64)
                    node_write_pos[current_hid] = 0
                continue
            if remaining > 0:
                remaining -= 1
                pparts = line.split()
                if len(pparts) != 2:
                    continue
                try:
                    pid = int(pparts[0])
                    ptype = int(pparts[1])
                except Exception:
                    continue
                if current_hid is None:
                    continue

                arr = node_members.get(current_hid)
                if arr is not None and arr.size > 0:
                    pos = node_write_pos.get(current_hid, 0)
                    if pos < arr.shape[0]:
                        arr[pos, 0] = pid
                        arr[pos, 1] = ptype
                        node_write_pos[current_hid] = pos + 1

                if ptype == 4:
                    si = pid_to_star_index.get(pid)
                    if si is not None:
                        gi = int(staridx_to_galidx[si])
                        if gi >= 0:
                            gal_candidate_nodes[gi].add(current_hid)

                if remaining == 0 and current_hid is not None:
                    arr = node_members.get(current_hid)
                    if arr is not None and arr.size > 0:
                        used = node_write_pos.get(current_hid, arr.shape[0])
                        if used > 0:
                            sub = arr[:used]
                            star_mask = (sub[:, 1] == 4)
                            if np.any(star_mask):
                                star_pids = np.unique(sub[star_mask, 0].astype(np.uint64))
                            else:
                                star_pids = np.empty(0, dtype=np.uint64)
                        else:
                            star_pids = np.empty(0, dtype=np.uint64)
                        star_node_pids[current_hid] = star_pids
                    if current_hid in node_members:
                        del node_members[current_hid]
                    if current_hid in node_write_pos:
                        del node_write_pos[current_hid]

    # Build hierarchy and depth cache.
    parent_of, children_of = _read_ahf_hierarchy(ahf_particles_file)
    depth_cache: Dict[int, int] = {}

    def depth(h: int) -> int:
        if h in depth_cache:
            return depth_cache[h]
        d = 0
        cur = h
        while True:
            p = parent_of.get(cur, 0)
            if p is None or p == 0:
                break
            d += 1
            cur = p
        depth_cache[h] = d
        return d

    # Prepare typed lists for Numba selector.
    ngal = len(sim.galaxy_list)
    node_ids_list = sorted(star_node_pids.keys())
    nnode = len(node_ids_list)
    node_ids_arr = np.asarray(node_ids_list, dtype=np.int64)
    node_depth = np.zeros(nnode, dtype=np.int64)
    for idx, hid in enumerate(node_ids_list):
        node_depth[idx] = depth(int(hid))

    id_to_node_index: Dict[int, int] = {hid: i for i, hid in enumerate(node_ids_list)}

    gal_pid_lists_nb = NumbaList()
    for gi in range(ngal):
        gal_pid_lists_nb.append(gal_star_pids[gi])

    node_pid_lists_nb = NumbaList()
    for hid in node_ids_list:
        arr = star_node_pids.get(hid)
        if arr is None:
            arr = np.empty(0, dtype=np.int64)
        node_pid_lists_nb.append(arr)

    cand_node_lists_nb = NumbaList()
    for gi in range(ngal):
        cands = gal_candidate_nodes[gi]
        if not cands:
            cand_node_lists_nb.append(np.empty(0, dtype=np.int64))
            continue
        indices: List[int] = []
        for hid in cands:
            idx = id_to_node_index.get(hid, -1)
            if idx >= 0:
                indices.append(idx)
        if indices:
            cand_node_lists_nb.append(np.asarray(indices, dtype=np.int64))
        else:
            cand_node_lists_nb.append(np.empty(0, dtype=np.int64))

    # Run Numba selector (parallel) with fallback to Python if needed.
    selected: List[int] = []
    try:
        try:
            set_num_threads(max(1, int(getattr(sim, 'nproc', 1))))
        except Exception:
            pass
        primaries = _numba_select_ahf_for_gals(
            gal_pid_lists_nb,
            node_pid_lists_nb,
            node_depth,
            node_ids_arr,
            cand_node_lists_nb,
            galaxy_star_counts.astype(np.int64),
        )
        for gi in range(ngal):
            val = int(primaries[gi])
            selected.append(val if val != -1 else -1)
    except Exception:
        # Fallback to serial Python logic
        indices = list(range(len(sim.galaxy_list)))

        def _select_for_gal(gi: int) -> Tuple[int, int]:
            gal_pids = gal_star_pids[gi]
            if gal_pids.size == 0:
                return -1, -1
            candidates = gal_candidate_nodes[gi]
            if not candidates:
                return -1, -1
            counts: Dict[int, int] = {}
            best_hid = -1
            best_count = -1
            for hid in candidates:
                node_pids = star_node_pids.get(hid)
                if node_pids is None or node_pids.size == 0:
                    continue
                overlap = np.intersect1d(gal_pids, node_pids, assume_unique=True)
                c = int(overlap.size)
                if c <= 0:
                    continue
                counts[hid] = c
                if c > best_count or (c == best_count and hid > best_hid):
                    best_count = c
                    best_hid = hid
            if not counts:
                return -1, -1
            total = int(galaxy_star_counts[gi])
            thresh = 0.5 * total if total > 0 else 0
            cands = [hid for hid, c in counts.items() if c > thresh] if total > 0 else []
            if cands:
                cands.sort(key=lambda h: (depth(int(h)), counts[h], h))
                primary = int(cands[-1])
            else:
                primary = -1
            if primary == -1 and best_count > 0:
                primary = int(best_hid)
            return primary, best_hid

        selection_info: List[Tuple[int, int]] = [_select_for_gal(gi) for gi in indices]
        for primary, best in selection_info:
            if primary is not None and primary != -1:
                selected.append(int(primary))
            else:
                chosen = best if best is not None else -1
                selected.append(int(chosen) if chosen != -1 else -1)

    galaxy_to_ahf_nodes = [int(h) if h is not None else -1 for h in selected]

    from collections import defaultdict as _dd

    mapping: Dict[int, List[int]] = _dd(list)
    for gi, hid in enumerate(selected):
        if hid is not None and int(hid) != -1:
            mapping[int(hid)].append(gi)

    if not mapping:
        sim._ahf_galaxy_hosts = [-1] * len(sim.galaxy_list)
        sim._ahf_galaxy_ahf_ids = galaxy_to_ahf_nodes
        for gi, node_id in enumerate(galaxy_to_ahf_nodes):
            ahf_val = int(node_id) if node_id is not None and node_id >= 0 else -1
            if gi < len(sim.galaxy_list):
                setattr(sim.galaxy_list[gi], 'AHF_haloID', ahf_val)
        return

    matched_ids = set(mapping.keys())
    if not matched_ids:
        sim._ahf_galaxy_hosts = [-1] * len(sim.galaxy_list)
        sim._ahf_galaxy_ahf_ids = galaxy_to_ahf_nodes
        for gi, node_id in enumerate(galaxy_to_ahf_nodes):
            ahf_val = int(node_id) if node_id is not None and node_id >= 0 else -1
            if gi < len(sim.galaxy_list):
                setattr(sim.galaxy_list[gi], 'AHF_haloID', ahf_val)
        return

    pid_maps_sel: Dict[str, _PidLookup] = _build_selected_pid_maps(sim)

    def map_set(pidset: Set[int], key: str) -> np.ndarray:
        lookup = pid_maps_sel.get(key)
        if lookup is None or not pidset:
            return np.empty(0, dtype=np.int32)
        mapped = lookup.map(pidset)
        if mapped.size == 0:
            return mapped
        return np.unique(mapped)

    exclusive_gal_dm = None

    if do_ahf_check or do_ahf_assert:
        post_ngas = []
        try:
            post_ngas = [len(getattr(g, 'glist', [])) if getattr(g, 'glist', None) is not None else 0 for g in getattr(sim, 'galaxy_list', [])]
        except Exception:
            post_ngas = []
        mylog.info('AHF match: post-collapse galaxies with gas=%d (total=%d)', sum(1 for v in post_ngas if v > 0), len(post_ngas))
        if do_ahf_assert and pre_ngas is not None and post_ngas:
            pre_with = sum(1 for v in pre_ngas if v > 0)
            post_with = sum(1 for v in post_ngas if v > 0)
            if pre_with > 0 and post_with == 0:
                mylog.error('Assertion: Gas lost after AHF collapse (pre_with=%d, post_with=%d)', pre_with, post_with)
                raise AssertionError('Gas lost after AHF collapse: nonzero pre-collapse gas count dropped to zero')

    if exclusive_gal_dm is not None:
        setattr(sim, '_exclusive_galaxy_dmlist', exclusive_gal_dm)

    if sim.ngalaxies > 0:
        try:
            from caesar.group import get_group_properties
            get_group_properties(sim, sim.galaxy_list)
        except Exception:
            pass

    missing_halo_masses = [
        halo for halo in sim.halo_list
        if not isinstance(getattr(halo, 'masses', None), dict)
        or 'H2' not in halo.masses
    ]

    final_map: Dict[int, Tuple[float, float]] = {}
    if missing_halo_masses:
        final_ok, final_map = _populate_hydrogen_masses(sim, missing_halo_masses)
        if not final_ok:
            mylog.info('Final halo pass missing HI/H2; invoking hydrogen_mass_calc()')
            import caesar.hydrogen_mass_calc as hydrogen_mass_calc
            hydrogen_mass_calc.hydrogen_mass_calc(sim)
            final_ok, final_map = _populate_hydrogen_masses(sim, missing_halo_masses)
            if not final_ok:
                mylog.warning('Unable to populate HI/H2 masses after hydrogen_mass_calc(); affected halos may lack gas data')
                final_map = {}
    if final_map:
        existing = getattr(sim, '_ahf_halo_hydrogen_masses', {})
        existing.update(final_map)
        setattr(sim, '_ahf_halo_hydrogen_masses', existing)

    for halo in sim.halo_list:
        if 'dm2' in getattr(sim.data_manager, 'ptypes', []):
            if not hasattr(halo, 'dm2list'):
                halo.dm2list = np.empty(0, dtype=np.int64)
            if not hasattr(halo, 'ndm2'):
                halo.ndm2 = len(halo.dm2list)
            halo.masses.setdefault('dm2', _compute_mass_quantity(sim, 0.0))
        if 'dm3' in getattr(sim.data_manager, 'ptypes', []):
            if not hasattr(halo, 'dm3list'):
                halo.dm3list = np.empty(0, dtype=np.int64)
            if not hasattr(halo, 'ndm3'):
                halo.ndm3 = len(halo.dm3list)
            halo.masses.setdefault('dm3', _compute_mass_quantity(sim, 0.0))

    _update_ahf_halo_maps(sim)


__all__ = ["integrate_ahf_match_prune_inplace"]
