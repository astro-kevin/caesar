from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional, Dict, Iterable, Iterator
import os
import re
import glob
from itertools import islice

import h5py
from tqdm import tqdm
import numpy as np
import gzip

try:  # pragma: no cover - optional acceleration
    from numba import njit, prange, types, set_num_threads
    from numba.typed import List as NumbaList, Set as NumbaSet
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    _NUMBA_AVAILABLE = False
    njit = None
    prange = range
    types = None
    set_num_threads = None
    NumbaList = None
    NumbaSet = None

try:  # optional progress bar for numba kernels
    from numba_progress import ProgressBar
    _HAS_NUMBA_PROGRESS = True
except Exception:  # pragma: no cover
    _HAS_NUMBA_PROGRESS = False
    ProgressBar = None


def _open_ahf_particles(path: str):
    """Open an AHF particles/halos text file, transparently handling gzip."""
    if path.endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path, 'r')


@dataclass
class ParticleMembership:
    """Store particle IDs for a single halo or galaxy."""
    id: int
    parttype0: Set[int] = field(default_factory=set)
    parttype1: Set[int] = field(default_factory=set)
    parttype4: Set[int] = field(default_factory=set)
    parttype5: Set[int] = field(default_factory=set)


def _build_numba_star_sets(memberships: List["ParticleMembership"]):
    """Convert parttype4 memberships into Numba typed sets."""
    if not _NUMBA_AVAILABLE:
        raise RuntimeError("Numba is not available")
    nb_list = NumbaList()
    sizes = np.empty(len(memberships), dtype=np.int64)
    key_type = types.int64
    for idx, pm in enumerate(memberships):
        nb_set = NumbaSet.empty(key_type)
        for pid in pm.parttype4:
            nb_set.add(int(pid))
        nb_list.append(nb_set)
        sizes[idx] = len(pm.parttype4)
    return nb_list, sizes


if _NUMBA_AVAILABLE:

    @njit(parallel=True)
    def _numba_match_galaxies_to_halos(gal_sets, gal_sizes, halo_sets, progress_proxy=None):
        ng = len(gal_sets)
        nh = len(halo_sets)
        result = np.empty(ng, dtype=np.int64)
        for i in prange(ng, schedule='dynamic'):
            gsize = gal_sizes[i]
            if gsize == 0 or nh == 0:
                result[i] = -1
                continue
            gset = gal_sets[i]
            best_idx = -1
            best_overlap = 0
            for j in range(nh):
                hset = halo_sets[j]
                inter = 0
                for val in gset:
                    if val in hset:
                        inter += 1
                if inter * 2 > gsize:
                    best_idx = j
                    break
                if inter > best_overlap:
                    best_overlap = inter
                    best_idx = j
            result[i] = best_idx
            if progress_proxy is not None:
                progress_proxy.update(1)
        return result

else:  # pragma: no cover - executed only when Numba missing

    def _numba_match_galaxies_to_halos(*args, **kwargs):  # type: ignore
        raise RuntimeError("Numba is not available")


try:  # Optional Cython accelerators
    from caesar._fast_ahf import iter_memberships as _fast_iter_memberships
    from caesar._fast_ahf import read_memberships_for_ids as _fast_read_memberships
    _HAS_FAST_AHF = True
except ImportError:  # pragma: no cover - optional extension
    _HAS_FAST_AHF = False


def read_file_to_structure(
    file_path: str,
    *,
    lines: Optional[List[str]] = None,
    max_lines: Optional[int] = None,
    ptype_filter: Optional[Set[int]] = None,
) -> List[ParticleMembership]:
    """Parse an AHF particle file into ``ParticleMembership`` objects."""
    if lines is None:
        with open(file_path, "r") as f:
            lines = f.readlines()
    if not lines:
        return []

    n = len(lines) if max_lines is None else min(len(lines), max_lines)
    results: List[ParticleMembership] = []
    i = 0
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        parts = line.split()
        if len(parts) != 2:
            i += 1
            continue
        try:
            expected_particles = int(parts[0])
            current_id = int(parts[1])
        except ValueError:
            i += 1
            continue
        membership = ParticleMembership(current_id)
        i += 1
        particle_lines_parsed = 0
        while particle_lines_parsed < expected_particles and i < n:
            pline = lines[i].strip()
            if not pline:
                i += 1
                continue
            pparts = pline.split()
            if len(pparts) != 2:
                raise ValueError(f"Invalid particle line at line {i+1}: {pline}")
            pid = int(pparts[0])
            ptype = int(pparts[1])
            if ptype_filter is not None and ptype not in ptype_filter:
                particle_lines_parsed += 1
                i += 1
                continue
            if ptype == 0:
                membership.parttype0.add(pid)
            elif ptype == 1:
                membership.parttype1.add(pid)
            elif ptype == 4:
                membership.parttype4.add(pid)
            elif ptype == 5:
                membership.parttype5.add(pid)
            particle_lines_parsed += 1
            i += 1
        if particle_lines_parsed != expected_particles:
            raise ValueError(
                f"Expected {expected_particles} particle lines but parsed {particle_lines_parsed}"
            )
        results.append(membership)
    return results


def load_hdf5_to_namedata(filename: str) -> List[ParticleMembership]:
    """Load HDF5 datasets named by halo ID into ``ParticleMembership`` objects."""
    with h5py.File(filename, "r") as f:
        dataset_names = list(f.keys())

    membership: List[ParticleMembership] = []
    with h5py.File(filename, "r") as f:
        for name in tqdm(dataset_names, desc="Processing datasets..."):
            data = f[name][:]
            membership.append(
                ParticleMembership(id=int(name), parttype4=set(map(int, data)))
            )
    return membership


def galaxies_to_namedata(galaxies, star_particle_ids) -> List[ParticleMembership]:
    """Convert galaxy objects and star particle IDs into ``ParticleMembership``"""
    membership: List[ParticleMembership] = []
    for gal in galaxies:
        try:
            slist = getattr(gal, "slist", [])
        except Exception:
            slist = []
        ids = set(int(star_particle_ids[i]) for i in slist)
        gid = int(getattr(gal, "GroupID", getattr(gal, "id", 0)))
        membership.append(ParticleMembership(id=gid, parttype4=ids))
    return membership


def load_snapshot_to_namedata(snapshot_file: str, galaxies) -> List[ParticleMembership]:
    """Load star particle IDs from a snapshot and build ``ParticleMembership``."""
    from readgadget import readsnap

    particle_ids = readsnap(snapshot_file, "pid", "star")
    return galaxies_to_namedata(galaxies, particle_ids)


def find_best_matches(
    list1: List[ParticleMembership],
    list2: List[ParticleMembership],
    n_jobs: Optional[int] = None,
    show_progress: bool = False,
) -> List[Tuple[int, int]]:
    """Match halos by overlapping ``parttype4`` particle IDs using Numba."""

    if not _NUMBA_AVAILABLE:
        raise RuntimeError(
            "AHF matching requires the 'numba' package; please install numba to continue."
        )

    if n_jobs is not None and set_num_threads is not None:
        try:
            set_num_threads(max(1, int(n_jobs)))
        except Exception:
            pass

    n_list1 = len(list1)
    if n_list1 == 0:
        return []
    if len(list2) == 0:
        return [(pm.id, -1) for pm in list1]

    list1_ids = np.fromiter((pm.id for pm in list1), dtype=np.int64, count=n_list1)

    gal_sets, gal_sizes = _build_numba_star_sets(list1)
    halo_sets, _ = _build_numba_star_sets(list2)
    if len(halo_sets) == 0:
        return [(pm.id, -1) for pm in list1]

    if show_progress and _HAS_NUMBA_PROGRESS:
        with ProgressBar(total=n_list1) as progress_proxy:
            match_indices = _numba_match_galaxies_to_halos(gal_sets, gal_sizes, halo_sets, progress_proxy)
    else:
        match_indices = _numba_match_galaxies_to_halos(gal_sets, gal_sizes, halo_sets, None)
    halo_ids = np.fromiter((pm.id for pm in list2), dtype=np.int64, count=len(list2))
    hcount = halo_ids.size

    results: List[Tuple[int, int]] = []
    for i, idx in enumerate(match_indices):
        hid = -1
        if 0 <= idx < hcount:
            hid = int(halo_ids[idx])
        results.append((int(list1_ids[i]), hid))
    return results

def _build_selected_pid_maps(sim) -> Dict[str, Dict[int, int]]:
    """Build PID -> selected-index maps for each ptype present in the CAESAR-selected subset.

    The selected index is the index within the CAESAR DataManager per-type lists
    (e.g., ``data_manager.slist``, ``glist``, etc.), not the full-snapshot index.

    Returns a dict like { 'star': {pid: sel_idx}, ... } for the ptypes available.
    """
    from caesar.property_manager import get_property, has_ptype

    maps: Dict[str, Dict[int, int]] = {}

    def add_map(ptype: str, list_name: str) -> None:
        if not has_ptype(sim, ptype):
            return
        sel = getattr(sim.data_manager, list_name, None)
        if sel is None or len(sel) == 0:
            return
        # Map CAESAR-selected per-type indices back to full-snapshot per-type indices
        # via DataManager.indexes, then obtain PIDs and build pid->selected-index map.
        try:
            full_indices = sim.data_manager.indexes[sel]
            full_pids = get_property(sim, 'pid', ptype).d.astype(np.int64)
            sel_pids = full_pids[full_indices]
            maps[ptype] = {int(pid): int(i) for i, pid in enumerate(sel_pids.tolist())}
        except Exception:
            return

    add_map('star', 'slist')
    add_map('gas', 'glist')
    add_map('bh', 'bhlist')
    add_map('dm', 'dmlist')
    return maps


def _collect_all_node_ids(parent_of: Dict[int, Optional[int]], children_of: Dict[int, List[int]]) -> Set[int]:
    ids: Set[int] = set(parent_of.keys())
    for kids in children_of.values():
        ids.update(kids)
    return ids


def _group_nodes_by_root(parent_of: Dict[int, Optional[int]]) -> Tuple[Dict[int, Set[int]], Dict[int, int]]:
    """Return {root: {nodes}} plus a node -> root cache for quick lookup."""

    host_to_nodes: Dict[int, Set[int]] = {}
    node_to_root: Dict[int, int] = {}

    def resolve_root(node: int) -> int:
        trace: List[int] = []
        cur = node
        while True:
            cached = node_to_root.get(cur)
            if cached is not None:
                root = cached
                break
            parent = parent_of.get(cur, 0)
            if parent in (0, None):
                root = cur
                break
            trace.append(cur)
            cur = int(parent)
        for visited in trace:
            node_to_root[visited] = root
        node_to_root[cur] = root
        return root

    for hid in parent_of.keys():
        root = resolve_root(int(hid))
        host_to_nodes.setdefault(root, set()).add(int(hid))

    return host_to_nodes, node_to_root


def _iter_memberships_stream(
    path: str,
    needed: Set[int],
    *,
    load_dm: bool = False,
) -> Iterator[ParticleMembership]:
    """Yield memberships for nodes in ``needed`` while streaming the file."""

    if _HAS_FAST_AHF:
        yield from _fast_iter_memberships(path, needed, load_dm)
        return

    with _open_ahf_particles(path) as fh:
        current_hid = None
        remaining = 0
        cur_pm: Optional[ParticleMembership] = None

        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if remaining == 0 and len(parts) == 2:
                if cur_pm is not None and cur_pm.id in needed:
                    yield cur_pm
                try:
                    remaining = int(parts[0])
                    current_hid = int(parts[1])
                except Exception:
                    current_hid = None
                    remaining = 0
                    cur_pm = None
                    continue
                cur_pm = ParticleMembership(current_hid) if current_hid in needed else None
                continue

            if remaining > 0:
                remaining -= 1
                if cur_pm is None:
                    continue
                pparts = line.split()
                if len(pparts) != 2:
                    continue
                try:
                    pid = int(pparts[0])
                    ptype = int(pparts[1])
                except Exception:
                    continue

                if ptype == 0:
                    cur_pm.parttype0.add(pid)
                elif ptype == 4:
                    cur_pm.parttype4.add(pid)
                elif ptype == 5:
                    cur_pm.parttype5.add(pid)
                elif load_dm and ptype == 1:
                    cur_pm.parttype1.add(pid)

        if cur_pm is not None and cur_pm.id in needed:
            yield cur_pm


def _read_memberships_for_ids(
    path: str,
    needed: Set[int],
    *,
    load_dm: bool = True,
) -> Dict[int, ParticleMembership]:
    """Read AHF particle memberships only for the given IDs (supports .gz)."""
    if _HAS_FAST_AHF:
        return _fast_read_memberships(path, needed, load_dm)

    out: Dict[int, ParticleMembership] = {}

    with _open_ahf_particles(path) as f:
        current_hid = None
        remaining = 0
        cur_pm: Optional[ParticleMembership] = None
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if remaining == 0 and len(parts) == 2:
                # finalize previous if any
                if cur_pm is not None and cur_pm.id in needed:
                    out[cur_pm.id] = cur_pm
                # new header
                try:
                    remaining = int(parts[0])
                    current_hid = int(parts[1])
                except Exception:
                    current_hid = None
                    remaining = 0
                    cur_pm = None
                    continue
                cur_pm = ParticleMembership(current_hid) if current_hid in needed else None
                continue
            if remaining > 0:
                remaining -= 1
                if cur_pm is None:
                    continue
                pparts = line.split()
                if len(pparts) != 2:
                    continue
                try:
                    pid = int(pparts[0])
                    ptype = int(pparts[1])
                except Exception:
                    continue
                if ptype == 0:
                    cur_pm.parttype0.add(pid)
                elif ptype == 1 and load_dm:
                    cur_pm.parttype1.add(pid)
                elif ptype == 4:
                    cur_pm.parttype4.add(pid)
                elif ptype == 5:
                    cur_pm.parttype5.add(pid)
        # finalize last
        if cur_pm is not None and cur_pm.id in needed:
            out[cur_pm.id] = cur_pm
    return out


def _root_of(node: int, parent_of: Dict[int, Optional[int]]) -> int:
    cur = node
    seen = set()
    while True:
        if cur in seen:
            return cur
        seen.add(cur)
        p = parent_of.get(cur, 0)
        if p is None or p == 0:
            return cur
        cur = int(p)


def build_galaxies_from_ahf_fast(
    sim,
    ahf_particles_file: str,
    *,
    min_stars: int = 16,
    n_jobs: Optional[int] = None,
) -> None:
    """Build galaxies directly from AHF nodes using a star-count gate.

    Streaming host-by-host keeps memory bounded while joblib threads reuse
    the shared DataManager arrays for per-galaxy construction.
    """

    from caesar.group import create_new_group
    from caesar.group import get_group_properties as _get_group_properties
    from caesar.property_manager import get_property, has_ptype

    pid_maps_sel = _build_selected_pid_maps(sim)
    dm_pid_map: Dict[int, int] = pid_maps_sel.get('dm', {})
    pid_to_dm_fullidx: Dict[int, int] = {}
    ndm_full = 0
    if dm_pid_map and has_ptype(sim, 'dm'):
        dm_pids_full = get_property(sim, 'pid', 'dm').d.astype(np.int64)
        ndm_full = len(dm_pids_full)
        if ndm_full > 0:
            pid_to_dm_fullidx = {int(pid): int(i) for i, pid in enumerate(dm_pids_full.tolist())}

    parent_of, children_of = _read_ahf_hierarchy(ahf_particles_file)
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

    nodes_remaining: Dict[int, int] = {root: len(nodes) for root, nodes in host_to_nodes.items()}
    pending_members: Dict[int, Dict[int, ParticleMembership]] = {}

    jobs = n_jobs
    if jobs is None:
        jobs = getattr(sim, 'nproc', 1)
    try:
        jobs = int(jobs)
    except Exception:
        jobs = 1
    jobs = max(1, jobs)

    show_progress = bool(getattr(sim, '_show_progress', True))
    total_hosts = len(host_to_nodes)
    host_progress = tqdm(
        total=total_hosts,
        desc="Building galaxies (AHF-FAST)",
        disable=(not show_progress or total_hosts == 0),
        leave=False,
    )

    def map_sel(pidset: Set[int], key: str) -> np.ndarray:
        mp = pid_maps_sel.get(key, {})
        if not mp or not pidset:
            return np.array([], dtype=np.int32)
        arr = np.array([mp[pid] for pid in pidset if pid in mp], dtype=np.int32)
        if arr.size == 0:
            return arr
        return np.unique(arr)

    def process_host(root_id: int, bucket: Dict[int, ParticleMembership]) -> None:
        nodes_for_host = host_to_nodes.get(root_id, set())
        if not nodes_for_host:
            if host_progress is not None:
                host_progress.update(1)
            return

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
            return

        def build_group(payload: Tuple[int, Set[int], Set[int], Set[int], Set[int]]):
            node_id, star_set, gas_set, bh_set, dm_exc = payload
            # Use the streamed membership data for inclusive DM lookups so
            # joblib workers do not rely on a non-existent outer scope
            dm_pm = bucket.get(node_id)
            dm_inclusive = dm_pm.parttype1 if dm_pm is not None else set()
            grp = create_new_group(sim, 'galaxy')
            grp.slist = map_set(star_set, 'star')
            grp.glist = map_set(gas_set, 'gas')
            if 'bh' in pid_maps_sel:
                grp.bhlist = map_set(bh_set, 'bh')
            if 'dm' in pid_maps_sel:
                dm_selected = map_set(dm_inclusive, 'dm')
            else:
                dm_selected = np.array([], dtype=np.int32)
            grp.dmlist = dm_selected
            grp.global_indexes = np.array([], dtype=np.int64)
            return grp, dm_exc

        results: List[Tuple] = []
        try:
            from joblib import Parallel, delayed

            results = Parallel(
                n_jobs=jobs,
                backend='threading',
                prefer='threads',
                require='sharedmem',
            )(delayed(build_group)(payload) for payload in payloads)
        except Exception:
            results = [build_group(payload) for payload in payloads]

        base_index = len(galaxies)
        for offset, (grp, dm_exc) in enumerate(results):
            galaxies.append(grp)
            # Group IDs will be reassigned later; keep exclusive DM on object
            if dm_exc:
                grp.__dict__['_dm_exclusive_pids'] = set(dm_exc)
            else:
                grp.__dict__['_dm_exclusive_pids'] = set()

        if host_progress is not None:
            host_progress.update(1)

    load_dm = bool(dm_pid_map)
    for pm in _iter_memberships_stream(ahf_particles_file, all_needed_nodes, load_dm=load_dm):
        root = node_to_root.get(pm.id)
        if root is None:
            continue
        bucket = pending_members.setdefault(root, {})
        bucket[pm.id] = pm
        nodes_remaining[root] = nodes_remaining.get(root, 0) - 1
        if nodes_remaining[root] <= 0:
            process_host(root, bucket)
            bucket.clear()
            del pending_members[root]
            nodes_remaining.pop(root, None)

    for root, bucket in list(pending_members.items()):
        if bucket:
            process_host(root, bucket)
        pending_members.pop(root, None)
        nodes_remaining.pop(root, None)

    if host_progress is not None:
        host_progress.close()

    sim.galaxy_list = galaxies
    sim.ngalaxies = len(galaxies)
    setattr(sim, "_ahf_matched", True)
    setattr(sim, "_include_dm_in_galaxies", True)
    if sim.ngalaxies == 0:
        return

    def _refresh_global_indexes(gal) -> None:
        blocks = []
        try:
            if hasattr(gal, 'glist') and gal.glist is not None and len(gal.glist) > 0:
                blocks.append(sim.data_manager.glist[gal.glist])
        except Exception:
            pass
        try:
            if hasattr(gal, 'slist') and gal.slist is not None and len(gal.slist) > 0:
                blocks.append(sim.data_manager.slist[gal.slist])
        except Exception:
            pass
        try:
            if hasattr(gal, 'dmlist') and gal.dmlist is not None and len(gal.dmlist) > 0 and has_ptype(sim, 'dm'):
                blocks.append(sim.data_manager.dmlist[gal.dmlist])
        except Exception:
            pass
        try:
            if hasattr(gal, 'bhlist') and gal.bhlist is not None and len(gal.bhlist) > 0:
                blocks.append(sim.data_manager.bhlist[gal.bhlist])
        except Exception:
            pass
        if blocks:
            gal.global_indexes = np.concatenate(blocks).astype(np.int64)
        else:
            gal.global_indexes = np.array([], dtype=np.int64)

    for gal in sim.galaxy_list:
        _refresh_global_indexes(gal)

    class _Ctx:
        def __init__(self, sim):
            self.obj = sim
            self.obj_type = 'galaxy'
            self.nproc = getattr(sim, 'nproc', 1)
            self.load_pot = getattr(sim, 'load_pot', True)
            self.nparttot = sum(len(getattr(g, 'global_indexes', [])) for g in sim.galaxy_list)
            self.nparttype = {
                p: len(getattr(sim.data_manager, f"{p}list", []))
                for p in ['gas', 'star', 'bh', 'dm', 'dm2', 'dm3']
                if hasattr(sim.data_manager, f"{p}list")
            }
            self.counts = {'galaxy': len(sim.galaxy_list)}

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
        if 'galaxy' not in sim.group_types:
            sim.group_types.append('galaxy')
    except Exception:
        pass

    if ndm_full > 0 and dm_pid_map:
        exclusive_gal_dm = np.full(ndm_full, -1, dtype=np.int32)
        for gal in sim.galaxy_list:
            dm_exc = getattr(gal, '_dm_exclusive_pids', None)
            if not dm_exc:
                continue
            for pid in dm_exc:
                mapped = pid_to_dm_fullidx.get(int(pid))
                if mapped is not None:
                    exclusive_gal_dm[mapped] = int(getattr(gal, 'GroupID', -1))
            # drop the cached set to free memory
            try:
                del gal.__dict__['_dm_exclusive_pids']
            except KeyError:
                pass
        setattr(sim, '_exclusive_galaxy_dmlist', exclusive_gal_dm)
    else:
        for gal in sim.galaxy_list:
            if '_dm_exclusive_pids' in gal.__dict__:
                del gal.__dict__['_dm_exclusive_pids']
        if hasattr(sim, '_exclusive_galaxy_dmlist'):
            delattr(sim, '_exclusive_galaxy_dmlist')


def get_caesar_file(directory: str, number: int) -> str:
    filename = os.path.join(directory, f"Simba_M200_snap_{number:03d}.h5")
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File {filename} does not exist.")
    return filename


def get_AHF_file(directory: str, number: int) -> str:
    num_string = f"{number:03d}"
    pattern = os.path.join(directory, f"Simba_M200_snap_{num_string}.z*.AHF_particles")
    for path in glob.glob(pattern):
        if re.search(r"\.z\d+\.\d{3}\.AHF_particles$", path):
            return path
    raise FileNotFoundError(
        f"No matching file found for number {number} in the specified directory."
    )


def summarize_matches(
    best_matches: List[Tuple[int, int]],
    caesar_data: List[ParticleMembership],
    ahf_data: List[ParticleMembership],
) -> None:
    total = len(best_matches)
    matched = [m for m in best_matches if m[1] != -1]
    percent = 100 * len(matched) / total if total > 0 else 0
    print(
        f"Number of galaxies matched: {len(matched)} / {total} ({percent:.2f}%)"
    )

    ahf_dict = {pm.id: pm for pm in ahf_data}
    caesar_dict = {pm.id: pm for pm in caesar_data}

    overlaps = []
    for caesar_id, ahf_id in matched:
        caesar_set = caesar_dict.get(caesar_id, ParticleMembership(caesar_id)).parttype4
        ahf_set = ahf_dict.get(ahf_id, ParticleMembership(ahf_id)).parttype4
        if caesar_set:
            overlaps.append(len(caesar_set & ahf_set) / len(caesar_set))

    if overlaps:
        print("Overlap fraction statistics (based on parttype4):")
        print(f"  Mean: {sum(overlaps)/len(overlaps):.4f}")
        med = sorted(overlaps)[len(overlaps)//2]
        print(f"  Median: {med:.4f}")
        print(f"  Min: {min(overlaps):.4f}")
        print(f"  Max: {max(overlaps):.4f}")
    else:
        print("No overlap fractions to report.")

    counts = {}
    for _, hid in matched:
        counts[hid] = counts.get(hid, 0) + 1
    repeated = sum(1 for v in counts.values() if v > 1)
    print(f"Number of repeated halo matches: {repeated}")

    top5 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 most frequently matched halos:")
    for halo_id, count in top5:
        print(f"  Halo ID {halo_id} matched {count} times")


def read_single_halo_from_file(file_path: str, halo_id: int) -> Optional[ParticleMembership]:
    """Read a single halo from an AHF particle file."""
    with open(file_path, "r") as io:
        lines = iter(io.readlines())
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                expected = int(parts[0])
                current = int(parts[1])
            except ValueError:
                continue
            if current == halo_id:
                pm = ParticleMembership(current)
                parsed = 0
                for pline in islice(lines, expected):
                    pline = pline.strip()
                    if not pline:
                        continue
                    pparts = pline.split()
                    if len(pparts) != 2:
                        continue
                    pid = int(pparts[0])
                    ptype = int(pparts[1])
                    if ptype == 0:
                        pm.parttype0.add(pid)
                    elif ptype == 1:
                        pm.parttype1.add(pid)
                    elif ptype == 4:
                        pm.parttype4.add(pid)
                    elif ptype == 5:
                        pm.parttype5.add(pid)
                    parsed += 1
                return pm
            else:
                # skip this halo
                try:
                    skip = int(parts[0])
                except ValueError:
                    continue
                for _ in range(skip):
                    next(lines, None)
    return None


def match_subhalos_to_galaxies(
    sim,
    *,
    ahf_file: str,
    snapshot_file: Optional[str] = None,
    star_particle_ids: Optional[List[int]] = None,
) -> None:
    """Match AHF subhalos to CAESAR galaxies and update ``sim`` in-place.

    Parameters
    ----------
    sim : :class:`CAESAR`
        Loaded CAESAR simulation object with galaxies already found.
    ahf_file : str
        Path to the ``AHF_particles`` file containing substructure information.
    snapshot_file : str, optional
        Path to the snapshot used to build ``sim``. Required if ``star_particle_ids``
        is not provided.
    star_particle_ids : list[int], optional
        Pre-loaded star particle IDs corresponding to ``sim.galaxies``.  This is
        mainly for testing purposes to avoid snapshot I/O.
    """

    if star_particle_ids is None:
        if snapshot_file is None:
            raise ValueError("Must specify `snapshot_file` or `star_particle_ids`")
        caesar_data = load_snapshot_to_namedata(snapshot_file, sim.galaxies)
    else:
        caesar_data = galaxies_to_namedata(sim.galaxies, star_particle_ids)

    ahf_data = read_file_to_structure(ahf_file, ptype_filter={1, 4})
    ahf_by_id = {pm.id: pm for pm in ahf_data}
    matches = find_best_matches(
        caesar_data,
        ahf_data,
        n_jobs=getattr(sim, 'nproc', None),
        show_progress=bool(getattr(sim, '_show_progress', True)),
    )

    # Build mapping from AHF halo -> list of galaxy indices
    from collections import defaultdict

    mapping: defaultdict[int, List[int]] = defaultdict(list)
    for idx, (_, hid) in enumerate(matches):
        if hid != -1:
            mapping[hid].append(idx)

    updated = []
    used = set()
    for ahf_id, gal_indices in mapping.items():
        subhalo = ahf_by_id.get(ahf_id)
        if subhalo is None:
            continue
        base = sim.galaxies[gal_indices[0]]
        merged_slist: List[int] = []
        for gi in gal_indices:
            merged_slist.extend(list(sim.galaxies[gi].slist))
            used.add(gi)
        base.slist = merged_slist
        base.masses['dm'] = float(len(subhalo.parttype1))
        updated.append(base)

    # add unmatched galaxies
    for idx, gal in enumerate(sim.galaxies):
        if idx not in used:
            updated.append(gal)

    # help Python release memory held by the particle cache
    del ahf_data

    # Recalculate galaxy properties with the merged particle lists if possible
    sim.galaxies = updated
    sim.ngalaxies = len(updated)
    try:
        from caesar.group import get_group_properties
        get_group_properties(sim, sim.galaxies)
    except Exception:  # pragma: no cover - optional heavy deps or incomplete sim
        return


def _maybe_readlines(file_path: str) -> List[str]:
    """Read all lines from a text file, supporting optional gzip compression."""
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rt') as f:
            return f.readlines()
    with open(file_path, 'r') as f:
        return f.readlines()


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


def _collect_ancestors(hids: Iterable[int], parent_of: Dict[int, int]) -> Set[int]:
    out: Set[int] = set()
    for h in hids:
        cur = h
        while True:
            p = parent_of.get(cur, 0)
            if p is None or p == 0:
                break
            if p in out:
                break
            out.add(p)
            cur = p
    return out


def _collect_descendants(hids: Iterable[int], children_of: Dict[int, List[int]]) -> Set[int]:
    out: Set[int] = set()
    stack = list(hids)
    while stack:
        cur = stack.pop()
        for c in children_of.get(cur, []):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def _compute_exclusive_memberships(
    ahf_members: Dict[int, ParticleMembership],
    children_of: Dict[int, List[int]],
    needed: Set[int],
) -> Dict[int, ParticleMembership]:
    """Compute exclusive particle memberships E(h) for needed AHF nodes.

    E(h) = M(h) - union(E(children(h))) computed bottom-up.
    """
    exclusives: Dict[int, ParticleMembership] = {}

    def postorder(h: int) -> ParticleMembership:
        if h in exclusives:
            return exclusives[h]
        M = ahf_members.get(h)
        if M is None:
            # No data; return empty
            exclusives[h] = ParticleMembership(h)
            return exclusives[h]
        # Copy raw sets
        e0 = set(M.parttype0)
        e1 = set(M.parttype1)
        e4 = set(M.parttype4)
        e5 = set(M.parttype5)
        # subtract children exclusives
        for c in children_of.get(h, []):
            if c not in needed:
                continue
            Ec = postorder(c)
            if Ec.parttype0:
                e0.difference_update(Ec.parttype0)
            if Ec.parttype1:
                e1.difference_update(Ec.parttype1)
            if Ec.parttype4:
                e4.difference_update(Ec.parttype4)
            if Ec.parttype5:
                e5.difference_update(Ec.parttype5)
        exclusives[h] = ParticleMembership(id=h, parttype0=e0, parttype1=e1, parttype4=e4, parttype5=e5)
        return exclusives[h]

    # Traverse all needed nodes
    for h in list(needed):
        postorder(h)
    return exclusives


def _galaxies_to_namedata_from_group_list(galaxy_list, star_particle_ids) -> List[ParticleMembership]:
    """Like galaxies_to_namedata but expects CAESAR Group objects (galaxy_list)."""
    membership: List[ParticleMembership] = []
    for gal in galaxy_list:
        slist = getattr(gal, 'slist', [])
        ids = set(int(star_particle_ids[i]) for i in slist)
        gid = int(getattr(gal, 'GroupID', getattr(gal, 'id', 0)))
        membership.append(ParticleMembership(id=gid, parttype4=ids))
    return membership


def _pid_to_index_map(arr: np.ndarray) -> Dict[int, int]:
    return {int(pid): int(i) for i, pid in enumerate(arr.tolist())}


def integrate_ahf_match_prune_inplace(sim, ahf_particles_file: str) -> None:
    """Integrate AHF matching per your spec, then prune and reassign.

    Matching policy:
    - For each CAESAR galaxy, record all AHF nodes whose star overlap > 50%.
    - After collecting candidates, choose the lowest-level (deepest) node.
    - Then compute exclusives and reassign particle lists from that node.

    Honors ``sim.nproc`` for parallel selection across galaxies.
    """
    from caesar.property_manager import get_property, has_ptype

    # Must have a galaxy list already
    if not hasattr(sim, 'galaxy_list') or len(sim.galaxy_list) == 0:
        return

    # Resolve n_jobs
    n_jobs = getattr(sim, 'nproc', None)
    try:
        n_jobs = int(n_jobs) if n_jobs is not None else 1
    except Exception:
        n_jobs = 1

    # Star PID array and PID->index map
    star_ids = get_property(sim, 'pid', 'star').d.astype(np.int64)
    pid_to_star_index: Dict[int, int] = _pid_to_index_map(star_ids)

    # Build star_index (full per-type) -> galaxy_index map and per-galaxy star counts.
    # Convert CAESAR-selected star indices (gal.slist) to full per-type star indices via
    # the concatenated-index mapping in DataManager (slist -> concatenated -> indexes -> full).
    nstar = len(star_ids)
    staridx_to_galidx = np.full(nstar, -1, dtype=np.int32)
    galaxy_star_counts = np.zeros(len(sim.galaxy_list), dtype=np.int64)
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
            galaxy_star_counts[gi] = sl.size
        except Exception:
            # If mapping fails for any reason, skip this galaxy for matching
            continue

    # Read AHF particles file once and tally per-galaxy AHF star overlaps
    from collections import defaultdict
    gal_to_counts: List[Dict[int, int]] = [defaultdict(int) for _ in range(len(sim.galaxy_list))]

    def _open_particles(path: str):
        if path.endswith('.gz'):
            import gzip
            return gzip.open(path, 'rt')
        return open(path, 'r')

    with _open_particles(ahf_particles_file) as f:
        current_hid = None
        remaining = 0
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if remaining == 0 and len(parts) == 2:
                # Header line: "npart hid"
                try:
                    remaining = int(parts[0])
                    current_hid = int(parts[1])
                except Exception:
                    current_hid = None
                    remaining = 0
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
                # Only consider stars for galaxy matching
                if ptype != 4 or current_hid is None:
                    continue
                si = pid_to_star_index.get(pid)
                if si is None:
                    continue
                gi = int(staridx_to_galidx[si])
                if gi < 0:
                    continue
                gal_to_counts[gi][current_hid] += 1

    # Load AHF hierarchy to support lowest-level selection and exclusives
    parent_of, children_of = _read_ahf_hierarchy(ahf_particles_file)

    # Cache depths for speed
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

    # Choose selected AHF node per galaxy
    indices = list(range(len(sim.galaxy_list)))
    def _select_for_gal(gi: int) -> int:
        counts = gal_to_counts[gi]
        total = int(galaxy_star_counts[gi])
        if total <= 0 or not counts:
            return -1
        thresh = 0.5 * total
        # Candidates > 50%
        cands = [hid for hid, c in counts.items() if c > thresh]
        if not cands:
            # Fallback: best by count
            hid = max(counts.items(), key=lambda kv: kv[1])[0]
            return int(hid)
        # Pick lowest-level (max depth); tie-break by count then id
        cands.sort(key=lambda h: (depth(h), counts[h], h))
        return int(cands[-1])

    selected: List[int]
    if n_jobs is not None and n_jobs > 1:
        try:
            from joblib import Parallel, delayed
            selected = Parallel(n_jobs=n_jobs, backend='loky')(delayed(_select_for_gal)(gi) for gi in indices)
        except Exception:
            selected = [_select_for_gal(gi) for gi in indices]
    else:
        selected = [_select_for_gal(gi) for gi in indices]

    # Build mapping from selected AHF ID -> list of galaxy indices
    from collections import defaultdict as _dd
    mapping: Dict[int, List[int]] = _dd(list)
    for gi, hid in enumerate(selected):
        if hid is not None and int(hid) != -1:
            mapping[int(hid)].append(gi)

    if not mapping:
        return

    matched_ids = set(mapping.keys())
    # Need all ancestors of matched nodes and all their descendants for pruning
    ancestors = _collect_ancestors(matched_ids, parent_of)
    descendants = _collect_descendants(matched_ids, children_of)
    needed_ids = matched_ids | ancestors | descendants

    # Load memberships only for needed IDs to save time/memory
    def _read_needed_memberships(path: str, needed: Set[int]) -> Dict[int, ParticleMembership]:
        out: Dict[int, ParticleMembership] = {}
        with _open_particles(path) as f:
            current_hid = None
            remaining = 0
            cur_pm: Optional[ParticleMembership] = None
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if remaining == 0 and len(parts) == 2:
                    # finalize previous if any
                    if cur_pm is not None and cur_pm.id in needed:
                        out[cur_pm.id] = cur_pm
                    # new header
                    try:
                        remaining = int(parts[0])
                        current_hid = int(parts[1])
                    except Exception:
                        current_hid = None
                        remaining = 0
                        cur_pm = None
                        continue
                    cur_pm = ParticleMembership(current_hid) if current_hid in needed else None
                    continue
                if remaining > 0:
                    remaining -= 1
                    if cur_pm is None:
                        continue
                    pparts = line.split('\t')
                    if len(pparts) != 2:
                        continue
                    try:
                        pid = int(pparts[0])
                        ptype = int(pparts[1])
                    except Exception:
                        continue
                    if ptype == 0:
                        cur_pm.parttype0.add(pid)
                    elif ptype == 1:
                        cur_pm.parttype1.add(pid)
                    elif ptype == 4:
                        cur_pm.parttype4.add(pid)
                    elif ptype == 5:
                        cur_pm.parttype5.add(pid)
            # finalize last
            if cur_pm is not None and cur_pm.id in needed:
                out[cur_pm.id] = cur_pm
        return out

    ahf_by_id = _read_needed_memberships(ahf_particles_file, needed_ids)

    # Compute exclusives bottom-up for needed nodes (for baryons)
    exclusives = _compute_exclusive_memberships(ahf_by_id, children_of, needed_ids)

    # Build PID->selected-index maps for each ptype present (for reassignment)
    pid_maps_sel: Dict[str, Dict[int, int]] = _build_selected_pid_maps(sim)

    # Also build full-snapshot PID->index maps for DM to construct the exclusive reverse map
    ndm_full = 0
    pid_to_dm_fullidx: Dict[int, int] = {}
    if has_ptype(sim, 'dm'):
        dm_pids_full = get_property(sim, 'pid', 'dm').d.astype(np.int64)
        ndm_full = len(dm_pids_full)
        if ndm_full > 0:
            pid_to_dm_fullidx = {int(pid): int(i) for i, pid in enumerate(dm_pids_full.tolist())}

    # Helper to map PID sets to selected index arrays
    def map_set(pidset: Set[int], key: str) -> np.ndarray:
        mp = pid_maps_sel.get(key, {})
        if not mp or not pidset:
            return np.array([], dtype=np.int32)
        arr = np.fromiter((mp[pid] for pid in pidset if pid in mp), dtype=np.int32)
        if arr.size == 0:
            return arr
        return np.unique(arr)

    # Exclusive DM reverse map for global list reconstruction
    exclusive_gal_dm = None
    if ndm_full > 0:
        exclusive_gal_dm = np.full(ndm_full, -1, dtype=np.int32)

    # Merge galaxies per AHF ID and overwrite particle lists.
    # Stars/gas/BH use exclusives; DM uses the inclusive membership of the
    # selected AHF node.  This step can be expensive, so parallelise across
    # matched galaxies when possible.
    to_remove: Set[int] = set()

    tasks = []
    for ahf_id, gal_indices in mapping.items():
        if ahf_id not in exclusives:
            continue
        indices = sorted(set(gal_indices))
        if not indices:
            continue
        base_idx = indices[0]
        base_gid = int(getattr(sim.galaxy_list[base_idx], 'GroupID', -1))
        tasks.append((ahf_id, indices, base_idx, base_gid))

    def process_matching(task):
        ahf_id, indices, base_idx, base_gid = task
        e = exclusives.get(ahf_id)
        if e is None:
            return None

        dm_pm = ahf_by_id.get(ahf_id)
        dm_set: Set[int] = dm_pm.parttype1 if dm_pm is not None else set()

        slist_arr = map_set(e.parttype4, 'star')
        glist_arr = map_set(e.parttype0, 'gas')
        bh_arr = map_set(e.parttype5, 'bh') if 'bh' in pid_maps_sel else np.array([], dtype=np.int32)
        dm_arr = map_set(dm_set, 'dm')

        dm_exclusive_idx: Optional[np.ndarray] = None
        if exclusive_gal_dm is not None and e.parttype1 and pid_to_dm_fullidx:
            idx_list = [pid_to_dm_fullidx[pid] for pid in e.parttype1 if pid in pid_to_dm_fullidx]
            if idx_list:
                dm_exclusive_idx = np.asarray(idx_list, dtype=np.int64)

        return (base_idx, indices, base_gid, slist_arr, glist_arr, bh_arr, dm_arr, dm_exclusive_idx)

    results: List[Optional[Tuple[int, List[int], int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]] = []
    if tasks:
        n_jobs = getattr(sim, 'nproc', 1)
        try:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=max(1, int(n_jobs)), backend='threading')(
                delayed(process_matching)(task) for task in tasks
            )
        except Exception:
            results = [process_matching(task) for task in tasks]
    else:
        results = []

    for item in results:
        if item is None:
            continue
        base_idx, indices, base_gid, slist_arr, glist_arr, bh_arr, dm_arr, dm_exclusive_idx = item
        base = sim.galaxy_list[base_idx]

        base.slist = slist_arr
        base.glist = glist_arr
        if 'bh' in pid_maps_sel:
            if hasattr(base, 'bhlist'):
                base.bhlist = bh_arr
            else:
                setattr(base, 'bhlist', bh_arr)
        base.dmlist = dm_arr

        gi = []
        try:
            if glist_arr.size > 0:
                gi.append(sim.data_manager.glist[glist_arr])
        except Exception:
            pass
        try:
            if slist_arr.size > 0:
                gi.append(sim.data_manager.slist[slist_arr])
        except Exception:
            pass
        try:
            if dm_arr.size > 0 and has_ptype(sim, 'dm'):
                gi.append(sim.data_manager.dmlist[dm_arr])
        except Exception:
            pass
        try:
            if 'bh' in pid_maps_sel and bh_arr.size > 0 and hasattr(sim.data_manager, 'bhlist'):
                gi.append(sim.data_manager.bhlist[bh_arr])
        except Exception:
            pass
        try:
            if hasattr(base, 'dlist') and base.dlist is not None and len(base.dlist) > 0:
                gi.append(sim.data_manager.dlist[base.dlist])
        except Exception:
            pass
        base.global_indexes = np.concatenate(gi).astype(np.int64) if gi else np.array([], dtype=np.int64)

        if exclusive_gal_dm is not None and dm_exclusive_idx is not None and dm_exclusive_idx.size > 0:
            exclusive_gal_dm[dm_exclusive_idx] = base_gid

        for gi in indices[1:]:
            to_remove.add(gi)

    if to_remove:
        # Build new galaxy list and renumber GroupIDs
        new_list = []
        for i, g in enumerate(sim.galaxy_list):
            if i in to_remove:
                continue
            new_list.append(g)
        # Renumber GroupIDs to be sequential
        for new_id, g in enumerate(new_list):
            g.GroupID = new_id
        sim.galaxy_list = new_list
        sim.ngalaxies = len(new_list)

    # Do not recompute properties here; caller (member_search flow) will handle it
    # Stash exclusive DM reverse map for global list construction
    if exclusive_gal_dm is not None:
        setattr(sim, '_exclusive_galaxy_dmlist', exclusive_gal_dm)
