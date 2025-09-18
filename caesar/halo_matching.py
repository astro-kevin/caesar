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
            pparts = pline.split("\t")
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
) -> List[Tuple[int, int]]:
    """Match halos by overlapping ``parttype4`` particle IDs.

    Builds a star->AHF lookup once, then counts overlaps per-galaxy using
    NumPy (optionally threaded with joblib). Falls back to a serial
    iterator with ``tqdm`` progress if joblib is unavailable or ``n_jobs``
    is ``None``/``1``.
    """

    n_list1 = len(list1)
    if n_list1 == 0:
        return []
    if len(list2) == 0:
        return [(pm.id, -1) for pm in list1]

    list1_ids = np.fromiter((pm.id for pm in list1), dtype=np.int64, count=n_list1)
    galaxy_stars: List[np.ndarray] = []
    galaxy_sizes = np.empty(n_list1, dtype=np.int64)
    for i, pm in enumerate(list1):
        count = len(pm.parttype4)
        if count:
            arr = np.fromiter(pm.parttype4, dtype=np.int64, count=count)
            arr.sort()
        else:
            arr = np.empty(0, dtype=np.int64)
        galaxy_stars.append(arr)
        galaxy_sizes[i] = arr.size

    from collections import defaultdict

    star_to_halos: Dict[int, np.ndarray] = {}
    _accumulator: Dict[int, List[int]] = defaultdict(list)
    for pm in list2:
        if not pm.parttype4:
            continue
        hid = pm.id
        for pid in pm.parttype4:
            _accumulator[pid].append(hid)
    for pid, entries in _accumulator.items():
        star_to_halos[pid] = np.fromiter(entries, dtype=np.int64, count=len(entries))
    _accumulator.clear()

    def match_one(idx: int) -> Tuple[int, int]:
        stars = galaxy_stars[idx]
        if stars.size == 0:
            return (int(list1_ids[idx]), -1)
        candidate_lists: List[np.ndarray] = []
        for pid in stars:
            arr = star_to_halos.get(int(pid))
            if arr is not None:
                candidate_lists.append(arr)
        if not candidate_lists:
            return (int(list1_ids[idx]), -1)
        candidates = candidate_lists[0] if len(candidate_lists) == 1 else np.concatenate(candidate_lists)
        if candidates.size == 0:
            return (int(list1_ids[idx]), -1)
        uniq, counts = np.unique(candidates, return_counts=True)
        if uniq.size == 0:
            return (int(list1_ids[idx]), -1)
        best_pos = int(np.argmax(counts))
        best_count = counts[best_pos]
        if best_count == 0:
            return (int(list1_ids[idx]), -1)
        return (int(list1_ids[idx]), int(uniq[best_pos]))

    if n_jobs is None or n_jobs <= 1:
        return [match_one(i) for i in tqdm(range(n_list1), desc="Matching Progress")]

    try:
        from joblib import Parallel, delayed

        return Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(match_one)(i) for i in range(n_list1)
        )
    except Exception:
        return [match_one(i) for i in tqdm(range(n_list1), desc="Matching Progress")]

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


def _iter_memberships_stream(path: str, needed: Set[int]) -> Iterator[ParticleMembership]:
    """Yield memberships for nodes in ``needed`` while streaming the file."""

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
                elif ptype == 4:
                    cur_pm.parttype4.add(pid)
                elif ptype == 5:
                    cur_pm.parttype5.add(pid)

        if cur_pm is not None and cur_pm.id in needed:
            yield cur_pm


def _read_memberships_for_ids(
    path: str,
    needed: Set[int],
    *,
    load_dm: bool = True,
) -> Dict[int, ParticleMembership]:
    """Read AHF particle memberships only for the given IDs (supports .gz)."""
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
    selected_index_to_pid: Optional[List[Optional[int]]] = None
    if dm_pid_map:
        max_idx = max(dm_pid_map.values(), default=-1)
        selected_index_to_pid = [None] * (max_idx + 1)
        for pid, idx in dm_pid_map.items():
            if idx >= len(selected_index_to_pid):
                selected_index_to_pid.extend([None] * (idx + 1 - len(selected_index_to_pid)))
            selected_index_to_pid[idx] = int(pid)

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
    dm_nodes_inclusive: Dict[int, Set[int]] = {}
    dm_nodes_exclusive: Dict[int, Set[int]] = {}

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
            return

        # Ensure every node has a membership object (possibly empty)
        for node in nodes_for_host:
            bucket.setdefault(node, ParticleMembership(node))

        exclusives = _compute_exclusive_memberships(bucket, children_of, nodes_for_host)

        eligible = [n for n in nodes_for_host if len(exclusives.get(n, ParticleMembership(n)).parttype4) >= min_stars]
        if not eligible:
            return

        central = max(eligible, key=lambda n: (len(exclusives[n].parttype4), n))

        deposit_star: Set[int] = set()
        deposit_gas: Set[int] = set()
        deposit_bh: Set[int] = set()
        deposit_dm_nodes: Set[int] = set()

        for node in nodes_for_host:
            if node == central:
                continue
            stars = exclusives.get(node, ParticleMembership(node)).parttype4
            if len(stars) >= min_stars:
                continue
            if parent_of.get(node, 0) not in (None, 0):
                ex = exclusives.get(node, ParticleMembership(node))
                deposit_star |= ex.parttype4
                deposit_gas |= ex.parttype0
                deposit_bh |= ex.parttype5
                deposit_dm_nodes.add(node)

        cen_ex = exclusives[central]
        payloads: List[Tuple[ParticleMembership, Set[int], Set[int]]] = []

        cen_star = set(cen_ex.parttype4) | deposit_star
        cen_gas = set(cen_ex.parttype0) | deposit_gas
        cen_bh = set(cen_ex.parttype5) | deposit_bh
        central_payload = (
            ParticleMembership(
                central,
                parttype0=cen_gas,
                parttype4=cen_star,
                parttype5=cen_bh,
            ),
            {central} | deposit_dm_nodes,
            {central},
        )
        payloads.append(central_payload)

        for node in eligible:
            if node == central:
                continue
            ex = exclusives[node]
            payloads.append(
                (
                    ex,
                    {node},
                    {node},
                )
            )

        def build_group(payload: Tuple[ParticleMembership, Set[int], Set[int]]):
            pm, dm_inc, dm_exc = payload
            grp = create_new_group(sim, 'galaxy')
            grp.slist = map_sel(pm.parttype4, 'star')
            grp.glist = map_sel(pm.parttype0, 'gas')
            if 'bh' in pid_maps_sel:
                grp.bhlist = map_sel(pm.parttype5, 'bh')
            grp.global_indexes = np.array([], dtype=np.int64)
            return grp, dm_inc, dm_exc

        results: List[Tuple] = []
        try:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=jobs, backend='threading')(
                delayed(build_group)(payload) for payload in payloads
            )
        except Exception:
            results = [build_group(payload) for payload in payloads]

        base_index = len(galaxies)
        for offset, (grp, dm_inc, dm_exc) in enumerate(results):
            galaxies.append(grp)
            idx = base_index + offset
            dm_nodes_inclusive[idx] = set(dm_inc)
            dm_nodes_exclusive[idx] = set(dm_exc)

    for pm in _iter_memberships_stream(ahf_particles_file, all_needed_nodes):
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

    sim.galaxy_list = galaxies
    sim.ngalaxies = len(galaxies)
    setattr(sim, "_ahf_matched", True)
    setattr(sim, "_include_dm_in_galaxies", True)
    if sim.ngalaxies == 0:
        return

    def _collect_dm_index_lists(
        inclusive_nodes: Dict[int, Set[int]],
        exclusive_nodes: Dict[int, Set[int]],
        dm_pid_to_sel: Dict[int, int],
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        if not dm_pid_to_sel or not inclusive_nodes:
            return {}, {}
        node_to_gal_inclusive: Dict[int, List[int]] = {}
        for gi, ids in inclusive_nodes.items():
            for hid in ids:
                node_to_gal_inclusive.setdefault(hid, []).append(gi)
        node_to_gal_exclusive: Dict[int, List[int]] = {}
        for gi, ids in exclusive_nodes.items():
            for hid in ids:
                node_to_gal_exclusive.setdefault(hid, []).append(gi)

        inclusive_lists: Dict[int, List[int]] = {gi: [] for gi in inclusive_nodes}
        exclusive_lists: Dict[int, List[int]] = {gi: [] for gi in exclusive_nodes}

        with _open_ahf_particles(ahf_particles_file) as fh:
            current_hid = None
            remaining = 0
            for raw in fh:
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
                    continue
                if remaining > 0:
                    remaining -= 1
                    if current_hid is None:
                        continue
                    pparts = line.split('\t')
                    if len(pparts) != 2:
                        continue
                    try:
                        pid = int(pparts[0])
                        ptype = int(pparts[1])
                    except Exception:
                        continue
                    if ptype != 1:
                        continue
                    sel_idx = dm_pid_to_sel.get(pid)
                    if sel_idx is None:
                        continue
                    for gi in node_to_gal_inclusive.get(current_hid, []):
                        inclusive_lists.setdefault(gi, []).append(sel_idx)
                    for gi in node_to_gal_exclusive.get(current_hid, []):
                        exclusive_lists.setdefault(gi, []).append(sel_idx)
        return inclusive_lists, exclusive_lists

    dm_inclusive_indices, dm_exclusive_indices = _collect_dm_index_lists(
        dm_nodes_inclusive, dm_nodes_exclusive, dm_pid_map
    )

    for gi, gal in enumerate(sim.galaxy_list):
        sel = dm_inclusive_indices.get(gi, [])
        if sel:
            sel_arr = np.unique(np.asarray(sel, dtype=np.int64))
            gal.dmlist = sel_arr.astype(np.int32)
        else:
            gal.dmlist = np.array([], dtype=np.int32)

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

    ctx = _Ctx(sim)
    _get_group_properties(ctx, sim.galaxy_list)

    try:
        if 'galaxy' not in sim.group_types:
            sim.group_types.append('galaxy')
    except Exception:
        pass

    if ndm_full > 0 and dm_pid_map and selected_index_to_pid is not None:
        exclusive_gal_dm = np.full(ndm_full, -1, dtype=np.int32)
        for gi, gal in enumerate(sim.galaxy_list):
            sel = dm_exclusive_indices.get(gi)
            if not sel:
                continue
            sel_arr = np.unique(np.asarray(sel, dtype=np.int64))
            full_indices: List[int] = []
            for idx in sel_arr:
                if idx < 0 or idx >= len(selected_index_to_pid):
                    continue
                pid = selected_index_to_pid[idx]
                if pid is None:
                    continue
                mapped = pid_to_dm_fullidx.get(pid)
                if mapped is not None:
                    full_indices.append(mapped)
            if full_indices:
                exclusive_gal_dm[np.asarray(full_indices, dtype=np.int64)] = int(gal.GroupID)
        setattr(sim, '_exclusive_galaxy_dmlist', exclusive_gal_dm)


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
                    pparts = pline.split("\t")
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
    matches = find_best_matches(caesar_data, ahf_data)

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
                pparts = line.split('\t')
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
    def map_set(pidset: Set[int], key: str) -> List[int]:
        mp = pid_maps_sel.get(key, {})
        if not mp or not pidset:
            return []
        return [mp[pid] for pid in pidset if pid in mp]

    # Exclusive DM reverse map for global list reconstruction
    exclusive_gal_dm = None
    if ndm_full > 0:
        exclusive_gal_dm = np.full(ndm_full, -1, dtype=np.int32)

    # Merge galaxies per AHF ID and overwrite particle lists
    # - Stars/gas/BH: use exclusives
    # - DM: use inclusive membership (selected node + all its subhalos already included in AHF block)
    to_remove: Set[int] = set()
    for ahf_id, gal_indices in mapping.items():
        if ahf_id not in exclusives:
            continue
        e = exclusives[ahf_id]
        gal_indices = sorted(set(gal_indices))
        base_i = gal_indices[0]
        base = sim.galaxy_list[base_i]

        # Overwrite lists from exclusives (baryons)
        base.slist = np.array(map_set(e.parttype4, 'star'), dtype=np.int32)
        base.glist = np.array(map_set(e.parttype0, 'gas'), dtype=np.int32) if hasattr(base, 'glist') else np.array(map_set(e.parttype0, 'gas'), dtype=np.int32)
        if 'bh' in pid_maps_sel:
            base.bhlist = np.array(map_set(e.parttype5, 'bh'), dtype=np.int32)

        # DM: use inclusive membership from the selected AHF node (AHF block includes subhalos)
        dm_pm = ahf_by_id.get(ahf_id)
        dm_set: Set[int] = set()
        if dm_pm is not None:
            dm_set = dm_pm.parttype1
        base.dmlist = np.array(map_set(dm_set, 'dm'), dtype=np.int32)
        # Populate exclusive reverse map for this galaxy (in full-snapshot DM index space)
        if exclusive_gal_dm is not None and e.parttype1:
            if pid_to_dm_fullidx:
                ex_full = [pid_to_dm_fullidx[pid] for pid in e.parttype1 if pid in pid_to_dm_fullidx]
                if ex_full:
                    exclusive_gal_dm[np.array(ex_full, dtype=np.int64)] = int(base.GroupID)

        # Rebuild global_indexes to include current per-type lists (so overall properties see DM when enabled)
        gi = []
        try:
            # gas
            if hasattr(base, 'glist') and base.glist is not None and len(base.glist) > 0:
                gi.append(sim.data_manager.glist[base.glist])
        except Exception:
            pass
        try:
            # stars
            if hasattr(base, 'slist') and base.slist is not None and len(base.slist) > 0:
                gi.append(sim.data_manager.slist[base.slist])
        except Exception:
            pass
        try:
            # dm
            if hasattr(base, 'dmlist') and base.dmlist is not None and len(base.dmlist) > 0 and has_ptype(sim, 'dm'):
                gi.append(sim.data_manager.dmlist[base.dmlist])
        except Exception:
            pass
        try:
            # bh
            if hasattr(base, 'bhlist') and base.bhlist is not None and len(base.bhlist) > 0 and 'bh' in pid_maps_sel:
                gi.append(sim.data_manager.bhlist[base.bhlist])
        except Exception:
            pass
        try:
            # dust
            if hasattr(base, 'dlist') and base.dlist is not None and len(base.dlist) > 0:
                gi.append(sim.data_manager.dlist[base.dlist])
        except Exception:
            pass
        if gi:
            base.global_indexes = np.concatenate(gi).astype(np.int64)

        # Mark other matched galaxies for removal
        for gi in gal_indices[1:]:
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
