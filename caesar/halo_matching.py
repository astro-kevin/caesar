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

from bidict import bidict

from yt.funcs import mylog

from caesar.group import create_new_group

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


def _update_ahf_galaxy_maps(sim, ids: Optional[Iterable]) -> None:
    """Cache galaxy AHF ID lookups for O(1) access."""

    if ids is None:
        ids_iter = list(getattr(sim, '_ahf_galaxy_ahf_ids', []))
    else:
        ids_iter = list(ids)
        sim._ahf_galaxy_ahf_ids = list(ids_iter)

    gal_map = bidict()

    for idx, hid in enumerate(ids_iter):
        try:
            hid_val = int(hid)
        except Exception:
            hid_val = None
        if hid_val is not None and hid_val < 0:
            hid_val = None
        if hid_val is not None:
            gal_map[hid_val] = idx

    sim._ahf_galaxy_map = gal_map
    sim._ahf_galaxy_id_to_index = gal_map
    sim._ahf_galaxy_index_to_id = gal_map.inv


def build_halos_from_ahf(sim, ahf_particles_file: str, *, full_particle_load: bool = False):
    """Populate ``sim.halo_list`` directly from an AHF catalogue.

    For AHF-driven runs we bypass the 6D-FOF halo finder and seed the
    CAESAR halo structures from the external membership file instead.  All
    non-AHF modes continue to use the existing fof6d path.
    """

    if not ahf_particles_file:
        raise ValueError("AHF particles file must be provided when haloid='AHF'.")

    from yt.funcs import mylog

    from caesar.fof6d import fof6d
    from caesar.fubar import get_mean_interparticle_separation
    from caesar.group import get_group_properties

    halos = fof6d(sim, 'halo')

    halos.MIS = get_mean_interparticle_separation(sim).d
    halos.load_haloid()

    if full_particle_load:
        halos.keep_all_groups = True
    haloid_mode = str(sim._kwargs.get('haloid', '')).upper() if hasattr(sim, '_kwargs') else ''
    if haloid_mode == 'AHF-FAST':
        select_arg = halos.haloid
    else:
        select_arg = halos.haloid

    sim.data_manager._member_search_init(select=select_arg)

    if isinstance(halos.haloid, dict):
        flattened: List[np.ndarray] = []
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

    halos.keep_all_groups = getattr(halos, 'keep_all_groups', False)
    halos.load_lists()
    if hasattr(halos, 'keep_all_groups'):
        delattr(halos, 'keep_all_groups')
    if len(sim.halo_list) == 0:
        mylog.warning('No valid halos found! Aborting member search')
        return None

    get_group_properties(halos, sim.halo_list)

    _update_ahf_halo_maps(sim)

    if 'halo' not in sim.group_types:
        sim.group_types.append('halo')
    sim.halos = sim.halo_list
    sim.nhalos = len(sim.halo_list)

    return halos


def _prune_halos_after_galaxies(sim) -> None:
    """Remove halos that end up with no galaxies after the AHF reconciliation."""

    if not hasattr(sim, 'halo_list') or not sim.halo_list:
        return

    keep: List = []
    keep_indices: List[int] = []

    for idx, halo in enumerate(sim.halo_list):
        galaxy_indices = list(getattr(halo, 'galaxy_index_list', []))
        if len(galaxy_indices) == 0:
            continue
        keep.append(halo)
        keep_indices.append(idx)

    if len(keep) == len(sim.halo_list):
        # nothing pruned
        return

    old_to_new = {old: new for new, old in enumerate(keep_indices)}

    for gal in getattr(sim, 'galaxy_list', []):
        old_host = getattr(gal, 'parent_halo_index', -1)
        gal.parent_halo_index = old_to_new.get(old_host, -1)

    for new_idx, halo in enumerate(keep):
        halo.GroupID = new_idx
        halo.galaxy_index_list = [gi for gi in getattr(halo, 'galaxy_index_list', []) if getattr(sim.galaxy_list[gi], 'parent_halo_index', -1) == new_idx]

    if hasattr(sim, '_ahf_galaxy_hosts'):
        sim._ahf_galaxy_hosts = [old_to_new.get(h, -1) if h is not None and h >= 0 else -1 for h in sim._ahf_galaxy_hosts]

    sim.halo_list = keep
    sim.halos = keep
    sim.nhalos = len(keep)

    _update_ahf_halo_maps(sim)

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
            best_overlap = -1
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
            if best_idx == -1 and nh > 0:
                best_idx = 0
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


def _ensure_missing_ahf_halos(
    sim,
    missing_ids: Set[int],
    ahf_particles_file: str,
    pid_maps_sel: Dict[str, Dict[int, int]],
) -> Dict[int, int]:
    """Guarantee that each requested AHF halo ID has a CAESAR halo entry.

    Returns a map of AHF haloID -> newly created halo index for any halos that
    were synthesized. Existing halos are ignored.
    """

    needed = {int(h) for h in missing_ids if h is not None and int(h) >= 0}
    if not needed:
        return {}

    existing = {
        int(getattr(halo, 'AHF_haloID', -1))
        for halo in getattr(sim, 'halo_list', [])
        if getattr(halo, 'AHF_haloID', None) is not None
    }
    pending = {hid for hid in needed if hid not in existing}
    if not pending:
        return {}

    memberships = _read_memberships_for_ids(ahf_particles_file, pending, load_dm=True)
    if not memberships:
        return {}

    created: Dict[int, int] = {}

    def _map_pidset(pidset: Iterable[int], key: str) -> np.ndarray:
        mapping = pid_maps_sel.get(key, {})
        if not mapping or not pidset:
            return np.empty(0, dtype=np.int64)
        idx = [mapping.get(int(pid)) for pid in pidset if int(pid) in mapping]
        if not idx:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.array(idx, dtype=np.int64))

    def _concat_indices(ptype: str, sel_idx: np.ndarray) -> np.ndarray:
        if sel_idx.size == 0:
            return np.empty(0, dtype=np.int64)
        try:
            concat = sim.data_manager.selected_to_concat(ptype, sel_idx)
            return np.asarray(concat, dtype=np.int64)
        except Exception:
            return np.empty(0, dtype=np.int64)

    dataset = getattr(sim, 'yt_dataset', None)
    units = getattr(sim, 'units', {})
    mass_unit = units.get('mass', None)

    def _to_quan(val: float):
        if dataset is not None and mass_unit is not None:
            try:
                return dataset.quan(val, mass_unit)
            except Exception:
                pass
        return val

    def _mass_from_concat(concat_idx: np.ndarray) -> float:
        if concat_idx.size == 0:
            return 0.0
        return float(np.sum(sim.data_manager.mass[concat_idx]))

    for hid in sorted(pending):
        pm = memberships.get(hid)
        if pm is None:
            continue

        halo = create_new_group(sim, 'halo')
        halo.obj_type = 'halo'
        halo.AHF_haloID = int(hid)

        gas_sel = _map_pidset(pm.parttype0, 'gas')
        star_sel = _map_pidset(pm.parttype4, 'star')
        bh_sel = _map_pidset(pm.parttype5, 'bh')
        dm_sel = _map_pidset(pm.parttype1, 'dm')

        halo.glist = gas_sel
        halo.ngas = gas_sel.size
        halo.slist = star_sel
        halo.nstar = star_sel.size
        halo.bhlist = bh_sel
        halo.nbh = bh_sel.size
        halo.dmlist = dm_sel
        halo.ndm = dm_sel.size

        concat_parts = []
        gas_concat = _concat_indices('gas', gas_sel)
        if gas_concat.size:
            concat_parts.append(gas_concat)
        star_concat = _concat_indices('star', star_sel)
        if star_concat.size:
            concat_parts.append(star_concat)
        bh_concat = _concat_indices('bh', bh_sel)
        if bh_concat.size:
            concat_parts.append(bh_concat)
        dm_concat = _concat_indices('dm', dm_sel)
        if dm_concat.size:
            concat_parts.append(dm_concat)

        halo.global_indexes = (
            np.sort(np.concatenate(concat_parts)).astype(np.int64)
            if concat_parts
            else np.empty(0, dtype=np.int64)
        )

        halo.galaxy_index_list = []
        halo._forced_include = True

        mass_gas = _mass_from_concat(gas_concat)
        mass_star = _mass_from_concat(star_concat)
        mass_bh = _mass_from_concat(bh_concat)
        mass_dm = _mass_from_concat(dm_concat)
        total_mass = mass_gas + mass_star + mass_bh + mass_dm

        halo.masses['gas'] = _to_quan(mass_gas)
        halo.masses['stellar'] = _to_quan(mass_star)
        halo.masses['bh'] = _to_quan(mass_bh)
        halo.masses['dm'] = _to_quan(mass_dm)
        halo.masses['baryon'] = _to_quan(mass_gas + mass_star + mass_bh)
        halo.masses['total'] = _to_quan(total_mass)

        halo.GroupID = len(sim.halo_list)
        sim.halo_list.append(halo)
        created[hid] = halo.GroupID

    if created:
        try:
            from types import SimpleNamespace
            from caesar.group import get_group_properties
            new_halos = [sim.halo_list[idx] for idx in created.values()]
            valid_halos = [halo for halo in new_halos if halo is not None]
            if valid_halos:
                nparttot = sum(len(getattr(halo, 'global_indexes', [])) for halo in valid_halos)
                mapping = {}
                dm = sim.data_manager
                for ptype in getattr(dm, 'ptypes', []):
                    if ptype == 'gas':
                        attr = 'glist'
                    elif ptype == 'star':
                        attr = 'slist'
                    elif ptype == 'bh':
                        attr = 'bhlist'
                    elif ptype == 'dust':
                        attr = 'dlist'
                    elif ptype == 'dm':
                        attr = 'dmlist'
                    elif ptype == 'dm2':
                        attr = 'dm2list'
                    elif ptype == 'dm3':
                        attr = 'dm3list'
                    else:
                        continue
                    mapping[ptype] = len(getattr(dm, attr, []))
                context = SimpleNamespace(
                    obj=sim,
                    obj_type='halo',
                    nproc=int(getattr(sim, 'nproc', 1)),
                    nparttot=max(int(nparttot), 0),
                    nparttype=mapping,
                )
                original_ids = {halo: halo.GroupID for halo in valid_halos}
                get_group_properties(context, valid_halos)
                for halo in valid_halos:
                    halo.GroupID = original_ids.get(halo, halo.GroupID)
        except Exception as exc:  # pragma: no cover - defensive
            mylog.warning('Failed to recompute properties for synthesized AHF halos: %s', exc)
        sim.halos = sim.halo_list
        sim.nhalos = len(sim.halo_list)
        _update_ahf_halo_maps(sim)

    return created


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
    galaxy_node_ids: List[int] = []

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

    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED

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
            grp = create_new_group(sim, 'galaxy')
            grp.AHF_haloID = int(node_id)
            grp.slist = map_sel(star_set, 'star')
            grp.glist = map_sel(gas_set, 'gas')
            if 'bh' in pid_maps_sel:
                grp.bhlist = map_sel(bh_set, 'bh')
            if 'dm' in pid_maps_sel:
                dm_selected = map_sel(dm_inclusive, 'dm')
            else:
                dm_selected = np.array([], dtype=np.int32)
            grp.dmlist = dm_selected
            grp.global_indexes = np.array([], dtype=np.int64)
            if dm_exc:
                grp.__dict__['_dm_exclusive_pids'] = set(dm_exc)
            else:
                grp.__dict__['_dm_exclusive_pids'] = set()

            mapped_star = len(grp.slist) if hasattr(grp, 'slist') else 0
            mapped_gas = len(grp.glist) if hasattr(grp, 'glist') else 0
            mapped_bh = len(grp.bhlist) if hasattr(grp, 'bhlist') else 0
            mapped_dm = len(dm_selected)
            particle_total = mapped_star + mapped_gas + mapped_bh + mapped_dm
            if particle_total == 0:
                skipped_empty_payloads += 1
                if skipped_empty_payloads <= 10:
                    mylog.warning(
                        'AHF-FAST: node %d had %d star / %d gas / %d bh / %d dm particles '
                        'from AHF but none mapped into CAESAR selection',
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

    load_dm = bool(dm_pid_map)

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        pending_futures: Dict = {}
        host_order = 0

        for pm in _iter_memberships_stream(ahf_particles_file, all_needed_nodes, load_dm=load_dm):
            root = node_to_root.get(pm.id)
            if root is None:
                continue
            bucket = pending_members.setdefault(root, {})
            bucket[pm.id] = pm
            nodes_remaining[root] = nodes_remaining.get(root, 0) - 1
            if nodes_remaining[root] <= 0:
                bucket_copy = dict(bucket)
                future = executor.submit(process_host, host_order, root, bucket_copy)
                pending_futures[future] = host_order
                host_order += 1
                pending_members.pop(root, None)
                nodes_remaining.pop(root, None)
                flush_completed(pending_futures, block=False)

        # Drain any remaining buckets that were never submitted
        for root, bucket in list(pending_members.items()):
            bucket_copy = dict(bucket)
            future = executor.submit(process_host, host_order, root, bucket_copy)
            pending_futures[future] = host_order
            host_order += 1
            pending_members.pop(root, None)
            nodes_remaining.pop(root, None)
        flush_completed(pending_futures, block=True)

    if host_progress is not None:
        host_progress.close()

    if skipped_empty_payloads > 0:
        mylog.warning(
            'AHF-FAST: skipped %d galaxy payload(s) with no mapped particles'
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
            ahf_hid = getattr(halo, 'AHF_haloID', None)
            if ahf_hid is None:
                continue
            try:
                ahf_to_halo_index[int(ahf_hid)] = halo_index
            except Exception:
                continue

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
        for node_id in galaxy_node_ids:
            if node_id is None or node_id == -1:
                host_indices.append(-1)
                continue
            resolved = _resolve_halo_index(int(node_id))
            if resolved < 0:
                missing_hosts.add(int(node_id))
            host_indices.append(resolved)
        if missing_hosts:
            mylog.warning(
                'AHF: %d host halo(s) referenced by galaxies were not loaded; keeping parent index = -1',
                len(missing_hosts),
            )
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
            self.obj_type = 'galaxy'
            self.nproc = getattr(sim, 'nproc', 1)
            self.load_pot = getattr(sim, 'load_pot', True)
            self.nparttot = sum(len(getattr(g, 'global_indexes', [])) for g in sim.galaxy_list)
            mapping = {
                'gas': 'glist',
                'star': 'slist',
                'bh': 'bhlist',
                'dm': 'dmlist',
                'dm2': 'dm2list',
                'dm3': 'dm3list',
            }
            present = set(getattr(sim.data_manager, 'ptypes', []))
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
    merged_count = 0
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
        if len(gal_indices) > 1:
            merged_count += len(gal_indices) - 1

    # add unmatched galaxies
    for idx, gal in enumerate(sim.galaxies):
        if idx not in used:
            updated.append(gal)

    # help Python release memory held by the particle cache
    del ahf_data

    # Recalculate galaxy properties with the merged particle lists if possible
    sim.galaxies = updated
    sim.galaxy_list = updated
    sim.ngalaxies = len(updated)
    if fof_helper is not None:
        try:
            fof_helper.counts['galaxy'] = len(updated)
        except Exception:
            pass
        try:
            fof_helper.obj.galaxy_list = updated
        except Exception:
            pass
    try:
        from caesar.group import get_group_properties
        get_group_properties(sim, sim.galaxies)
    except Exception:  # pragma: no cover - optional heavy deps or incomplete sim
        return

    if merged_count > 0:
        from yt.funcs import mylog
        mylog.info('AHF: collapsed %d CAESAR galaxy(ies) into matched AHF hosts', merged_count)


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


def integrate_ahf_match_prune_inplace(sim, ahf_particles_file: str, fof_helper=None) -> None:
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
    def _select_for_gal(gi: int) -> Tuple[int, int]:
        counts = gal_to_counts[gi]
        if not counts:
            return -1, -1
        total = int(galaxy_star_counts[gi])
        thresh = 0.5 * total if total > 0 else 0

        # Determine the best candidate above the threshold (if any)
        cands = []
        if total > 0:
            cands = [hid for hid, c in counts.items() if c > thresh]
        if cands:
            cands.sort(key=lambda h: (depth(int(h)), counts[h], h))
            primary = int(cands[-1])
        else:
            primary = -1

        # Always keep track of the best-overlap node for fallback purposes
        best_hid, _ = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        best_hid = int(best_hid)

        if primary == -1 and counts.get(best_hid, 0) > 0:
            primary = best_hid

        return primary, best_hid

    selection_info: List[Tuple[int, int]]
    if n_jobs is not None and n_jobs > 1:
        from joblib import Parallel, delayed

        try:
            selection_info = Parallel(
                n_jobs=n_jobs,
                backend='threading',
                prefer='threads',
            )(delayed(_select_for_gal)(gi) for gi in indices)
        except Exception:
            selection_info = [_select_for_gal(gi) for gi in indices]
    else:
        selection_info = [_select_for_gal(gi) for gi in indices]

    selected: List[int] = []
    for primary, best in selection_info:
        if primary is not None and primary != -1:
            selected.append(int(primary))
        else:
            chosen = best if best is not None else -1
            selected.append(int(chosen) if chosen != -1 else -1)

    # Free large selection helpers once the mapping is built
    del indices
    del gal_to_counts

    galaxy_to_ahf_nodes = [int(h) if h is not None else -1 for h in selected]

    # Build mapping from selected AHF ID -> list of galaxy indices
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
        _update_ahf_galaxy_maps(sim, galaxy_to_ahf_nodes)
        return

    matched_ids = set(mapping.keys())
    if not matched_ids:
        return

    pid_maps_sel: Dict[str, Dict[int, int]] = _build_selected_pid_maps(sim)

    def map_set(pidset: Set[int], key: str) -> np.ndarray:
        mp = pid_maps_sel.get(key, {})
        if not mp or not pidset:
            return np.array([], dtype=np.int32)
        arr = np.fromiter((mp[pid] for pid in pidset if pid in mp), dtype=np.int32)
        if arr.size == 0:
            return arr
        return np.unique(arr)

    exclusive_gal_dm = None

    final_galaxy_ahf_ids = list(galaxy_to_ahf_nodes)

    # Determine halo ownership using the AHF hierarchy
    def _build_ahf_index_map() -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for halo_index, halo in enumerate(sim.halo_list):
            ahf_hid = getattr(halo, 'AHF_haloID', None)
            if ahf_hid is None:
                continue
            try:
                mapping[int(ahf_hid)] = halo_index
            except Exception:
                continue
        return mapping

    def _resolve_halo_index(node_id: int, index_map: Dict[int, int]) -> int:
        cur = int(node_id)
        candidate = index_map.get(cur)
        visited: Set[int] = set()
        while True:
            parent = parent_of.get(cur, 0)
            if parent in (0, None):
                return candidate if candidate is not None else -1
            cur = int(parent)
            if cur in visited:
                break
            visited.add(cur)
            idx = index_map.get(cur)
            if idx is not None:
                candidate = idx
        return candidate if candidate is not None else -1

    def _assign_host_indices(index_map: Dict[int, int]) -> Tuple[List[int], List[int], Set[int]]:
        host_list: List[int] = []
        normalized_ids: List[int] = []
        for node_id in final_galaxy_ahf_ids:
            if node_id is None or node_id == -1:
                normalized_ids.append(-1)
                host_list.append(-1)
                continue
            top_id = int(node_id)
            normalized_ids.append(top_id)
            host_idx = _resolve_halo_index(top_id, index_map)
            host_list.append(int(host_idx) if host_idx is not None else -1)
        orphan_indices = {i for i, host_idx in enumerate(host_list) if host_idx is None or int(host_idx) < 0}
        return host_list, normalized_ids, orphan_indices

    ahf_to_halo_index = _build_ahf_index_map()
    host_indices, normalized_ahf_ids, orphan_set = _assign_host_indices(ahf_to_halo_index)

    if orphan_set:
        missing_halo_ids: Set[int] = {
            normalized_ahf_ids[i]
            for i in orphan_set
            if i < len(normalized_ahf_ids) and normalized_ahf_ids[i] is not None and normalized_ahf_ids[i] >= 0
        }
        created_map = _ensure_missing_ahf_halos(sim, missing_halo_ids, ahf_particles_file, pid_maps_sel)
        if created_map:
            ahf_to_halo_index = _build_ahf_index_map()
            host_indices, normalized_ahf_ids, orphan_set = _assign_host_indices(ahf_to_halo_index)

    if orphan_set:
        mylog.warning(
            'AHF: %d galaxy(ies) reference host halos that were not loaded; keeping parent index = -1',
            len(orphan_set),
        )

    sim._ahf_galaxy_hosts = host_indices

    # Rebuild halo->galaxy links based on the new mapping to avoid stale indices
    for halo in sim.halo_list:
        halo.galaxy_index_list = []

    for gi, host_idx in enumerate(host_indices):
        idx = int(host_idx) if host_idx is not None else -1
        sim.galaxy_list[gi].parent_halo_index = idx
        if idx >= 0 and idx < len(sim.halo_list):
            sim.halo_list[idx].galaxy_index_list.append(gi)

    sim._ahf_galaxy_ahf_ids = normalized_ahf_ids
    for gi, ahf_id in enumerate(normalized_ahf_ids):
        ahf_val = int(ahf_id) if ahf_id is not None and ahf_id >= 0 else -1
        if gi < len(sim.galaxy_list):
            setattr(sim.galaxy_list[gi], 'AHF_haloID', ahf_val)
    _update_ahf_galaxy_maps(sim, normalized_ahf_ids)

    _prune_halos_after_galaxies(sim)

    # Stash exclusive DM reverse map for global list construction (none in this path)
    if exclusive_gal_dm is not None:
        setattr(sim, '_exclusive_galaxy_dmlist', exclusive_gal_dm)

    if sim.ngalaxies > 0:
        try:
            from caesar.group import get_group_properties

            get_group_properties(sim, sim.galaxy_list)
        except Exception:
            pass
