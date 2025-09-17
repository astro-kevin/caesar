from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional, Dict, Iterable
import os
import re
import glob
from itertools import islice

import h5py
from tqdm import tqdm
import numpy as np
import gzip


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

    If ``n_jobs`` > 1 and joblib is available, parallelize across ``list1``.
    """

    def best_for(pm1: ParticleMembership) -> Tuple[int, int]:
        set1 = pm1.parttype4
        best_intersection = 0
        best_id = -1
        if not set1:
            return (pm1.id, -1)
        for pm2 in list2:
            inter = len(set1 & pm2.parttype4)
            if inter > 0.5 * len(set1):
                return (pm1.id, pm2.id)
            if inter > best_intersection:
                best_intersection = inter
                best_id = pm2.id
        return (pm1.id, best_id)

    if n_jobs is None or n_jobs == 1:
        return [best_for(pm1) for pm1 in tqdm(list1, desc="Matching Progress")]

    try:
        from joblib import Parallel, delayed
        # Use process-based parallelism to avoid GIL for Python-level loops
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(best_for)(pm1) for pm1 in list1
        )
        return results
    except Exception:
        # Fallback to serial
        return [best_for(pm1) for pm1 in tqdm(list1, desc="Matching Progress")]


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


def _read_memberships_for_ids(path: str, needed: Set[int]) -> Dict[int, ParticleMembership]:
    """Read AHF particle memberships only for the given IDs (supports .gz)."""
    out: Dict[int, ParticleMembership] = {}

    def _open_particles(p: str):
        if p.endswith('.gz'):
            import gzip
            return gzip.open(p, 'rt')
        return open(p, 'r')

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

    - Uses exclusive baryon memberships computed from the AHF hierarchy.
    - Keeps nodes with >= min_stars as galaxies.
    - Deposits sub-threshold satellites into the host central; discards
      top-level sub-threshold hosts with no eligible central.
    - DM is inclusive for each selected node; an exclusive reverse map
      is constructed for galaxies after properties are computed.
    """
    from caesar.group import create_new_group
    from caesar.group import get_group_properties as _get_group_properties
    from caesar.property_manager import get_property, has_ptype

    # Build PID -> selected-index maps for per-type lists
    pid_maps_sel = _build_selected_pid_maps(sim)

    # Full-snapshot DM index map for reverse assignment
    pid_to_dm_fullidx: Dict[int, int] = {}
    ndm_full = 0
    if has_ptype(sim, 'dm'):
        dm_pids_full = get_property(sim, 'pid', 'dm').d.astype(np.int64)
        ndm_full = len(dm_pids_full)
        if ndm_full > 0:
            pid_to_dm_fullidx = {int(pid): int(i) for i, pid in enumerate(dm_pids_full.tolist())}

    # Load hierarchy and memberships
    parent_of, children_of = _read_ahf_hierarchy(ahf_particles_file)
    needed_ids = _collect_all_node_ids(parent_of, children_of)
    if not needed_ids:
        sim.galaxy_list = []
        sim.ngalaxies = 0
        return
    ahf_by_id = _read_memberships_for_ids(ahf_particles_file, needed_ids)
    exclusives = _compute_exclusive_memberships(ahf_by_id, children_of, needed_ids)

    # Partition by host root
    by_host: Dict[int, List[int]] = {}
    for hid in needed_ids:
        root = _root_of(hid, parent_of)
        by_host.setdefault(root, []).append(hid)

    galaxies = []
    galnode_to_dm_exclusive: Dict[int, Set[int]] = {}

    # Helper to map PID sets to selected indices
    def map_sel(pidset: Set[int], key: str) -> List[int]:
        mp = pid_maps_sel.get(key, {})
        if not mp or not pidset:
            return []
        return [mp[pid] for pid in pidset if pid in mp]

    # Build galaxy groups
    for host, nodes in by_host.items():
        # Eligible by star threshold
        eligible = [n for n in nodes if len(exclusives.get(n, ParticleMembership(n)).parttype4) >= min_stars]
        if not eligible:
            # No central -> discard all nodes under this host
            continue
        # Choose central as eligible with max stars (tie-break by id)
        central = max(eligible, key=lambda n: (len(exclusives[n].parttype4), n))

        # Precompute deposit sets for central
        deposit_star: Set[int] = set()
        deposit_gas: Set[int] = set()
        deposit_bh: Set[int] = set()
        deposit_dm_inclusive: Set[int] = set()

        for n in nodes:
            if n == central:
                continue
            stars_n = exclusives.get(n, ParticleMembership(n)).parttype4
            if len(stars_n) >= min_stars:
                continue  # satellite galaxy remains separate
            # Sub-threshold: deposit only if satellite (has parent)
            if parent_of.get(n, 0) not in (None, 0):
                ex = exclusives.get(n, ParticleMembership(n))
                deposit_star |= ex.parttype4
                deposit_gas |= ex.parttype0
                deposit_bh |= ex.parttype5
                dm_pm = ahf_by_id.get(n)
                if dm_pm is not None:
                    deposit_dm_inclusive |= dm_pm.parttype1
            # else top-level and sub-threshold: discard

        # Build central galaxy
        cen_ex = exclusives[central]
        cen_stars = set(cen_ex.parttype4) | deposit_star
        cen_gas = set(cen_ex.parttype0) | deposit_gas
        cen_bh = set(cen_ex.parttype5) | deposit_bh
        cen_dm_inclusive = set(ahf_by_id.get(central, ParticleMembership(central)).parttype1) | deposit_dm_inclusive

        g = create_new_group(sim, 'galaxy')
        g.slist = np.array(map_sel(cen_stars, 'star'), dtype=np.int32)
        g.glist = np.array(map_sel(cen_gas, 'gas'), dtype=np.int32)
        if 'bh' in pid_maps_sel:
            g.bhlist = np.array(map_sel(cen_bh, 'bh'), dtype=np.int32)
        if 'dm' in pid_maps_sel:
            g.dmlist = np.array(map_sel(cen_dm_inclusive, 'dm'), dtype=np.int32)
        # Build global indexes for property kernels
        gi = []
        if len(g.glist) > 0:
            gi.append(sim.data_manager.glist[g.glist])
        if len(g.slist) > 0:
            gi.append(sim.data_manager.slist[g.slist])
        if hasattr(g, 'dmlist') and g.dmlist is not None and len(getattr(g, 'dmlist')) > 0 and has_ptype(sim, 'dm'):
            gi.append(sim.data_manager.dmlist[g.dmlist])
        if hasattr(g, 'bhlist') and g.bhlist is not None and len(getattr(g, 'bhlist')) > 0:
            gi.append(sim.data_manager.bhlist[g.bhlist])
        if gi:
            g.global_indexes = np.concatenate(gi).astype(np.int64)
        galaxies.append(g)
        galnode_to_dm_exclusive[id(g)] = set(exclusives[central].parttype1)

        # Satellite galaxies (eligible)
        for n in eligible:
            if n == central:
                continue
            ex = exclusives[n]
            dm_incl = ahf_by_id.get(n, ParticleMembership(n)).parttype1
            sg = create_new_group(sim, 'galaxy')
            sg.slist = np.array(map_sel(ex.parttype4, 'star'), dtype=np.int32)
            sg.glist = np.array(map_sel(ex.parttype0, 'gas'), dtype=np.int32)
            if 'bh' in pid_maps_sel:
                sg.bhlist = np.array(map_sel(ex.parttype5, 'bh'), dtype=np.int32)
            if 'dm' in pid_maps_sel:
                sg.dmlist = np.array(map_sel(dm_incl, 'dm'), dtype=np.int32)
            gi = []
            if len(sg.glist) > 0:
                gi.append(sim.data_manager.glist[sg.glist])
            if len(sg.slist) > 0:
                gi.append(sim.data_manager.slist[sg.slist])
            if hasattr(sg, 'dmlist') and sg.dmlist is not None and len(getattr(sg, 'dmlist')) > 0 and has_ptype(sim, 'dm'):
                gi.append(sim.data_manager.dmlist[sg.dmlist])
            if hasattr(sg, 'bhlist') and sg.bhlist is not None and len(getattr(sg, 'bhlist')) > 0:
                gi.append(sim.data_manager.bhlist[sg.bhlist])
            if gi:
                sg.global_indexes = np.concatenate(gi).astype(np.int64)
            galaxies.append(sg)
            galnode_to_dm_exclusive[id(sg)] = set(exclusives[n].parttype1)

    # Assign to sim and compute properties
    sim.galaxy_list = galaxies
    sim.ngalaxies = len(galaxies)

    class _Ctx:
        def __init__(self, sim):
            self.obj = sim
            self.obj_type = 'galaxy'
            self.nproc = getattr(sim, 'nproc', 1)
            self.load_pot = getattr(sim, 'load_pot', True)
            self.nparttot = sum(len(getattr(g, 'global_indexes', [])) for g in sim.galaxy_list)
            # not used by our collate, but keep a dict present
            self.nparttype = {p: len(getattr(sim.data_manager, f"{p}list", [])) for p in ['gas','star','bh','dm','dm2','dm3'] if hasattr(sim.data_manager, f"{p}list")}

    ctx = _Ctx(sim)
    _get_group_properties(ctx, sim.galaxy_list)

    # Ensure group_types includes 'galaxy' so downstream reverse maps are built
    try:
        if 'galaxy' not in sim.group_types:
            sim.group_types.append('galaxy')
    except Exception:
        pass

    # Exclusive DM reverse map in full-snapshot space, using final GroupIDs
    if ndm_full > 0:
        exclusive_gal_dm = np.full(ndm_full, -1, dtype=np.int32)
        for g in sim.galaxy_list:
            ex_dm = galnode_to_dm_exclusive.get(id(g))
            if not ex_dm:
                continue
            idxs = [pid_to_dm_fullidx[pid] for pid in ex_dm if pid in pid_to_dm_fullidx]
            if idxs:
                exclusive_gal_dm[np.array(idxs, dtype=np.int64)] = int(g.GroupID)
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

    ahf_data = read_file_to_structure(ahf_file)
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
        subhalo = read_single_halo_from_file(ahf_file, ahf_id)
        if subhalo is None:
            continue
        base = sim.galaxies[gal_indices[0]]
        # merge star lists from all matched galaxies
        merged_slist = []
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
        if 'bh' in pid_maps:
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
            if hasattr(base, 'bhlist') and base.bhlist is not None and len(base.bhlist) > 0 and 'bh' in pid_maps:
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
