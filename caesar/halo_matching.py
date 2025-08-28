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
    """Integrate AHF star-only matching, hierarchical pruning, and reassignment.

    Modifies ``sim.galaxy_list`` in-place before property recomputation.
    """
    from caesar.property_manager import get_property, has_ptype
    from caesar.group import get_group_properties

    # Must have a galaxy list already
    if not hasattr(sim, 'galaxy_list') or len(sim.galaxy_list) == 0:
        return

    # Build CAESAR galaxy star PID memberships
    star_ids = get_property(sim, 'pid', 'star').d.astype(np.int64)
    caesar_data = _galaxies_to_namedata_from_group_list(sim.galaxy_list, star_ids)

    # Load full AHF particle memberships (all ptypes) and index by ID
    ahf_lines = _maybe_readlines(ahf_particles_file)
    ahf_all = read_file_to_structure(ahf_particles_file, lines=ahf_lines)
    ahf_by_id: Dict[int, ParticleMembership] = {pm.id: pm for pm in ahf_all}

    # Match galaxies to AHF by star intersection
    n_jobs = getattr(sim, 'nproc', None)
    try:
        n_jobs = int(n_jobs) if n_jobs is not None else None
    except Exception:
        n_jobs = None
    matches = find_best_matches(caesar_data, ahf_all, n_jobs=n_jobs)

    # Map AHF ID -> list of CAESAR galaxy indices in sim.galaxy_list
    from collections import defaultdict
    mapping: Dict[int, List[int]] = defaultdict(list)
    caesar_gid_to_index = {int(g.GroupID): i for i, g in enumerate(sim.galaxy_list)}
    for idx, (gid, ahf_id) in enumerate(matches):
        if ahf_id != -1:
            gi = caesar_gid_to_index.get(int(gid))
            if gi is not None:
                mapping[int(ahf_id)].append(gi)

    if not mapping:
        return

    # Build AHF hierarchy
    parent_of, children_of = _read_ahf_hierarchy(ahf_particles_file)

    matched_ids = set(mapping.keys())
    # Need all ancestors of matched nodes and all their descendants for pruning
    ancestors = _collect_ancestors(matched_ids, parent_of)
    descendants = _collect_descendants(matched_ids, children_of)
    needed_ids = matched_ids | ancestors | descendants

    # Compute exclusive memberships bottom-up
    exclusives = _compute_exclusive_memberships(ahf_by_id, children_of, needed_ids)

    # Build PID->index maps for each ptype present
    pid_maps: Dict[str, Dict[int, int]] = {}
    if has_ptype(sim, 'star'):
        pid_maps['star'] = _pid_to_index_map(star_ids)
    if has_ptype(sim, 'dm'):
        pid_maps['dm'] = _pid_to_index_map(get_property(sim, 'pid', 'dm').d.astype(np.int64))
    if has_ptype(sim, 'gas'):
        pid_maps['gas'] = _pid_to_index_map(get_property(sim, 'pid', 'gas').d.astype(np.int64))
    if has_ptype(sim, 'bh'):
        pid_maps['bh'] = _pid_to_index_map(get_property(sim, 'pid', 'bh').d.astype(np.int64))

    # Helper to map PID sets to index arrays
    def map_set(pidset: Set[int], key: str) -> List[int]:
        mp = pid_maps.get(key, {})
        return [mp[pid] for pid in pidset if pid in mp]

    # Merge galaxies per AHF ID and overwrite particle lists with exclusive sets
    to_remove: Set[int] = set()
    for ahf_id, gal_indices in mapping.items():
        if ahf_id not in exclusives:
            continue
        e = exclusives[ahf_id]
        gal_indices = sorted(set(gal_indices))
        base_i = gal_indices[0]
        base = sim.galaxy_list[base_i]

        # Overwrite lists from exclusives
        base.slist = np.array(map_set(e.parttype4, 'star'), dtype=np.int32)
        base.dmlist = np.array(map_set(e.parttype1, 'dm'), dtype=np.int32) if hasattr(base, 'dmlist') else np.array(map_set(e.parttype1, 'dm'), dtype=np.int32)
        base.glist = np.array(map_set(e.parttype0, 'gas'), dtype=np.int32) if hasattr(base, 'glist') else np.array(map_set(e.parttype0, 'gas'), dtype=np.int32)
        if 'bh' in pid_maps:
            base.bhlist = np.array(map_set(e.parttype5, 'bh'), dtype=np.int32)

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
