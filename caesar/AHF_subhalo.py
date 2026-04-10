from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from yt.funcs import mylog

from caesar.pipeline_utils import reset_global_particle_IDs, load_global_lists


@dataclass(frozen=True)
class AHFSubhaloTask:
    node_id: int
    parent_id: int
    top_id: int
    depth: int
    ancestors: Tuple[int, ...]
    star_count: int
    fof_candidates: int


def _ancestor_chain(parent_of: Dict[int, int], node_id: int) -> Tuple[int, ...]:
    chain: List[int] = []
    cur = int(node_id)
    seen: Set[int] = set()
    while True:
        parent = int(parent_of.get(cur, 0) or 0)
        if parent <= 0 or parent in seen:
            break
        chain.append(parent)
        seen.add(parent)
        cur = parent
    return tuple(chain)


def _build_task_manifest(
    *,
    parent_of: Dict[int, int],
    host_to_nodes: Dict[int, Set[int]],
    node_npart: Dict[int, int],
    node_nstar: Dict[int, int],
    min_stars: int,
) -> Tuple[List[AHFSubhaloTask], Dict[int, List[AHFSubhaloTask]]]:
    tasks: List[AHFSubhaloTask] = []
    tasks_by_root: Dict[int, List[AHFSubhaloTask]] = {}
    for root_id, nodes in sorted(host_to_nodes.items(), key=lambda kv: int(kv[0])):
        root_tasks: List[AHFSubhaloTask] = []
        for node_id in sorted(nodes):
            ancestors = _ancestor_chain(parent_of, int(node_id))
            depth = len(ancestors)
            star_count = int(node_nstar.get(int(node_id), 0))
            if star_count < int(min_stars):
                continue
            fof_candidates = int(node_npart.get(int(node_id), 0))
            task = AHFSubhaloTask(
                node_id=int(node_id),
                parent_id=int(parent_of.get(int(node_id), 0) or 0),
                top_id=int(root_id),
                depth=int(depth),
                ancestors=ancestors,
                star_count=int(star_count),
                fof_candidates=int(max(1, fof_candidates)),
            )
            tasks.append(task)
            root_tasks.append(task)
        if root_tasks:
            tasks_by_root[int(root_id)] = root_tasks
    return tasks, tasks_by_root


def _env_int(name: str, default: int) -> int:
    import os

    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    import os

    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _env_str(name: str, default: str) -> str:
    import os

    value = os.environ.get(name, default)
    if value is None:
        return str(default)
    return str(value)


def _dense_gas_selected(
    sim,
    gidx: np.ndarray,
    *,
    fof_nHlim: float,
    fof_Tlim: float,
    fof_use_sfr_gate: bool,
) -> np.ndarray:
    gidx_arr = np.asarray(gidx, dtype=np.int64)
    if gidx_arr.size == 0:
        return np.empty(0, dtype=np.int32)
    dm = getattr(sim, "data_manager", None)
    if dm is None:
        return gidx_arr.astype(np.int32, copy=False)
    gnh = getattr(dm, "gnh", None)
    gT = getattr(dm, "gT", None)
    gsfr = getattr(dm, "gsfr", None)
    if gnh is None or gT is None or gsfr is None:
        return gidx_arr.astype(np.int32, copy=False)
    try:
        nh = gnh[gidx_arr]
        temp = gT[gidx_arr]
        sfr = gsfr[gidx_arr]
        if fof_use_sfr_gate:
            mask = (nh > fof_nHlim) & ((temp < fof_Tlim) | (sfr > 0))
        else:
            mask = (nh > fof_nHlim) & (temp < fof_Tlim)
        return gidx_arr[mask].astype(np.int32, copy=False)
    except Exception:
        return gidx_arr.astype(np.int32, copy=False)


def _sorted_unique_int64(values: Iterable[int]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.int64)
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(arr.astype(np.int64, copy=False))


def _array_union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return np.asarray(b, dtype=np.int64)
    if b.size == 0:
        return np.asarray(a, dtype=np.int64)
    return np.union1d(a.astype(np.int64, copy=False), b.astype(np.int64, copy=False))


def _array_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return np.empty(0, dtype=np.int64)
    if b.size == 0:
        return np.asarray(a, dtype=np.int64)
    return np.setdiff1d(a.astype(np.int64, copy=False), b.astype(np.int64, copy=False), assume_unique=False)


def _build_candidate_group(
    sim,
    *,
    task: AHFSubhaloTask,
    star_sel: np.ndarray,
    gas_sel: np.ndarray,
    bh_sel: np.ndarray,
    dm_sel: np.ndarray,
):
    from caesar.group import create_new_group

    grp = create_new_group(sim, "galaxy")
    grp.AHF_haloID = int(task.node_id)
    grp.AHF_parent_haloID = int(task.parent_id)
    grp.AHF_top_haloID = int(task.top_id)
    grp.AHF_depth = int(task.depth)
    grp.AHF_ancestor_haloIDs = np.asarray(task.ancestors, dtype=np.int64)
    grp.slist = np.asarray(star_sel, dtype=np.int32)
    grp.glist = np.asarray(gas_sel, dtype=np.int32)
    grp.bhlist = np.asarray(bh_sel, dtype=np.int32)
    grp.dmlist = np.asarray(dm_sel, dtype=np.int32)
    grp.global_indexes = np.array([], dtype=np.int64)
    grp._ahf_host_halo_index = -1
    return grp


def _resolve_node_membership(
    *,
    node_id: int,
    membership_arrays: Dict[int, np.ndarray],
    pid_maps_sel,
    sim,
    fof_nHlim: float,
    fof_Tlim: float,
    fof_use_sfr_gate: bool,
):
    arr = membership_arrays.get(int(node_id))
    if arr is None:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    arr = np.asarray(arr, dtype=np.int64)
    if arr.size == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
    if arr.ndim != 2 or arr.shape[1] != 2:
        arr = arr.reshape(-1, 2)

    pids = arr[:, 0]
    ptypes = arr[:, 1]

    def _map(mask, key: str) -> np.ndarray:
        lookup = pid_maps_sel.get(key)
        if lookup is None or not np.any(mask):
            return np.empty(0, dtype=np.int32)
        mapped = lookup.map(pids[mask])
        if mapped.size == 0:
            return np.empty(0, dtype=np.int32)
        return np.unique(mapped.astype(np.int32, copy=False))

    gas_sel = _map(ptypes == 0, "gas")
    star_sel = _map(ptypes == 4, "star")
    bh_sel = _map(ptypes == 5, "bh")
    dm_sel = _map(ptypes == 1, "dm")
    gas_sel_dense = _dense_gas_selected(
        sim,
        gas_sel,
        fof_nHlim=float(fof_nHlim),
        fof_Tlim=float(fof_Tlim),
        fof_use_sfr_gate=bool(fof_use_sfr_gate),
    )
    return gas_sel_dense, star_sel, bh_sel, dm_sel


def _fof_for_task(
    sim,
    *,
    task: AHFSubhaloTask,
    membership_arrays: Dict[int, np.ndarray],
    pid_maps_sel,
    min_stars: int,
    fof_ll: float,
    fof_vel_ll: Optional[float],
    fof_nHlim: float,
    fof_Tlim: float,
    fof_use_sfr_gate: bool,
    backend: str,
    cc_backend: str,
    max_pairs_per_batch: int,
    blocked_star_idx: Optional[np.ndarray] = None,
):
    from caesar.fof6d_graph import fof6d_on_pool_graph

    gas_sel_dense, star_sel, bh_sel, dm_sel = _resolve_node_membership(
        node_id=int(task.node_id),
        membership_arrays=membership_arrays,
        pid_maps_sel=pid_maps_sel,
        sim=sim,
        fof_nHlim=float(fof_nHlim),
        fof_Tlim=float(fof_Tlim),
        fof_use_sfr_gate=bool(fof_use_sfr_gate),
    )

    if blocked_star_idx is not None and blocked_star_idx.size > 0:
        star_sel = _array_diff(np.asarray(star_sel, dtype=np.int64), blocked_star_idx).astype(np.int32, copy=False)

    if int(star_sel.size) < int(min_stars):
        return []

    gas_concat = sim.data_manager.selected_to_concat("gas", np.asarray(gas_sel_dense, dtype=np.int64))
    star_concat = sim.data_manager.selected_to_concat("star", np.asarray(star_sel, dtype=np.int64))
    if bh_sel.size > 0:
        bh_concat = sim.data_manager.selected_to_concat("bh", np.asarray(bh_sel, dtype=np.int64))
    else:
        bh_concat = np.empty(0, dtype=np.int64)

    eligible_concat = np.concatenate((gas_concat, star_concat, bh_concat), axis=None).astype(np.int64, copy=False)
    if eligible_concat.size < int(min_stars):
        return []

    tags, _ = fof6d_on_pool_graph(
        sim.data_manager.pos[eligible_concat],
        sim.data_manager.vel[eligible_concat],
        fof_ll=float(fof_ll),
        vel_ll=fof_vel_ll,
        mingrp=int(min_stars),
        periodic=False,
        kernel="caesar_table",
        backend="auto" if backend == "auto" else backend,
        cc_backend="auto" if cc_backend == "auto" else cc_backend,
        max_pairs_per_batch=int(max_pairs_per_batch),
    )

    tags = np.asarray(tags, dtype=np.int64)
    valid_tags = np.unique(tags[tags >= 0])
    if valid_tags.size == 0:
        return []

    ng = int(gas_sel_dense.size)
    ns = int(star_sel.size)
    nb = int(bh_sel.size)
    allow_dm = bool(valid_tags.size == 1)
    out = []
    for gid in valid_tags:
        mask = tags == int(gid)
        gsub = gas_sel_dense[mask[:ng]] if ng > 0 else np.empty(0, dtype=np.int32)
        ssub = star_sel[mask[ng:ng + ns]] if ns > 0 else np.empty(0, dtype=np.int32)
        bsub = bh_sel[mask[ng + ns:ng + ns + nb]] if nb > 0 else np.empty(0, dtype=np.int32)
        if int(ssub.size) < int(min_stars):
            continue
        grp = _build_candidate_group(
            sim,
            task=task,
            star_sel=ssub,
            gas_sel=gsub,
            bh_sel=bsub,
            dm_sel=dm_sel if allow_dm else np.empty(0, dtype=np.int32),
        )
        out.append(grp)
    return out


def _reconcile_root(
    *,
    root_id: int,
    tasks: Sequence[AHFSubhaloTask],
    initial_candidates_by_node: Dict[int, List],
    membership_arrays: Dict[int, np.ndarray],
    pid_maps_sel,
    sim,
    min_stars: int,
    fof_ll: float,
    fof_vel_ll: Optional[float],
    fof_nHlim: float,
    fof_Tlim: float,
    fof_use_sfr_gate: bool,
    backend: str,
    cc_backend: str,
    max_pairs_per_batch: int,
) -> List:
    tasks_desc = sorted(tasks, key=lambda t: (int(t.depth), int(t.node_id)), reverse=True)
    carry_claimed: Dict[int, np.ndarray] = {}
    final_candidates: List = []

    for task in tasks_desc:
        descendant_claimed = np.asarray(carry_claimed.pop(int(task.node_id), np.empty(0, dtype=np.int64)), dtype=np.int64)
        if descendant_claimed.size == 0:
            node_candidates = list(initial_candidates_by_node.get(int(task.node_id), []))
        else:
            node_candidates = _fof_for_task(
                sim,
                task=task,
                membership_arrays=membership_arrays,
                pid_maps_sel=pid_maps_sel,
                min_stars=int(min_stars),
                fof_ll=float(fof_ll),
                fof_vel_ll=fof_vel_ll,
                fof_nHlim=float(fof_nHlim),
                fof_Tlim=float(fof_Tlim),
                fof_use_sfr_gate=bool(fof_use_sfr_gate),
                backend=backend,
                cc_backend=cc_backend,
                max_pairs_per_batch=int(max_pairs_per_batch),
                blocked_star_idx=descendant_claimed,
            )

        node_claimed = np.asarray(descendant_claimed, dtype=np.int64)
        for gal in node_candidates:
            node_claimed = _array_union(node_claimed, np.asarray(gal.slist, dtype=np.int64))
            final_candidates.append(gal)

        parent_id = int(task.parent_id)
        if parent_id > 0 and node_claimed.size > 0:
            carry_claimed[parent_id] = _array_union(
                np.asarray(carry_claimed.get(parent_id, np.empty(0, dtype=np.int64)), dtype=np.int64),
                node_claimed,
            )

    return final_candidates


def build_galaxies_from_ahf_subhalo(
    sim,
    ahf_particles_file: str,
    *,
    min_stars: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> None:
    import os as _os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    from caesar.group import get_min_stars as _get_min_stars
    from caesar.group import get_group_properties as _get_group_properties
    from caesar.AHF_FAST_loader import load_ahf_halos_dataframe, load_ahf_hierarchy, load_ahf_particle_blocks
    from caesar.fubar import get_b as _get_b
    from caesar.fubar import get_mean_interparticle_separation as _get_mean_interparticle_separation
    from caesar.halo_matching import (
        _build_selected_pid_maps,
        _ensure_missing_ahf_halos,
        _group_nodes_by_root,
        _prune_halos_after_galaxies,
        _update_ahf_galaxy_maps,
    )

    min_stars = _get_min_stars(sim, override=min_stars)

    parent_of = getattr(sim, "_ahf_fast_parent_of", None)
    children_of = getattr(sim, "_ahf_fast_children_of", None)
    membership_arrays = getattr(sim, "_ahf_fast_memberships", None)
    halos_df = getattr(sim, "_ahf_fast_halos_df", None)

    if halos_df is None:
        halos_df = load_ahf_halos_dataframe(ahf_particles_file)
        sim._ahf_fast_halos_df = halos_df
    if parent_of is None or children_of is None:
        hier = load_ahf_hierarchy(ahf_particles_file)
        parent_of = hier.parent_of
        children_of = hier.children_of
        sim._ahf_fast_parent_of = parent_of
        sim._ahf_fast_children_of = children_of
    if membership_arrays is None:
        membership_arrays = load_ahf_particle_blocks(
            ahf_particles_file,
            needed_nodes=parent_of.keys(),
            load_dm=True,
        )
        sim._ahf_fast_memberships = membership_arrays

    if not parent_of:
        sim.galaxy_list = []
        sim.galaxies = []
        sim.ngalaxies = 0
        sim._ahf_galaxy_hosts = []
        sim._ahf_galaxy_ahf_ids = []
        sim._ahf_galaxy_top_ahf_ids = []
        _update_ahf_galaxy_maps(sim, [])
        return

    node_npart = {
        int(hid): int(npart)
        for hid, npart in zip(halos_df["hid"].to_numpy(), halos_df["npart"].to_numpy())
    }
    node_nstar = {
        int(hid): max(0, int(nstar))
        for hid, nstar in zip(halos_df["hid"].to_numpy(), halos_df["n_star"].to_numpy())
    }

    host_to_nodes, _ = _group_nodes_by_root(parent_of)
    tasks, tasks_by_root = _build_task_manifest(
        parent_of=parent_of,
        host_to_nodes=host_to_nodes,
        node_npart=node_npart,
        node_nstar=node_nstar,
        min_stars=int(min_stars),
    )

    if not tasks:
        sim.galaxy_list = []
        sim.galaxies = []
        sim.ngalaxies = 0
        sim._ahf_galaxy_hosts = []
        sim._ahf_galaxy_ahf_ids = []
        sim._ahf_galaxy_top_ahf_ids = []
        _update_ahf_galaxy_maps(sim, [])
        return

    jobs = n_jobs
    if jobs is None:
        jobs = getattr(sim, "nproc", 1)
    try:
        jobs = int(jobs)
    except Exception:
        jobs = 1
    jobs = max(1, jobs)

    backend = _env_str("CAESAR_AHF_SUBHALO_BACKEND", "auto").lower()
    if backend == "cpu":
        backend = "numpy"
    elif backend == "gpu":
        backend = "cupy"
    cc_backend = _env_str("CAESAR_AHF_SUBHALO_CC_BACKEND", "auto").lower()
    max_pairs_per_batch = max(1, _env_int("CAESAR_AHF_SUBHALO_MAX_PAIRS_PER_BATCH", 5_000_000))

    fof_vel_ll = 1.0
    try:
        _vel_env = _os.environ.get("CAESAR_FOF6D_VEL_LL")
        if _vel_env is not None and _vel_env != "":
            fof_vel_ll = float(_vel_env)
    except Exception:
        pass
    if _os.environ.get("CAESAR_FOF6D_DISABLE_VEL", "0") == "1":
        fof_vel_ll = None

    fof_nHlim = _env_float("CAESAR_AHF_FAST_FOF_NHLIM", 0.13)
    fof_Tlim = _env_float("CAESAR_AHF_FAST_FOF_TLIM", 1.0e5)
    fof_use_sfr_gate = _os.environ.get("CAESAR_AHF_FAST_FOF_USE_SFR", "1") == "1"

    fof_mis = float(_get_mean_interparticle_separation(sim).d)
    fof_ll = float(fof_mis * _get_b(sim, "galaxy"))

    pid_maps_sel = _build_selected_pid_maps(sim)

    show_progress = bool(getattr(sim, "_show_progress", True))
    task_progress = tqdm(
        total=len(tasks),
        desc="Building galaxies (AHF-subhalo)",
        disable=(not show_progress or len(tasks) == 0),
        leave=False,
        mininterval=2.0,
        miniters=32,
        smoothing=0.0,
    )

    initial_candidates_by_node: Dict[int, List] = {}
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                _fof_for_task,
                sim,
                task=task,
                membership_arrays=membership_arrays,
                pid_maps_sel=pid_maps_sel,
                min_stars=int(min_stars),
                fof_ll=float(fof_ll),
                fof_vel_ll=fof_vel_ll,
                fof_nHlim=float(fof_nHlim),
                fof_Tlim=float(fof_Tlim),
                fof_use_sfr_gate=bool(fof_use_sfr_gate),
                backend=backend,
                cc_backend=cc_backend,
                max_pairs_per_batch=int(max_pairs_per_batch),
            ): task
            for task in tasks
        }
        for fut in as_completed(futures):
            task = futures[fut]
            initial_candidates_by_node[int(task.node_id)] = list(fut.result())
            if task_progress is not None:
                task_progress.update(1)
    if task_progress is not None:
        task_progress.close()

    final_galaxies: List = []
    for root_id, root_tasks in sorted(tasks_by_root.items(), key=lambda kv: int(kv[0])):
        final_galaxies.extend(
            _reconcile_root(
                root_id=int(root_id),
                tasks=root_tasks,
                initial_candidates_by_node=initial_candidates_by_node,
                membership_arrays=membership_arrays,
                pid_maps_sel=pid_maps_sel,
                sim=sim,
                min_stars=int(min_stars),
                fof_ll=float(fof_ll),
                fof_vel_ll=fof_vel_ll,
                fof_nHlim=float(fof_nHlim),
                fof_Tlim=float(fof_Tlim),
                fof_use_sfr_gate=bool(fof_use_sfr_gate),
                backend=backend,
                cc_backend=cc_backend,
                max_pairs_per_batch=int(max_pairs_per_batch),
            )
        )

    sim.galaxy_list = final_galaxies
    sim.galaxies = final_galaxies
    sim.ngalaxies = len(final_galaxies)
    setattr(sim, "_ahf_matched", True)
    setattr(sim, "_include_dm_in_galaxies", True)

    if sim.ngalaxies == 0:
        sim._ahf_galaxy_hosts = []
        sim._ahf_galaxy_ahf_ids = []
        sim._ahf_galaxy_top_ahf_ids = []
        _update_ahf_galaxy_maps(sim, [])
        return

    ahf_to_halo_index = {}
    for halo_index, halo in enumerate(sim.halo_list):
        ahf_hid = getattr(halo, "AHF_haloID", None)
        if ahf_hid is None:
            continue
        try:
            ahf_to_halo_index[int(ahf_hid)] = int(halo_index)
        except Exception:
            continue

    missing_top_ids = {
        int(gal.AHF_top_haloID)
        for gal in sim.galaxy_list
        if int(gal.AHF_top_haloID) not in ahf_to_halo_index
    }
    if missing_top_ids:
        _ensure_missing_ahf_halos(sim, missing_top_ids, ahf_particles_file, pid_maps_sel)
        ahf_to_halo_index = {}
        for halo_index, halo in enumerate(sim.halo_list):
            ahf_hid = getattr(halo, "AHF_haloID", None)
            if ahf_hid is None:
                continue
            try:
                ahf_to_halo_index[int(ahf_hid)] = int(halo_index)
            except Exception:
                continue

    for halo in sim.halo_list:
        halo.galaxy_index_list = []

    host_indices: List[int] = []
    for gi, gal in enumerate(sim.galaxy_list):
        idx = int(ahf_to_halo_index.get(int(gal.AHF_top_haloID), -1))
        gal._ahf_host_halo_index = idx
        gal.parent_halo_index = idx
        host_indices.append(idx)
        if idx >= 0 and idx < len(sim.halo_list):
            sim.halo_list[idx].galaxy_index_list.append(gi)

    sim._ahf_galaxy_hosts = [int(h) for h in host_indices]
    sim._ahf_galaxy_ahf_ids = [int(getattr(gal, "AHF_haloID", -1)) for gal in sim.galaxy_list]
    sim._ahf_galaxy_top_ahf_ids = [int(getattr(gal, "AHF_top_haloID", -1)) for gal in sim.galaxy_list]
    _update_ahf_galaxy_maps(sim, sim._ahf_galaxy_ahf_ids)

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
            if hasattr(gal, "dmlist") and gal.dmlist is not None and len(gal.dmlist) > 0:
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

    for gal in sim.galaxy_list:
        gal.global_indexes = _compute_global_indexes(gal)

    ctx = _Ctx(sim)
    _get_group_properties(ctx, sim.galaxy_list)

    try:
        if "galaxy" not in sim.group_types:
            sim.group_types.append("galaxy")
    except Exception:
        pass


def run(obj):
    """AHF-subhalo member_search pipeline (haloid='AHF-subhalo')."""

    obj.nproc = 1
    if "nproc" in obj._kwargs:
        obj.nproc = int(obj._kwargs["nproc"])
    if obj.nproc != 1:
        import joblib

        if obj.nproc < 0:
            obj.nproc += joblib.cpu_count() + 1
        if obj.nproc == 0:
            obj.nproc = joblib.cpu_count()
    mylog.info("member_search() running on %d cores" % obj.nproc)

    from caesar.AHF_FAST_halos import build_halos_from_ahf_fast

    if not obj._kwargs.get("haloid_file"):
        raise ValueError("AHF-subhalo requires an AHF_particles file via haloid_file.")

    halos = build_halos_from_ahf_fast(obj, obj._kwargs["haloid_file"])
    if halos is None:
        return
    if not obj.simulation.baryons_present:
        return

    from caesar.group import get_min_stars

    ms = get_min_stars(obj)
    build_galaxies_from_ahf_subhalo(obj, obj._kwargs["haloid_file"], min_stars=ms)

    reset_global_particle_IDs(obj)
    load_global_lists(obj)
