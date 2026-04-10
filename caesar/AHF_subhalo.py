from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
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


@dataclass(frozen=True)
class AHFSubhaloBatch:
    tasks: Tuple[AHFSubhaloTask, ...]
    is_tiny_batch: bool = False
    target_backend: str = "cpu"
    target_device: Optional[int] = None
    estimated_cost: int = 0


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


def _task_ancestor_set(task: AHFSubhaloTask) -> Set[int]:
    return set(int(v) for v in task.ancestors)


def _tasks_are_batch_compatible(existing: Sequence[AHFSubhaloTask], candidate: AHFSubhaloTask) -> bool:
    cand_id = int(candidate.node_id)
    cand_anc = _task_ancestor_set(candidate)
    for other in existing:
        other_id = int(other.node_id)
        other_anc = _task_ancestor_set(other)
        if cand_id == other_id:
            return False
        if cand_id in other_anc or other_id in cand_anc:
            return False
    return True


def _available_gpu_device_ids() -> List[int]:
    try:
        from caesar.fof6d_graph import _try_import_cupy
    except Exception:
        return []

    cp = _try_import_cupy()
    if cp is None:
        return []
    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return []
    return list(range(max(0, ndev)))


def _build_tiny_batches(
    tasks: Sequence[AHFSubhaloTask],
    *,
    max_nodes_per_batch: int,
    max_fof_candidates_per_batch: int,
) -> List[AHFSubhaloBatch]:
    pending = sorted(tasks, key=lambda t: (int(t.fof_candidates), int(t.star_count), int(t.node_id)))
    batches: List[AHFSubhaloBatch] = []
    while pending:
        batch_tasks: List[AHFSubhaloTask] = []
        total_candidates = 0
        keep: List[AHFSubhaloTask] = []
        for task in pending:
            if len(batch_tasks) >= int(max_nodes_per_batch):
                keep.append(task)
                continue
            next_candidates = int(total_candidates + int(task.fof_candidates))
            if batch_tasks and next_candidates > int(max_fof_candidates_per_batch):
                keep.append(task)
                continue
            if not _tasks_are_batch_compatible(batch_tasks, task):
                keep.append(task)
                continue
            batch_tasks.append(task)
            total_candidates = next_candidates
        if not batch_tasks:
            batch_tasks.append(pending[0])
            keep = pending[1:]
            total_candidates = int(batch_tasks[0].fof_candidates)
        batches.append(
            AHFSubhaloBatch(
                tasks=tuple(batch_tasks),
                is_tiny_batch=len(batch_tasks) > 1,
                target_backend="cpu",
                target_device=None,
                estimated_cost=int(total_candidates),
            )
        )
        pending = keep
    return batches


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


def _serialize_candidate_group(gal) -> Dict[str, np.ndarray | int]:
    payload = {
        "AHF_haloID": int(getattr(gal, "AHF_haloID", -1)),
        "AHF_parent_haloID": int(getattr(gal, "AHF_parent_haloID", -1)),
        "AHF_top_haloID": int(getattr(gal, "AHF_top_haloID", -1)),
        "AHF_depth": int(getattr(gal, "AHF_depth", 0)),
        "AHF_ancestor_haloIDs": np.asarray(getattr(gal, "AHF_ancestor_haloIDs", []), dtype=np.int64),
        "slist": np.asarray(getattr(gal, "slist", []), dtype=np.int32),
        "glist": np.asarray(getattr(gal, "glist", []), dtype=np.int32),
        "bhlist": np.asarray(getattr(gal, "bhlist", []), dtype=np.int32),
        "dmlist": np.asarray(getattr(gal, "dmlist", []), dtype=np.int32),
    }
    if hasattr(gal, "_merge_id"):
        payload["_merge_id"] = int(getattr(gal, "_merge_id"))
    if hasattr(gal, "_ahf_host_halo_index"):
        payload["_ahf_host_halo_index"] = int(getattr(gal, "_ahf_host_halo_index"))
    return payload


def _deserialize_candidate_group(sim, payload: Dict[str, np.ndarray | int]):
    from caesar.group import create_new_group

    grp = create_new_group(sim, "galaxy")
    grp.AHF_haloID = int(payload["AHF_haloID"])
    grp.AHF_parent_haloID = int(payload["AHF_parent_haloID"])
    grp.AHF_top_haloID = int(payload["AHF_top_haloID"])
    grp.AHF_depth = int(payload["AHF_depth"])
    grp.AHF_ancestor_haloIDs = np.asarray(payload["AHF_ancestor_haloIDs"], dtype=np.int64)
    grp.slist = np.asarray(payload["slist"], dtype=np.int32)
    grp.glist = np.asarray(payload["glist"], dtype=np.int32)
    grp.bhlist = np.asarray(payload["bhlist"], dtype=np.int32)
    grp.dmlist = np.asarray(payload["dmlist"], dtype=np.int32)
    grp.global_indexes = np.array([], dtype=np.int64)
    grp._ahf_host_halo_index = int(payload.get("_ahf_host_halo_index", -1))
    if "_merge_id" in payload:
        grp._merge_id = int(payload["_merge_id"])
    return grp


def _serialize_group_state(group, *, id_key: str) -> Dict:
    state = {"id": int(getattr(group, id_key))}
    skip = {
        "obj",
        "halo",
        "galaxies",
        "satellite_galaxies",
        "clouds",
        "galaxy",
        "central_galaxy",
        "satellite_galaxy_index_list",
    }
    for key, value in group.__dict__.items():
        if key in skip:
            continue
        state[key] = value
    return state


def _apply_group_state(group, state: Dict) -> None:
    skip = {"id", "obj", "halo", "galaxies", "satellite_galaxies", "clouds", "galaxy", "central_galaxy"}
    for key, value in state.items():
        if key in skip:
            continue
        setattr(group, key, value)


def _compute_group_properties_subset(sim, *, group_type: str, groups: Sequence) -> None:
    if not groups:
        return

    from caesar.group_funcs import (
        get_group_bh_properties,
        get_group_dust_properties,
        get_group_gas_properties,
        get_group_overall_properties,
        get_group_star_properties,
    )
    from caesar.group import has_property

    class _Ctx:
        def __init__(self, sim, group_type, groups):
            self.obj = sim
            self.obj_type = group_type
            self.nproc = getattr(sim, "nproc", 1)
            self.load_pot = getattr(sim, "load_pot", True)
            self.nparttot = sum(len(getattr(g, "global_indexes", [])) for g in groups)
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
            self.counts = {group_type: len(groups)}

    ctx = _Ctx(sim, group_type, groups)
    get_group_overall_properties(ctx, groups)
    if "gas" in sim.data_manager.ptypes:
        get_group_gas_properties(ctx, groups)
    if ("star" in sim.data_manager.ptypes) and (sim.simulation.nstar > 0):
        get_group_star_properties(ctx, groups)
    if "dust" in sim.data_manager.ptypes:
        get_group_dust_properties(ctx, groups)
    if sim.data_manager.blackholes and has_property(sim, "bh", "bhmdot"):
        get_group_bh_properties(ctx, groups)

    if group_type == "galaxy" and len(groups) > 0:
        from caesar.hydrogen_mass_calc import get_HIH2_masses, _get_aperture_quan

        if "aperture" in sim._kwargs:
            aperture = float(sim._kwargs["aperture"])
        else:
            aperture = 30
        get_HIH2_masses(ctx, aperture=aperture)
        _get_aperture_quan(ctx, aperture=aperture)
        if "half_stellar_radius_property" in sim._kwargs:
            _get_aperture_quan(
                ctx,
                aperture=np.asarray([i.radii["stellar_half_mass"].value for i in groups]),
                aptname="stellar_half_mass_radius",
            )


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


def _fof_for_batch(
    sim,
    *,
    batch: AHFSubhaloBatch,
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
    device_id: Optional[int] = None,
) -> Dict[int, List]:
    from caesar.fof6d_graph import fof6d_on_pool_graph

    resolved_backend = str(backend).lower()
    if resolved_backend == "cpu":
        resolved_backend = "numpy"
    elif resolved_backend == "gpu":
        resolved_backend = "cupy"

    if resolved_backend == "cupy" and device_id is not None:
        from caesar.fof6d_graph import _try_import_cupy

        cp = _try_import_cupy()
        if cp is None:
            raise RuntimeError("GPU batch requested but CuPy/CUDA is unavailable")
        cp.cuda.Device(int(device_id)).use()

    entry_data = []
    concat_blocks = []
    owner_blocks = []
    particle_offsets = []

    for owner_idx, task in enumerate(batch.tasks):
        gas_sel_dense, star_sel, bh_sel, dm_sel = _resolve_node_membership(
            node_id=int(task.node_id),
            membership_arrays=membership_arrays,
            pid_maps_sel=pid_maps_sel,
            sim=sim,
            fof_nHlim=float(fof_nHlim),
            fof_Tlim=float(fof_Tlim),
            fof_use_sfr_gate=bool(fof_use_sfr_gate),
        )
        if int(star_sel.size) < int(min_stars):
            entry_data.append(None)
            continue

        gas_concat = sim.data_manager.selected_to_concat("gas", np.asarray(gas_sel_dense, dtype=np.int64))
        star_concat = sim.data_manager.selected_to_concat("star", np.asarray(star_sel, dtype=np.int64))
        if bh_sel.size > 0:
            bh_concat = sim.data_manager.selected_to_concat("bh", np.asarray(bh_sel, dtype=np.int64))
        else:
            bh_concat = np.empty(0, dtype=np.int64)

        eligible_concat = np.concatenate((gas_concat, star_concat, bh_concat), axis=None).astype(np.int64, copy=False)
        if eligible_concat.size < int(min_stars):
            entry_data.append(None)
            continue

        entry = {
            "task": task,
            "gas_sel": np.asarray(gas_sel_dense, dtype=np.int32),
            "star_sel": np.asarray(star_sel, dtype=np.int32),
            "bh_sel": np.asarray(bh_sel, dtype=np.int32),
            "dm_sel": np.asarray(dm_sel, dtype=np.int32),
            "ng": int(gas_sel_dense.size),
            "ns": int(star_sel.size),
            "nb": int(bh_sel.size),
            "offset": int(sum(block.size for block in concat_blocks)),
            "size": int(eligible_concat.size),
            "owner_idx": int(owner_idx),
        }
        entry_data.append(entry)
        concat_blocks.append(eligible_concat)
        owner_blocks.append(np.full(int(eligible_concat.size), int(owner_idx), dtype=np.int32))

    out: Dict[int, List] = {int(task.node_id): [] for task in batch.tasks}
    if not concat_blocks:
        return out

    combined_concat = np.concatenate(concat_blocks, axis=None).astype(np.int64, copy=False)
    combined_owner = np.concatenate(owner_blocks, axis=None).astype(np.int32, copy=False)

    tags, _ = fof6d_on_pool_graph(
        sim.data_manager.pos[combined_concat],
        sim.data_manager.vel[combined_concat],
        fof_ll=float(fof_ll),
        vel_ll=fof_vel_ll,
        mingrp=int(min_stars),
        periodic=False,
        kernel="caesar_table",
        backend=resolved_backend,
        cc_backend="auto" if cc_backend == "auto" else cc_backend,
        max_pairs_per_batch=int(max_pairs_per_batch),
    )
    tags = np.asarray(tags, dtype=np.int64)
    valid_tags = np.unique(tags[tags >= 0])
    if valid_tags.size == 0:
        return out

    comps_by_owner: Dict[int, List[int]] = {}
    for comp in valid_tags:
        comp_mask = tags == int(comp)
        owners = np.unique(combined_owner[comp_mask])
        if owners.size != 1:
            comp_task_ids = [int(batch.tasks[int(i)].node_id) for i in owners.tolist()]
            raise AssertionError(
                "AHF-subhalo invariant violated: graph component spans multiple batched halos "
                f"{comp_task_ids}"
            )
        owner_idx = int(owners[0])
        comps_by_owner.setdefault(owner_idx, []).append(int(comp))

    for entry in entry_data:
        if entry is None:
            continue
        owner_idx = int(entry["owner_idx"])
        comps = comps_by_owner.get(owner_idx, [])
        if not comps:
            continue
        offset = int(entry["offset"])
        size = int(entry["size"])
        ng = int(entry["ng"])
        ns = int(entry["ns"])
        nb = int(entry["nb"])
        task = entry["task"]
        gas_sel = entry["gas_sel"]
        star_sel = entry["star_sel"]
        bh_sel = entry["bh_sel"]
        dm_sel = entry["dm_sel"]
        allow_dm = len(comps) == 1

        local_tags = tags[offset:offset + size]
        for comp in comps:
            local_mask = local_tags == int(comp)
            gsub = gas_sel[local_mask[:ng]] if ng > 0 else np.empty(0, dtype=np.int32)
            ssub = star_sel[local_mask[ng:ng + ns]] if ns > 0 else np.empty(0, dtype=np.int32)
            bsub = bh_sel[local_mask[ng + ns:ng + ns + nb]] if nb > 0 else np.empty(0, dtype=np.int32)
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
            out[int(task.node_id)].append(grp)

    return out


def _assign_batches_to_resources(
    *,
    tiny_batches: Sequence[AHFSubhaloBatch],
    regular_tasks: Sequence[AHFSubhaloTask],
    gpu_device_ids: Sequence[int],
    cpu_slots: int,
) -> Tuple[List[AHFSubhaloBatch], List[AHFSubhaloBatch]]:
    work_items: List[AHFSubhaloBatch] = []
    for task in regular_tasks:
        work_items.append(
            AHFSubhaloBatch(
                tasks=(task,),
                is_tiny_batch=False,
                target_backend="cpu",
                target_device=None,
                estimated_cost=int(task.fof_candidates),
            )
        )
    work_items.extend(tiny_batches)
    work_items.sort(key=lambda b: int(b.estimated_cost), reverse=True)

    gpu_items: List[AHFSubhaloBatch] = []
    cpu_items: List[AHFSubhaloBatch] = []
    if gpu_device_ids:
        gpu_loads = {int(dev): 0 for dev in gpu_device_ids}
        for item in work_items:
            dev = min(gpu_loads, key=lambda d: gpu_loads[d])
            gpu_loads[int(dev)] += int(item.estimated_cost)
            gpu_items.append(
                AHFSubhaloBatch(
                    tasks=item.tasks,
                    is_tiny_batch=bool(item.is_tiny_batch),
                    target_backend="gpu",
                    target_device=int(dev),
                    estimated_cost=int(item.estimated_cost),
                )
            )
    else:
        cpu_items = work_items

    if cpu_slots > 0 and gpu_items:
        gpu_items.sort(key=lambda b: int(b.estimated_cost), reverse=True)
        migrated = min(len(gpu_items), max(0, int(cpu_slots)))
        cpu_items.extend(
            AHFSubhaloBatch(
                tasks=item.tasks,
                is_tiny_batch=bool(item.is_tiny_batch),
                target_backend="cpu",
                target_device=None,
                estimated_cost=int(item.estimated_cost),
            )
            for item in gpu_items[:migrated]
        )
        gpu_items = gpu_items[migrated:]

    return gpu_items, cpu_items


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


def _finalize_subhalo_galaxies(
    sim,
    *,
    ahf_particles_file: str,
    galaxy_list: Sequence,
    pid_maps_sel,
):
    from caesar.group import get_group_properties as _get_group_properties
    from caesar.halo_matching import _ensure_missing_ahf_halos, _prune_halos_after_galaxies, _update_ahf_galaxy_maps

    sim.galaxy_list = list(galaxy_list)
    sim.galaxies = sim.galaxy_list
    sim.ngalaxies = len(sim.galaxy_list)
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


def _prepare_final_subhalo_galaxies(
    sim,
    *,
    ahf_particles_file: str,
    galaxy_list: Sequence,
    pid_maps_sel,
):
    from caesar.halo_matching import _ensure_missing_ahf_halos, _prune_halos_after_galaxies, _update_ahf_galaxy_maps

    sim.galaxy_list = list(galaxy_list)
    sim.galaxies = sim.galaxy_list
    sim.ngalaxies = len(sim.galaxy_list)
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


def _complete_finalization_after_properties(sim):
    import caesar.assignment as assign
    import caesar.linking as link
    from caesar.group import sort_groups
    from caesar.halo_matching import _update_ahf_galaxy_maps
    from caesar.utils import calculate_local_densities

    for idx, halo in enumerate(sim.halo_list):
        halo._old_halo_index = int(idx)
    sort_groups(sim.halo_list, "total")
    sim.halos = sim.halo_list
    sim.nhalos = len(sim.halo_list)
    old_to_new = {int(getattr(halo, "_old_halo_index", -1)): int(halo.GroupID) for halo in sim.halo_list}
    for halo in sim.halo_list:
        if hasattr(halo, "_old_halo_index"):
            delattr(halo, "_old_halo_index")
    for gal in sim.galaxy_list:
        old = int(getattr(gal, "parent_halo_index", -1))
        new = int(old_to_new.get(old, -1))
        gal.parent_halo_index = new
        gal._ahf_host_halo_index = new

    calculate_local_densities(sim, sim.halo_list)

    sort_groups(sim.galaxy_list, "stellar")
    sim.galaxies = sim.galaxy_list
    sim.ngalaxies = len(sim.galaxy_list)
    calculate_local_densities(sim, sim.galaxy_list)

    sim._ahf_galaxy_hosts = [int(getattr(gal, "parent_halo_index", -1)) for gal in sim.galaxy_list]
    sim._ahf_galaxy_ahf_ids = [int(getattr(gal, "AHF_haloID", -1)) for gal in sim.galaxy_list]
    sim._ahf_galaxy_top_ahf_ids = [int(getattr(gal, "AHF_top_haloID", -1)) for gal in sim.galaxy_list]
    _update_ahf_galaxy_maps(sim, sim._ahf_galaxy_ahf_ids)

    assign.assign_galaxies_to_halos(sim)
    assign.assign_clouds_to_galaxies(sim)
    link.link_galaxies_and_halos(sim)
    link.link_clouds_and_galaxies(sim)
    assign.assign_central_galaxies(sim)
    link.create_sublists(sim)

    try:
        if "galaxy" not in sim.group_types:
            sim.group_types.append("galaxy")
    except Exception:
        pass

    reset_global_particle_IDs(sim)
    load_global_lists(sim)


def build_galaxies_from_ahf_subhalo(
    sim,
    ahf_particles_file: str,
    *,
    min_stars: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> None:
    import os as _os
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

    root_limit = max(0, _env_int("CAESAR_AHF_SUBHALO_ROOT_LIMIT", 0))
    task_limit = max(0, _env_int("CAESAR_AHF_SUBHALO_TASK_LIMIT", 0))
    if root_limit > 0:
        keep_roots = set(sorted(tasks_by_root.keys())[: int(root_limit)])
        tasks_by_root = {int(root): list(tasks_by_root[int(root)]) for root in sorted(keep_roots)}
        tasks = [task for task in tasks if int(task.top_id) in keep_roots]
    if task_limit > 0 and len(tasks) > int(task_limit):
        keep_ids = {int(task.node_id) for task in tasks[: int(task_limit)]}
        tasks = [task for task in tasks if int(task.node_id) in keep_ids]
        tasks_by_root = {
            int(root): [task for task in root_tasks if int(task.node_id) in keep_ids]
            for root, root_tasks in tasks_by_root.items()
        }
        tasks_by_root = {int(root): root_tasks for root, root_tasks in tasks_by_root.items() if root_tasks}

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

    tiny_star_threshold = max(int(min_stars), _env_int("CAESAR_AHF_SUBHALO_TINY_STARS", 32))
    tiny_max_nodes_per_batch = max(1, _env_int("CAESAR_AHF_SUBHALO_TINY_MAX_NODES", 64))
    tiny_max_fof_candidates_per_batch = max(
        1,
        _env_int("CAESAR_AHF_SUBHALO_TINY_MAX_FOF_CANDIDATES", 250_000),
    )

    tiny_tasks = [task for task in tasks if int(task.star_count) <= int(tiny_star_threshold)]
    regular_tasks = [task for task in tasks if int(task.star_count) > int(tiny_star_threshold)]
    tiny_batches = _build_tiny_batches(
        tiny_tasks,
        max_nodes_per_batch=int(tiny_max_nodes_per_batch),
        max_fof_candidates_per_batch=int(tiny_max_fof_candidates_per_batch),
    )

    gpu_device_ids: List[int] = []
    if backend in {"auto", "cupy"}:
        gpu_device_ids = _available_gpu_device_ids()
    cpu_slots = 0 if gpu_device_ids else int(jobs)
    gpu_items, cpu_items = _assign_batches_to_resources(
        tiny_batches=tiny_batches,
        regular_tasks=regular_tasks,
        gpu_device_ids=gpu_device_ids,
        cpu_slots=int(cpu_slots),
    )

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
    for task in tasks:
        initial_candidates_by_node[int(task.node_id)] = []

    executors: Dict[Tuple[str, Optional[int]], ThreadPoolExecutor] = {}
    try:
        futures: Dict[Future, AHFSubhaloBatch] = {}

        for item in gpu_items:
            key = ("gpu", int(item.target_device))
            executor = executors.get(key)
            if executor is None:
                executor = ThreadPoolExecutor(max_workers=1)
                executors[key] = executor
            futures[
                executor.submit(
                    _fof_for_batch,
                    sim,
                    batch=item,
                    membership_arrays=membership_arrays,
                    pid_maps_sel=pid_maps_sel,
                    min_stars=int(min_stars),
                    fof_ll=float(fof_ll),
                    fof_vel_ll=fof_vel_ll,
                    fof_nHlim=float(fof_nHlim),
                    fof_Tlim=float(fof_Tlim),
                    fof_use_sfr_gate=bool(fof_use_sfr_gate),
                    backend="cupy",
                    cc_backend=cc_backend,
                    max_pairs_per_batch=int(max_pairs_per_batch),
                    device_id=item.target_device,
                )
            ] = item

        if cpu_items:
            cpu_workers = max(1, min(int(jobs), len(cpu_items)))
            cpu_executor = ThreadPoolExecutor(max_workers=cpu_workers)
            executors[("cpu", None)] = cpu_executor
            for item in cpu_items:
                futures[
                    cpu_executor.submit(
                        _fof_for_batch,
                        sim,
                        batch=item,
                        membership_arrays=membership_arrays,
                        pid_maps_sel=pid_maps_sel,
                        min_stars=int(min_stars),
                        fof_ll=float(fof_ll),
                        fof_vel_ll=fof_vel_ll,
                        fof_nHlim=float(fof_nHlim),
                        fof_Tlim=float(fof_Tlim),
                        fof_use_sfr_gate=bool(fof_use_sfr_gate),
                        backend="numpy",
                        cc_backend="cpu",
                        max_pairs_per_batch=int(max_pairs_per_batch),
                        device_id=None,
                    )
                ] = item

        for fut in as_completed(futures):
            item = futures[fut]
            result = fut.result()
            for task in item.tasks:
                initial_candidates_by_node[int(task.node_id)] = list(result.get(int(task.node_id), []))
            if task_progress is not None:
                task_progress.update(len(item.tasks))
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True)
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

    _finalize_subhalo_galaxies(
        sim,
        ahf_particles_file=ahf_particles_file,
        galaxy_list=final_galaxies,
        pid_maps_sel=pid_maps_sel,
    )


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
