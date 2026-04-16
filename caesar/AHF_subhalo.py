from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from yt.funcs import mylog

from caesar.pipeline_utils import reset_global_particle_IDs, load_global_lists

MINIMUM_DM_PER_TOPLEVEL_AHF_HALO = 64
MINIMUM_DM_PER_AHF_SUBHALO = 24


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


def _serialize_task(task: AHFSubhaloTask) -> Dict[str, int | Tuple[int, ...]]:
    return {
        "node_id": int(task.node_id),
        "parent_id": int(task.parent_id),
        "top_id": int(task.top_id),
        "depth": int(task.depth),
        "ancestors": tuple(int(v) for v in task.ancestors),
        "star_count": int(task.star_count),
        "fof_candidates": int(task.fof_candidates),
    }


def _deserialize_task(payload: Dict[str, int | Tuple[int, ...]]) -> AHFSubhaloTask:
    return AHFSubhaloTask(
        node_id=int(payload["node_id"]),
        parent_id=int(payload["parent_id"]),
        top_id=int(payload["top_id"]),
        depth=int(payload["depth"]),
        ancestors=tuple(int(v) for v in payload["ancestors"]),
        star_count=int(payload["star_count"]),
        fof_candidates=int(payload["fof_candidates"]),
    )


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


def _preferred_cc_backend(requested: str, *, backend: str) -> str:
    req = str(requested).lower()
    bkd = str(backend).lower()
    if req != "auto":
        return req
    if bkd in {"cpu", "numpy"}:
        return "cpu"
    if bkd in {"gpu", "cupy"}:
        try:
            from caesar.fof6d_graph import _try_import_cugraph

            if _try_import_cugraph() is not None:
                return "cugraph"
        except Exception:
            pass
        return "gpu"
    return "auto"


def _build_node_dm_counts(membership_arrays: Dict[int, np.ndarray]) -> Dict[int, int]:
    node_ndm: Dict[int, int] = {}
    for node_id, arr in membership_arrays.items():
        block = np.asarray(arr, dtype=np.int64)
        if block.size == 0:
            node_ndm[int(node_id)] = 0
            continue
        if block.ndim != 2 or block.shape[1] != 2:
            block = block.reshape(-1, 2)
        node_ndm[int(node_id)] = int(np.count_nonzero(block[:, 1] == 1))
    return node_ndm


def _effective_resolution_from_ndm(ndm: int) -> int:
    ndm = int(ndm)
    if ndm <= 0:
        return 0
    return max(1, int(round(float(ndm) ** (1.0 / 3.0))))


def _mean_interparticle_separation_from_boxsize(boxsize: float, ndm: int) -> float:
    effective_resolution = int(_effective_resolution_from_ndm(ndm))
    if effective_resolution <= 0:
        return 0.0
    return float(boxsize) / float(effective_resolution)


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
    node_ndm: Optional[Dict[int, int]] = None,
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
            if node_ndm is not None:
                dm_count = int(node_ndm.get(int(node_id), 0))
                min_dm = (
                    int(MINIMUM_DM_PER_TOPLEVEL_AHF_HALO)
                    if int(parent_of.get(int(node_id), 0) or 0) <= 0
                    else int(MINIMUM_DM_PER_AHF_SUBHALO)
                )
                if dm_count < int(min_dm):
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


def _build_task_manifest_from_direct_state(
    state,
    *,
    min_stars: int,
) -> Tuple[List[AHFSubhaloTask], Dict[int, List[AHFSubhaloTask]]]:
    tasks: List[AHFSubhaloTask] = []
    tasks_by_root: Dict[int, List[AHFSubhaloTask]] = {}
    for idx in range(len(state.nodes)):
        node_id = int(state.nodes.halo_id[idx])
        parent_id = int(state.nodes.parent_halo_id[idx])
        top_id = int(state.nodes.top_halo_id[idx])
        depth = int(state.nodes.depth[idx])
        star_count = int(state.nodes.star_count[idx])
        if star_count < int(min_stars):
            continue
        dm_count = int(state.nodes.dm_count[idx])
        min_dm = (
            int(MINIMUM_DM_PER_TOPLEVEL_AHF_HALO)
            if int(parent_id) <= 0
            else int(MINIMUM_DM_PER_AHF_SUBHALO)
        )
        if dm_count < int(min_dm):
            continue
        task = AHFSubhaloTask(
            node_id=node_id,
            parent_id=parent_id,
            top_id=top_id,
            depth=depth,
            ancestors=tuple(int(v) for v in state.nodes.ancestors_for(idx).tolist()),
            star_count=star_count,
            fof_candidates=int(max(1, int(state.nodes.fof_candidates[idx]))),
        )
        tasks.append(task)
        tasks_by_root.setdefault(int(top_id), []).append(task)

    tasks.sort(key=lambda task: (int(task.top_id), int(task.depth), int(task.node_id)))
    for root_id in list(tasks_by_root.keys()):
        tasks_by_root[int(root_id)] = sorted(
            tasks_by_root[int(root_id)],
            key=lambda task: (int(task.depth), int(task.node_id)),
        )
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


def _snapshot_file_from_obj(obj) -> str:
    if hasattr(obj, "_kwargs") and obj._kwargs.get("snapshot_file"):
        return str(obj._kwargs["snapshot_file"])

    ds = getattr(obj, "_ds", None)
    if ds not in (None, 0):
        directory = getattr(ds, "directory", None)
        basename = getattr(ds, "basename", None)
        if directory and basename:
            return os.path.join(str(directory), str(basename))
        fullpath = getattr(ds, "fullpath", None)
        if fullpath and basename:
            return os.path.join(str(fullpath), str(basename))
        if fullpath and os.path.isfile(str(fullpath)):
            return str(fullpath)

    try:
        ds = obj.yt_dataset
        directory = getattr(ds, "directory", None)
        basename = getattr(ds, "basename", None)
        if directory and basename:
            return os.path.join(str(directory), str(basename))
    except Exception:
        pass

    raise ValueError("AHF-subhalo direct path could not determine the snapshot file from the CAESAR object.")


def _direct_fof_linking_length(snapshot, *, kwargs: Optional[Dict[str, object]] = None) -> float:
    ndm = int(snapshot.particle_counts.get("dm", 0))
    if ndm <= 0:
        raise RuntimeError("AHF-subhalo direct runtime requires dark-matter particles to compute fof_ll")

    opts = dict(kwargs or {})
    b = 0.2
    if isinstance(opts.get("b_halo"), (int, float)):
        b = float(opts["b_halo"])
    if isinstance(opts.get("b_galaxy"), (int, float)):
        b = float(opts["b_galaxy"])
    else:
        b *= 0.1

    mis = float(snapshot.boxsize) / float(ndm) ** (1.0 / 3.0)
    return float(mis * b)


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


def _candidate_payload(
    *,
    task: AHFSubhaloTask,
    star_sel: np.ndarray,
    gas_sel: np.ndarray,
    bh_sel: np.ndarray,
    dm_sel: np.ndarray,
    merge_id: Optional[int] = None,
    ahf_host_halo_index: int = -1,
) -> Dict[str, np.ndarray | int]:
    payload: Dict[str, np.ndarray | int] = {
        "AHF_haloID": int(task.node_id),
        "AHF_parent_haloID": int(task.parent_id),
        "AHF_top_haloID": int(task.top_id),
        "AHF_depth": int(task.depth),
        "AHF_ancestor_haloIDs": np.asarray(task.ancestors, dtype=np.int64),
        "slist": np.asarray(star_sel, dtype=np.int32),
        "glist": np.asarray(gas_sel, dtype=np.int32),
        "bhlist": np.asarray(bh_sel, dtype=np.int32),
        "dmlist": np.asarray(dm_sel, dtype=np.int32),
        "_ahf_host_halo_index": int(ahf_host_halo_index),
    }
    if merge_id is not None:
        payload["_merge_id"] = int(merge_id)
    return payload


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


def _remap_index_array(values, mapping: Dict[int, int], *, dtype=np.int64) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if arr.size == 0:
        return np.empty(0, dtype=dtype)
    return np.asarray([mapping[int(v)] for v in arr.tolist()], dtype=dtype)


def _concat_unique_index_arrays(values: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(v, dtype=np.int64) for v in values if v is not None and np.asarray(v).size > 0]
    if not arrays:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(arrays).astype(np.int64, copy=False))


class _ShardYTUnitHelper:
    def __init__(self, *, redshift: float = 0.0):
        self.scale_factor = 1.0 / (1.0 + max(float(redshift), 0.0))
        from yt.units.yt_array import UnitRegistry
        from yt.units.yt_array import YTQuantity
        from unyt.dimensions import length

        self.unit_registry = UnitRegistry()
        for symbol, base_unit in (("pccm", "pc"), ("kpccm", "kpc"), ("Mpccm", "Mpc")):
            base_value = float((YTQuantity(1.0, base_unit) * self.scale_factor).to("cm").value)
            self.unit_registry.add(symbol, base_value, length, tex_repr=f"\\rm{{{symbol}}}")

    def quan(self, value, unit):
        from yt.units.yt_array import YTQuantity

        u = "dimensionless" if unit in (None, "") else str(unit)
        return YTQuantity(value, u, registry=self.unit_registry)

    def arr(self, value, unit):
        from yt.units.yt_array import YTArray

        u = "dimensionless" if unit in (None, "") else str(unit)
        return YTArray(value, u, registry=self.unit_registry)


class _ShardDatasetType:
    def __init__(self, *, ptypes: Sequence[str], data_manager_attrs: Set[str]):
        self._ptypes = {str(p) for p in ptypes}
        self._attrs = set(str(v) for v in data_manager_attrs)

    def has_ptype(self, requested_ptype):
        return str(requested_ptype).lower() in self._ptypes

    def has_property(self, requested_ptype, requested_prop):
        ptype = str(requested_ptype).lower()
        prop = str(requested_prop).lower()
        mapping = {
            ("gas", "fh2"): "gfH2",
            ("gas", "nh"): "gfHI",
            ("gas", "rho"): "gnh",
            ("bh", "bhmdot"): "bhmdot",
            ("bh", "bhmass"): "bhmass",
            ("gas", "dustmass"): "dustmass",
        }
        attr = mapping.get((ptype, prop))
        return attr in self._attrs if attr is not None else False


def _serialize_stage3_group_input(group, *, galaxy_index_map: Optional[Dict[int, int]] = None) -> Dict:
    payload = {
        "AHF_haloID": int(getattr(group, "AHF_haloID", -1)),
        "AHF_parent_haloID": int(getattr(group, "AHF_parent_haloID", -1)) if hasattr(group, "AHF_parent_haloID") else -1,
        "AHF_top_haloID": int(getattr(group, "AHF_top_haloID", -1)) if hasattr(group, "AHF_top_haloID") else -1,
        "AHF_depth": int(getattr(group, "AHF_depth", 0)) if hasattr(group, "AHF_depth") else 0,
        "AHF_ancestor_haloIDs": np.asarray(getattr(group, "AHF_ancestor_haloIDs", []), dtype=np.int64),
        "global_indexes": np.asarray(getattr(group, "global_indexes", []), dtype=np.int64),
        "glist": np.asarray(getattr(group, "glist", []), dtype=np.int64),
        "slist": np.asarray(getattr(group, "slist", []), dtype=np.int64),
        "dmlist": np.asarray(getattr(group, "dmlist", []), dtype=np.int64),
        "bhlist": np.asarray(getattr(group, "bhlist", []), dtype=np.int64),
        "dlist": np.asarray(getattr(group, "dlist", []), dtype=np.int64),
    }
    if hasattr(group, "_merge_id"):
        payload["_merge_id"] = int(getattr(group, "_merge_id"))
    if hasattr(group, "parent_halo_index"):
        payload["parent_halo_index"] = int(getattr(group, "parent_halo_index"))
    if hasattr(group, "_ahf_host_halo_index"):
        payload["_ahf_host_halo_index"] = int(getattr(group, "_ahf_host_halo_index"))
    if galaxy_index_map is not None and hasattr(group, "galaxy_index_list"):
        payload["galaxy_index_list"] = np.asarray(
            [galaxy_index_map[int(v)] for v in np.asarray(getattr(group, "galaxy_index_list", []), dtype=np.int64).tolist()],
            dtype=np.int32,
        )
    return payload


def _build_stage3_property_payload(sim, halos: Sequence) -> Dict[str, object]:
    halos = list(halos)
    halo_index_map = {int(getattr(halo, "AHF_haloID", -1)): int(i) for i, halo in enumerate(halos)}

    def _scalar(value, default=0.0):
        raw = getattr(value, "d", value)
        try:
            return float(raw)
        except Exception:
            return float(default)

    def _vector(value, default):
        raw = getattr(value, "d", value)
        try:
            return np.asarray(raw, dtype=np.float64)
        except Exception:
            return np.asarray(default, dtype=np.float64)

    galaxy_list = []
    seen_galaxy_index = set()
    galaxy_index_map: Dict[int, int] = {}
    for halo in halos:
        for gi in np.asarray(getattr(halo, "galaxy_index_list", []), dtype=np.int64).tolist():
            old_idx = int(gi)
            if old_idx in seen_galaxy_index:
                continue
            seen_galaxy_index.add(old_idx)
            galaxy_index_map[old_idx] = len(galaxy_list)
            gal = sim.galaxy_list[old_idx]
            galaxy_list.append(gal)

    global_ids = _concat_unique_index_arrays(
        [getattr(halo, "global_indexes", np.empty(0, dtype=np.int64)) for halo in halos]
        + [getattr(gal, "global_indexes", np.empty(0, dtype=np.int64)) for gal in galaxy_list]
    )
    gas_ids = _concat_unique_index_arrays(
        [getattr(halo, "glist", np.empty(0, dtype=np.int64)) for halo in halos]
        + [getattr(gal, "glist", np.empty(0, dtype=np.int64)) for gal in galaxy_list]
    )
    star_ids = _concat_unique_index_arrays(
        [getattr(halo, "slist", np.empty(0, dtype=np.int64)) for halo in halos]
        + [getattr(gal, "slist", np.empty(0, dtype=np.int64)) for gal in galaxy_list]
    )
    dm_ids = _concat_unique_index_arrays(
        [getattr(halo, "dmlist", np.empty(0, dtype=np.int64)) for halo in halos]
        + [getattr(gal, "dmlist", np.empty(0, dtype=np.int64)) for gal in galaxy_list]
    )
    bh_ids = _concat_unique_index_arrays(
        [getattr(halo, "bhlist", np.empty(0, dtype=np.int64)) for halo in halos]
        + [getattr(gal, "bhlist", np.empty(0, dtype=np.int64)) for gal in galaxy_list]
    )
    dust_ids = _concat_unique_index_arrays(
        [getattr(halo, "dlist", np.empty(0, dtype=np.int64)) for halo in halos]
        + [getattr(gal, "dlist", np.empty(0, dtype=np.int64)) for gal in galaxy_list]
    )

    global_map = {int(v): i for i, v in enumerate(global_ids.tolist())}
    gas_map = {int(v): i for i, v in enumerate(gas_ids.tolist())}
    star_map = {int(v): i for i, v in enumerate(star_ids.tolist())}
    dm_map = {int(v): i for i, v in enumerate(dm_ids.tolist())}
    bh_map = {int(v): i for i, v in enumerate(bh_ids.tolist())}
    dust_map = {int(v): i for i, v in enumerate(dust_ids.tolist())}

    dmgr = sim.data_manager
    payload_dm: Dict[str, object] = {
        "pos": np.asarray(dmgr.pos[global_ids]),
        "vel": np.asarray(dmgr.vel[global_ids]),
        "mass": np.asarray(dmgr.mass[global_ids]),
        "ptype": np.asarray(dmgr.ptype[global_ids]),
        "pot": np.asarray(dmgr.pot[global_ids]) if hasattr(dmgr, "pot") else np.zeros(len(global_ids), dtype=np.float32),
    }

    ptypes_local: List[str] = []
    if gas_ids.size > 0:
        payload_dm["glist"] = _remap_index_array(np.asarray(dmgr.glist[gas_ids], dtype=np.int64), global_map, dtype=np.int64)
        for attr in ("gnh", "gsfr", "gZ", "gT", "gfH2", "gfHI", "dustmass"):
            if hasattr(dmgr, attr):
                payload_dm[attr] = np.asarray(getattr(dmgr, attr)[gas_ids])
        ptypes_local.append("gas")
    if star_ids.size > 0:
        payload_dm["slist"] = _remap_index_array(np.asarray(dmgr.slist[star_ids], dtype=np.int64), global_map, dtype=np.int64)
        for attr in ("sZ", "age"):
            if hasattr(dmgr, attr):
                payload_dm[attr] = np.asarray(getattr(dmgr, attr)[star_ids])
        ptypes_local.append("star")
    if dm_ids.size > 0:
        payload_dm["dmlist"] = _remap_index_array(np.asarray(dmgr.dmlist[dm_ids], dtype=np.int64), global_map, dtype=np.int64)
        ptypes_local.append("dm")
    if bh_ids.size > 0:
        payload_dm["bhlist"] = _remap_index_array(np.asarray(dmgr.bhlist[bh_ids], dtype=np.int64), global_map, dtype=np.int64)
        for attr in ("bhmass", "bhmdot"):
            if hasattr(dmgr, attr):
                payload_dm[attr] = np.asarray(getattr(dmgr, attr)[bh_ids])
        ptypes_local.append("bh")
    if dust_ids.size > 0 and hasattr(dmgr, "dlist"):
        payload_dm["dlist"] = _remap_index_array(np.asarray(dmgr.dlist[dust_ids], dtype=np.int64), global_map, dtype=np.int64)
        ptypes_local.append("dust")

    halo_payloads = []
    for halo in halos:
        rec = _serialize_stage3_group_input(halo, galaxy_index_map=galaxy_index_map)
        rec["global_indexes"] = _remap_index_array(rec["global_indexes"], global_map, dtype=np.int64)
        rec["glist"] = _remap_index_array(rec["glist"], gas_map, dtype=np.int64)
        rec["slist"] = _remap_index_array(rec["slist"], star_map, dtype=np.int64)
        rec["dmlist"] = _remap_index_array(rec["dmlist"], dm_map, dtype=np.int64)
        rec["bhlist"] = _remap_index_array(rec["bhlist"], bh_map, dtype=np.int64)
        rec["dlist"] = _remap_index_array(rec["dlist"], dust_map, dtype=np.int64)
        halo_payloads.append(rec)

    galaxy_payloads = []
    for gal in galaxy_list:
        rec = _serialize_stage3_group_input(gal)
        rec["global_indexes"] = _remap_index_array(rec["global_indexes"], global_map, dtype=np.int64)
        rec["glist"] = _remap_index_array(rec["glist"], gas_map, dtype=np.int64)
        rec["slist"] = _remap_index_array(rec["slist"], star_map, dtype=np.int64)
        rec["dmlist"] = _remap_index_array(rec["dmlist"], dm_map, dtype=np.int64)
        rec["bhlist"] = _remap_index_array(rec["bhlist"], bh_map, dtype=np.int64)
        rec["dlist"] = _remap_index_array(rec["dlist"], dust_map, dtype=np.int64)
        old_parent_hid = int(getattr(sim.halo_list[int(rec["parent_halo_index"])], "AHF_haloID", -1))
        rec["parent_halo_index"] = int(halo_index_map.get(old_parent_hid, -1))
        rec["_ahf_host_halo_index"] = int(rec["parent_halo_index"])
        galaxy_payloads.append(rec)

    boxsize_val = getattr(getattr(sim.simulation, "boxsize", None), "d", getattr(sim.simulation, "boxsize", 0.0))
    ndm_val = int(getattr(sim.simulation, "ndm", 0))
    simulation_payload = {
        "XH": float(getattr(sim.simulation, "XH", 0.76)),
        "redshift": float(getattr(sim.simulation, "redshift", 0.0)),
        "omega_baryon": float(getattr(sim.simulation, "omega_baryon", 0.0)),
        "omega_matter": float(getattr(sim.simulation, "omega_matter", 0.0)),
        "omega_lambda": float(getattr(sim.simulation, "omega_lambda", 0.0)),
        "Om_z": float(getattr(sim.simulation, "Om_z", getattr(sim.simulation, "omega_matter", 0.0))),
        "hubble_constant": float(getattr(sim.simulation, "hubble_constant", 0.0)),
        "boxsize": float(boxsize_val),
        "critical_density": _scalar(getattr(sim.simulation, "critical_density", 0.0)),
        "G": _scalar(getattr(sim.simulation, "G", 4.51691362044e-39)),
        "H_z": _scalar(getattr(sim.simulation, "H_z", 0.0)),
        "Densities": _vector(
            getattr(sim.simulation, "Densities", np.asarray([0.0, 0.0, 0.0], dtype=np.float64)),
            [0.0, 0.0, 0.0],
        ),
        "ngas": int(gas_ids.size),
        "nstar": int(star_ids.size),
        "nbh": int(bh_ids.size),
        "ndust": int(dust_ids.size),
        "ndm": ndm_val,
        "ndm2": int(getattr(sim.simulation, "ndm2", 0)),
        "ndm3": int(getattr(sim.simulation, "ndm3", 0)),
        "ntot": int(global_ids.size),
        "baryons_present": bool(getattr(sim.simulation, "baryons_present", gas_ids.size > 0 or star_ids.size > 0)),
        "unbind_halos": bool(getattr(sim.simulation, "unbind_halos", False)),
        "effective_resolution": int(
            getattr(sim.simulation, "effective_resolution", _effective_resolution_from_ndm(ndm_val))
        ),
        "mean_interparticle_separation": _scalar(
            getattr(
                sim.simulation,
                "mean_interparticle_separation",
                _mean_interparticle_separation_from_boxsize(float(boxsize_val), ndm_val),
            )
        ),
    }

    return {
        "units": dict(getattr(sim, "units", {})),
        "kwargs": dict(getattr(sim, "_kwargs", {})),
        "load_pot": bool(getattr(sim, "load_pot", True)),
        "ptypes": list(ptypes_local),
        "blackholes": bool(getattr(dmgr, "blackholes", False) and bh_ids.size > 0),
        "simulation": simulation_payload,
        "data_manager": payload_dm,
        "halos": halo_payloads,
        "galaxies": galaxy_payloads,
    }


def _build_stage3_property_runtime(payload: Dict[str, object], *, nproc: int):
    from caesar.main import CAESAR
    from caesar.group import create_new_group

    sim = CAESAR()
    sim._kwargs = dict(payload.get("kwargs", {}))
    sim.units = dict(payload.get("units", sim.units))
    sim.load_pot = bool(payload.get("load_pot", True))
    sim.load_haloid = False
    sim.skip_hash_check = True
    sim.nproc = int(max(1, nproc))

    sim_payload = dict(payload.get("simulation", {}))
    sim._ds = _ShardYTUnitHelper(redshift=float(sim_payload.get("redshift", 0.0)))
    sim.simulation.XH = float(sim_payload.get("XH", 0.76))
    sim.simulation.redshift = float(sim_payload.get("redshift", 0.0))
    sim.simulation.omega_baryon = float(sim_payload.get("omega_baryon", 0.0))
    sim.simulation.omega_matter = float(sim_payload.get("omega_matter", 0.0))
    sim.simulation.omega_lambda = float(sim_payload.get("omega_lambda", 0.0))
    sim.simulation.Om_z = float(sim_payload.get("Om_z", sim.simulation.omega_matter))
    sim.simulation.hubble_constant = float(sim_payload.get("hubble_constant", 0.0))
    sim.simulation.boxsize = sim.yt_dataset.quan(float(sim_payload.get("boxsize", 0.0)), sim.units["length"])
    sim.simulation.critical_density = sim.yt_dataset.quan(
        float(sim_payload.get("critical_density", 0.0)),
        "Msun/kpc**3",
    )
    sim.simulation.G = sim.yt_dataset.quan(
        float(sim_payload.get("G", 4.51691362044e-39)),
        "kpc**3/(Msun * s**2)",
    )
    sim.simulation.H_z = sim.yt_dataset.quan(float(sim_payload.get("H_z", 0.0)), "1/s")
    sim.simulation.Densities = sim.yt_dataset.arr(
        np.asarray(sim_payload.get("Densities", [0.0, 0.0, 0.0]), dtype=np.float64),
        "Msun/kpc**3",
    )
    sim.simulation.ngas = int(sim_payload.get("ngas", 0))
    sim.simulation.nstar = int(sim_payload.get("nstar", 0))
    sim.simulation.nbh = int(sim_payload.get("nbh", 0))
    sim.simulation.ndust = int(sim_payload.get("ndust", 0))
    sim.simulation.ndm = int(sim_payload.get("ndm", 0))
    sim.simulation.ndm2 = int(sim_payload.get("ndm2", 0))
    sim.simulation.ndm3 = int(sim_payload.get("ndm3", 0))
    sim.simulation.ntot = int(sim_payload.get("ntot", 0))
    sim.simulation.baryons_present = bool(
        sim_payload.get("baryons_present", sim.simulation.ngas > 0 or sim.simulation.nstar > 0)
    )
    sim.simulation.unbind_halos = bool(sim_payload.get("unbind_halos", False))
    sim.simulation.effective_resolution = int(
        sim_payload.get("effective_resolution", _effective_resolution_from_ndm(sim.simulation.ndm))
    )
    sim.simulation.mean_interparticle_separation = sim.yt_dataset.quan(
        float(
            sim_payload.get(
                "mean_interparticle_separation",
                _mean_interparticle_separation_from_boxsize(
                    float(getattr(sim.simulation.boxsize, "d", sim.simulation.boxsize)),
                    sim.simulation.ndm,
                ),
            )
        ),
        sim.units["length"],
    )

    dm_payload = dict(payload.get("data_manager", {}))
    dm = SimpleNamespace()
    dm.ptypes = list(payload.get("ptypes", []))
    dm.blackholes = bool(payload.get("blackholes", False))
    for key, value in dm_payload.items():
        setattr(dm, key, np.asarray(value))
    sim._dm = dm
    sim._ds_type = _ShardDatasetType(ptypes=dm.ptypes, data_manager_attrs=set(dm_payload.keys()))
    sim.group_types = ["halo", "galaxy"]

    halo_list = []
    for rec in payload.get("halos", []):
        halo = create_new_group(sim, "halo")
        halo.AHF_haloID = int(rec["AHF_haloID"])
        halo.global_indexes = np.asarray(rec["global_indexes"], dtype=np.int64)
        halo.glist = np.asarray(rec["glist"], dtype=np.int64)
        halo.slist = np.asarray(rec["slist"], dtype=np.int64)
        halo.dmlist = np.asarray(rec["dmlist"], dtype=np.int64)
        halo.bhlist = np.asarray(rec["bhlist"], dtype=np.int64)
        halo.dlist = np.asarray(rec["dlist"], dtype=np.int64)
        halo.galaxy_index_list = np.asarray(rec.get("galaxy_index_list", []), dtype=np.int32)
        if "_merge_id" in rec:
            halo._merge_id = int(rec["_merge_id"])
        halo_list.append(halo)

    galaxy_list = []
    for rec in payload.get("galaxies", []):
        gal = create_new_group(sim, "galaxy")
        gal.AHF_haloID = int(rec["AHF_haloID"])
        gal.AHF_parent_haloID = int(rec.get("AHF_parent_haloID", -1))
        gal.AHF_top_haloID = int(rec.get("AHF_top_haloID", -1))
        gal.AHF_depth = int(rec.get("AHF_depth", 0))
        gal.AHF_ancestor_haloIDs = np.asarray(rec.get("AHF_ancestor_haloIDs", []), dtype=np.int64)
        gal.global_indexes = np.asarray(rec["global_indexes"], dtype=np.int64)
        gal.glist = np.asarray(rec["glist"], dtype=np.int64)
        gal.slist = np.asarray(rec["slist"], dtype=np.int64)
        gal.dmlist = np.asarray(rec["dmlist"], dtype=np.int64)
        gal.bhlist = np.asarray(rec["bhlist"], dtype=np.int64)
        gal.dlist = np.asarray(rec["dlist"], dtype=np.int64)
        gal.parent_halo_index = int(rec.get("parent_halo_index", -1))
        gal._ahf_host_halo_index = int(rec.get("_ahf_host_halo_index", gal.parent_halo_index))
        if "_merge_id" in rec:
            gal._merge_id = int(rec["_merge_id"])
        galaxy_list.append(gal)

    sim.halo_list = halo_list
    sim.halos = halo_list
    sim.nhalos = len(halo_list)
    sim.galaxy_list = galaxy_list
    sim.galaxies = galaxy_list
    sim.ngalaxies = len(galaxy_list)

    for gal in galaxy_list:
        idx = int(getattr(gal, "parent_halo_index", -1))
        if 0 <= idx < len(halo_list):
            gal.halo = halo_list[idx]

    return sim


def _raw_scalar(value, default: float = 0.0) -> float:
    raw = getattr(value, "d", getattr(value, "value", value))
    try:
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 0:
            return float(arr)
    except Exception:
        pass
    try:
        return float(raw)
    except Exception:
        return float(default)


def _raw_array(value, *, dtype=np.float64) -> np.ndarray:
    raw = getattr(value, "d", getattr(value, "value", value))
    return np.asarray(raw, dtype=dtype)


def _unit_factor(sim, unit_from: str, unit_to: str) -> float:
    cache = getattr(sim, "_stage3_unit_factor_cache", None)
    if cache is None:
        cache = {}
        sim._stage3_unit_factor_cache = cache
    key = (str(unit_from), str(unit_to))
    if key not in cache:
        cache[key] = float(sim.yt_dataset.quan(1.0, unit_from).to(unit_to).value)
    return float(cache[key])


def _quantity(sim, value, unit: str):
    arr = np.asarray(value)
    if arr.ndim > 0:
        return sim.yt_dataset.arr(arr, unit)
    return sim.yt_dataset.quan(float(arr), unit)


def _quantity_from_key(sim, value, unit_key: str):
    return sim.yt_dataset.quan(value, sim.units[unit_key])


def _array_from_key(sim, value, unit_key: str):
    return sim.yt_dataset.arr(value, sim.units[unit_key])


def _boxsize_raw(sim) -> float:
    return _raw_scalar(getattr(sim.simulation, "boxsize", 0.0), 0.0)


def _periodic_delta(values: np.ndarray, center: np.ndarray, boxsize: float) -> np.ndarray:
    delta = np.asarray(values, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    if float(boxsize) > 0.0:
        delta -= np.round(delta / float(boxsize)) * float(boxsize)
    return delta


def _weighted_periodic_center(pos: np.ndarray, weights: np.ndarray, boxsize: float) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if pos.size == 0:
        return np.zeros(3, dtype=np.float64)
    out = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        vals = pos[:, axis]
        if float(boxsize) > 0.0 and np.max(vals) - np.min(vals) > 0.5 * float(boxsize):
            theta = 2.0 * np.pi * vals / float(boxsize)
            zeta = np.cos(theta)
            xhi = np.sin(theta)
            theta_mean = np.arctan2(-np.average(xhi, weights=weights), -np.average(zeta, weights=weights)) + np.pi
            out[axis] = float(boxsize) * theta_mean / (2.0 * np.pi)
        else:
            out[axis] = np.average(vals, weights=weights)
    return out


def _weighted_velocity_center(vel: np.ndarray, weights: np.ndarray) -> np.ndarray:
    vel = np.asarray(vel, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if vel.size == 0:
        return np.zeros(3, dtype=np.float64)
    return np.average(vel, axis=0, weights=weights)


def _radius_summary(radii: np.ndarray, masses: np.ndarray) -> Tuple[float, float, float, float]:
    radii = np.asarray(radii, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if radii.size == 0 or np.sum(masses) <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    order = np.argsort(radii)
    r_sorted = radii[order]
    m_sorted = masses[order]
    csum = np.cumsum(m_sorted)
    total = float(csum[-1])

    def _qfrac(frac: float) -> float:
        idx = int(np.searchsorted(csum, frac * total, side="left"))
        idx = min(max(idx, 0), len(r_sorted) - 1)
        return float(r_sorted[idx])

    return float(r_sorted[-1]), _qfrac(0.2), _qfrac(0.5), _qfrac(0.8)


def _velocity_dispersion(vel: np.ndarray, masses: np.ndarray) -> float:
    vel = np.asarray(vel, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if vel.size == 0 or np.sum(masses) <= 0.0:
        return 0.0
    center = _weighted_velocity_center(vel, masses)
    dv = vel - center
    variance_1d = np.average(np.sum(dv * dv, axis=1), weights=masses) / 3.0
    return float(np.sqrt(max(variance_1d, 0.0)))


def _component_global_indices(sim, group, component: str) -> np.ndarray:
    dm = sim.data_manager
    attr_map = {
        "gas": ("glist", "glist"),
        "star": ("slist", "slist"),
        "dm": ("dmlist", "dmlist"),
        "bh": ("bhlist", "bhlist"),
        "dust": ("dlist", "dlist"),
    }
    if component not in attr_map:
        return np.empty(0, dtype=np.int64)
    group_attr, dm_attr = attr_map[component]
    local_idx = np.asarray(getattr(group, group_attr, []), dtype=np.int64)
    if local_idx.size == 0 or not hasattr(dm, dm_attr):
        return np.empty(0, dtype=np.int64)
    return np.asarray(getattr(dm, dm_attr)[local_idx], dtype=np.int64)


def _component_local_indices(group, component: str) -> np.ndarray:
    attr_map = {
        "gas": "glist",
        "star": "slist",
        "dm": "dmlist",
        "bh": "bhlist",
        "dust": "dlist",
    }
    if component not in attr_map:
        return np.empty(0, dtype=np.int64)
    return np.asarray(getattr(group, attr_map[component], []), dtype=np.int64)


def _sanitized_hydrogen_fractions(sim, gas_local: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dm = sim.data_manager
    hi_frac = np.clip(_raw_array(dm.gfHI[gas_local], dtype=np.float64), 0.0, 1.0)
    h2_frac = np.clip(_raw_array(dm.gfH2[gas_local], dtype=np.float64), 0.0, 1.0)
    if gas_local.size == 0:
        return hi_frac, h2_frac

    if hasattr(dm, "gnh"):
        gas_nh = _raw_array(dm.gnh[gas_local], dtype=np.float64)
        h2_frac = np.asarray(h2_frac, dtype=np.float64, copy=True)
        h2_frac[gas_nh < 0.13] = 0.0

    total_frac = hi_frac + h2_frac
    if np.any(total_frac > 1.0):
        hi_frac = np.asarray(hi_frac, dtype=np.float64, copy=True)
        hi_frac[total_frac > 1.0] = 1.0 - h2_frac[total_frac > 1.0]
        hi_frac = np.clip(hi_frac, 0.0, 1.0)
    return hi_frac, h2_frac


def _group_aperture_properties(sim, galaxy, aperture: float, *, gal_pos_raw: Optional[np.ndarray] = None) -> Tuple[Dict[str, object], Dict[str, object]]:
    dm = sim.data_manager
    halo = getattr(galaxy, "halo", None)
    if halo is None:
        return {}, {}

    boxsize = _boxsize_raw(sim)
    if gal_pos_raw is None:
        if not hasattr(galaxy, "pos"):
            return {}, {}
        gal_pos_raw = _raw_array(galaxy.pos, dtype=np.float64)
    else:
        gal_pos_raw = np.asarray(gal_pos_raw, dtype=np.float64)
    masses_out: Dict[str, object] = {}
    sigmas_out: Dict[str, object] = {}
    suffix = f"{int(aperture)}kpc" if float(aperture).is_integer() else f"{aperture:g}kpc"
    if aperture != aperture:  # NaN guard
        return masses_out, sigmas_out

    component_names = {
        "gas": "gas",
        "star": "stellar",
        "dm": "dm",
        "bh": "bh",
        "dust": "dust",
    }
    total_mass = 0.0
    for component, out_name in component_names.items():
        local_idx = _component_local_indices(halo, component)
        global_idx = _component_global_indices(sim, halo, component)
        if local_idx.size == 0 or global_idx.size == 0:
            masses_out[f"{out_name}_{suffix}"] = _quantity_from_key(sim, 0.0, "mass")
            sigmas_out[f"{out_name}_{suffix}"] = _quantity_from_key(sim, 0.0, "velocity")
            continue

        pos = _raw_array(dm.pos[global_idx], dtype=np.float64)
        vel = _raw_array(dm.vel[global_idx], dtype=np.float64)
        mass = _raw_array(dm.mass[global_idx], dtype=np.float64)
        radii = np.linalg.norm(_periodic_delta(pos, gal_pos_raw, boxsize), axis=1)
        inside = radii <= float(aperture)
        if not np.any(inside):
            masses_out[f"{out_name}_{suffix}"] = _quantity_from_key(sim, 0.0, "mass")
            sigmas_out[f"{out_name}_{suffix}"] = _quantity_from_key(sim, 0.0, "velocity")
            continue
        mass_sel = mass[inside]
        vel_sel = vel[inside]
        component_mass = float(np.sum(mass_sel))
        total_mass += component_mass
        masses_out[f"{out_name}_{suffix}"] = _quantity_from_key(sim, component_mass, "mass")
        sigmas_out[f"{out_name}_{suffix}"] = _quantity_from_key(sim, _velocity_dispersion(vel_sel, mass_sel), "velocity")

    gas_local = _component_local_indices(halo, "gas")
    gas_global = _component_global_indices(sim, halo, "gas")
    hi_mass = 0.0
    h2_mass = 0.0
    if gas_local.size > 0 and gas_global.size > 0 and hasattr(dm, "gfHI") and hasattr(dm, "gfH2"):
        gas_pos = _raw_array(dm.pos[gas_global], dtype=np.float64)
        gas_mass = _raw_array(dm.mass[gas_global], dtype=np.float64)
        gas_r = np.linalg.norm(_periodic_delta(gas_pos, gal_pos_raw, boxsize), axis=1)
        inside = gas_r <= float(aperture)
        if np.any(inside):
            hi_frac, h2_frac = _sanitized_hydrogen_fractions(sim, gas_local)
            hi_frac = hi_frac[inside]
            h2_frac = h2_frac[inside]
            gas_mass = gas_mass[inside]
            xh = float(getattr(sim.simulation, "XH", 0.76))
            hi_mass = float(np.sum(gas_mass * hi_frac) * xh)
            h2_mass = float(np.sum(gas_mass * h2_frac) * xh)
    masses_out[f"HI_{suffix}"] = _quantity_from_key(sim, hi_mass, "mass")
    masses_out[f"H2_{suffix}"] = _quantity_from_key(sim, h2_mass, "mass")
    masses_out[f"total_{suffix}"] = _quantity_from_key(sim, total_mass, "mass")
    return masses_out, sigmas_out


def _apply_host_local_hydrogen_assignment(sim, galaxies: Sequence) -> None:
    dm = sim.data_manager
    if not galaxies or not hasattr(dm, "gfHI") or not hasattr(dm, "gfH2"):
        return

    aperture = float(sim._kwargs.get("aperture", 30.0))
    aperture2 = float(aperture * aperture) if aperture == aperture else None
    aperture_suffix = f"{int(aperture)}kpc" if float(aperture).is_integer() else f"{aperture:g}kpc"
    mass_unit = sim.units["mass"]
    boxsize = _boxsize_raw(sim)
    hydrogen_fraction = float(getattr(sim.simulation, "XH", 0.76))
    processed_hosts = set()

    for galaxy in galaxies:
        host = getattr(galaxy, "halo", None)
        if host is None:
            continue
        host_key = int(getattr(host, "AHF_haloID", id(host)))
        if host_key in processed_hosts:
            continue
        processed_hosts.add(host_key)

        galaxy_ids = [
            int(gi)
            for gi in np.asarray(getattr(host, "galaxy_index_list", []), dtype=np.int64).tolist()
            if 0 <= int(gi) < len(getattr(sim, "galaxy_list", []))
        ]
        if not galaxy_ids:
            continue
        host_galaxies = [sim.galaxy_list[gi] for gi in galaxy_ids]

        gas_local = _component_local_indices(host, "gas")
        gas_global = _component_global_indices(sim, host, "gas")
        if gas_local.size == 0 or gas_global.size == 0:
            zero_mass = _quantity(sim, 0.0, mass_unit)
            for gal in host_galaxies:
                gal.masses["HI"] = zero_mass
                gal.masses["H2"] = zero_mass
                if aperture2 is not None:
                    gal.masses[f"HI_{aperture_suffix}"] = zero_mass
                    gal.masses[f"H2_{aperture_suffix}"] = zero_mass
            if hasattr(host, "masses"):
                host.masses["HI"] = zero_mass
                host.masses["H2"] = zero_mass
            continue

        hi_frac, h2_frac = _sanitized_hydrogen_fractions(sim, gas_local)
        gas_mass = _raw_array(dm.mass[gas_global], dtype=np.float64) * hydrogen_fraction
        hi_particle_mass = gas_mass * hi_frac
        h2_particle_mass = gas_mass * h2_frac

        galaxy_pos = np.asarray([_raw_array(gal.pos, dtype=np.float64) for gal in host_galaxies], dtype=np.float64)
        galaxy_mass = np.asarray(
            [_raw_scalar(getattr(gal, "masses", {}).get("total", 0.0)) for gal in host_galaxies],
            dtype=np.float64,
        )
        galaxy_hi_mass = np.zeros(len(host_galaxies), dtype=np.float64)
        galaxy_h2_mass = np.zeros(len(host_galaxies), dtype=np.float64)
        galaxy_hi_aperture = np.zeros(len(host_galaxies), dtype=np.float64)
        galaxy_h2_aperture = np.zeros(len(host_galaxies), dtype=np.float64)

        gas_pos = _raw_array(dm.pos[gas_global], dtype=np.float64)
        for idx in range(len(gas_global)):
            hi_mass = float(hi_particle_mass[idx])
            h2_mass = float(h2_particle_mass[idx])
            if hi_mass == 0.0 and h2_mass == 0.0:
                continue

            delta = _periodic_delta(galaxy_pos, gas_pos[idx], boxsize)
            d2 = np.sum(delta * delta, axis=1)
            zero_sep = d2 <= 0.0
            if np.any(zero_sep):
                candidate = np.where(zero_sep)[0]
                assign_idx = int(candidate[np.argmax(galaxy_mass[candidate])])
            else:
                assign_idx = int(np.argmax(galaxy_mass / d2))
            galaxy_hi_mass[assign_idx] += hi_mass
            galaxy_h2_mass[assign_idx] += h2_mass

            if aperture2 is not None:
                inside = d2 < aperture2
                if np.any(inside):
                    galaxy_hi_aperture[inside] += hi_mass
                    galaxy_h2_aperture[inside] += h2_mass

        for idx, gal in enumerate(host_galaxies):
            gal.masses["HI"] = _quantity(sim, galaxy_hi_mass[idx], mass_unit)
            gal.masses["H2"] = _quantity(sim, galaxy_h2_mass[idx], mass_unit)
            if aperture2 is not None:
                gal.masses[f"HI_{aperture_suffix}"] = _quantity(sim, galaxy_hi_aperture[idx], mass_unit)
                gal.masses[f"H2_{aperture_suffix}"] = _quantity(sim, galaxy_h2_aperture[idx], mass_unit)

        if hasattr(host, "masses"):
            host.masses["HI"] = _quantity(sim, float(np.sum(hi_particle_mass)), mass_unit)
            host.masses["H2"] = _quantity(sim, float(np.sum(h2_particle_mass)), mass_unit)


def _build_group_property_context(sim, group, group_type: str):
    dm = sim.data_manager
    global_idx = np.asarray(getattr(group, "global_indexes", []), dtype=np.int64)
    pos_all = _raw_array(dm.pos[global_idx], dtype=np.float64) if global_idx.size > 0 else np.empty((0, 3), dtype=np.float64)
    vel_all = _raw_array(dm.vel[global_idx], dtype=np.float64) if global_idx.size > 0 else np.empty((0, 3), dtype=np.float64)
    mass_all = _raw_array(dm.mass[global_idx], dtype=np.float64) if global_idx.size > 0 else np.empty(0, dtype=np.float64)

    gas_local = _component_local_indices(group, "gas")
    gas_global = _component_global_indices(sim, group, "gas")
    star_local = _component_local_indices(group, "star")
    star_global = _component_global_indices(sim, group, "star")
    dm_local = _component_local_indices(group, "dm")
    dm_global = _component_global_indices(sim, group, "dm")
    bh_local = _component_local_indices(group, "bh")
    bh_global = _component_global_indices(sim, group, "bh")
    dust_local = _component_local_indices(group, "dust")
    dust_global = _component_global_indices(sim, group, "dust")

    gas_mass = float(np.sum(_raw_array(dm.mass[gas_global], dtype=np.float64))) if gas_global.size > 0 else 0.0
    stellar_mass = float(np.sum(_raw_array(dm.mass[star_global], dtype=np.float64))) if star_global.size > 0 else 0.0
    dm_mass = float(np.sum(_raw_array(dm.mass[dm_global], dtype=np.float64))) if dm_global.size > 0 else 0.0
    baryon_mass = gas_mass + stellar_mass
    bh_mass = 0.0
    if bh_local.size > 0:
        if hasattr(dm, "bhmass"):
            bh_mass = float(np.max(_raw_array(dm.bhmass[bh_local], dtype=np.float64)))
        else:
            bh_mass = float(np.max(_raw_array(dm.mass[bh_global], dtype=np.float64)))
    if dust_local.size > 0:
        dust_mass = float(np.sum(_raw_array(dm.mass[dust_global], dtype=np.float64)))
    elif gas_local.size > 0 and hasattr(dm, "dustmass"):
        dust_mass = float(np.sum(_raw_array(dm.dustmass[gas_local], dtype=np.float64)))
    else:
        dust_mass = 0.0

    if global_idx.size > 0 and mass_all.size > 0 and np.sum(mass_all) > 0.0:
        pos_center = _weighted_periodic_center(pos_all, mass_all, _boxsize_raw(sim))
        vel_center = _weighted_velocity_center(vel_all, mass_all)
        if hasattr(dm, "pot") and getattr(sim, "load_pot", True):
            pot = _raw_array(dm.pot[global_idx], dtype=np.float64)
            minpot_i = int(np.argmin(pot))
            minpot_pos = pos_all[minpot_i]
            minpot_vel = vel_all[minpot_i]
        else:
            minpot_pos = pos_center
            minpot_vel = vel_center
    else:
        pos_center = np.zeros(3, dtype=np.float64)
        vel_center = np.zeros(3, dtype=np.float64)
        minpot_pos = np.zeros(3, dtype=np.float64)
        minpot_vel = np.zeros(3, dtype=np.float64)

    rel_pos_center = _periodic_delta(pos_all, pos_center, _boxsize_raw(sim))
    rel_pos_minpot = _periodic_delta(pos_all, minpot_pos, _boxsize_raw(sim))
    radii_center = np.linalg.norm(rel_pos_center, axis=1) if rel_pos_center.size > 0 else np.empty(0, dtype=np.float64)
    radii_minpot = np.linalg.norm(rel_pos_minpot, axis=1) if rel_pos_minpot.size > 0 else np.empty(0, dtype=np.float64)

    return SimpleNamespace(
        sim=sim,
        group=group,
        group_type=str(group_type),
        dm=dm,
        boxsize=_boxsize_raw(sim),
        mass_unit=sim.units["mass"],
        vel_unit=sim.units["velocity"],
        len_unit=sim.units["length"],
        global_idx=global_idx,
        pos_all=pos_all,
        vel_all=vel_all,
        mass_all=mass_all,
        gas_local=gas_local,
        gas_global=gas_global,
        star_local=star_local,
        star_global=star_global,
        dm_local=dm_local,
        dm_global=dm_global,
        bh_local=bh_local,
        bh_global=bh_global,
        dust_local=dust_local,
        dust_global=dust_global,
        total_mass=float(np.sum(mass_all)) if mass_all.size > 0 else 0.0,
        gas_mass=gas_mass,
        stellar_mass=stellar_mass,
        dm_mass=dm_mass,
        baryon_mass=baryon_mass,
        bh_mass=bh_mass,
        dust_mass=dust_mass,
        pos_center=pos_center,
        vel_center=vel_center,
        minpot_pos=minpot_pos,
        minpot_vel=minpot_vel,
        rel_pos_center=rel_pos_center,
        rel_pos_minpot=rel_pos_minpot,
        radii_center=radii_center,
        radii_minpot=radii_minpot,
    )


def _apply_common_property_pass(context, state: Dict[str, object], masses: Dict[str, object], radii: Dict[str, object], velocity_dispersions: Dict[str, object]) -> None:
    sim = context.sim
    dm = context.dm

    state["ngas"] = int(len(context.gas_local))
    state["nstar"] = int(len(context.star_local))
    state["ndm"] = int(len(context.dm_local))
    state["nbh"] = int(len(context.bh_local))
    state["ndust"] = int(len(context.dust_local))

    masses["total"] = _quantity(sim, context.total_mass, context.mass_unit)
    masses["gas"] = _quantity(sim, context.gas_mass, context.mass_unit)
    masses["stellar"] = _quantity(sim, context.stellar_mass, context.mass_unit)
    masses["dm"] = _quantity(sim, context.dm_mass, context.mass_unit)
    masses["baryon"] = _quantity(sim, context.baryon_mass, context.mass_unit)
    masses["H"] = _quantity(sim, context.gas_mass * float(getattr(sim.simulation, "XH", 0.76)), context.mass_unit)
    masses["dust"] = _quantity(sim, context.dust_mass, context.mass_unit)
    if context.bh_local.size > 0 or getattr(dm, "blackholes", False):
        masses["bh"] = _quantity(sim, context.bh_mass, context.mass_unit)

    state["gas_fraction"] = float(context.gas_mass / context.baryon_mass) if context.baryon_mass > 0.0 else 0.0
    state["pos"] = _array_from_key(sim, context.pos_center, "length")
    state["vel"] = _array_from_key(sim, context.vel_center, "velocity")
    state["minpotpos"] = _array_from_key(sim, context.minpot_pos, "length")
    state["minpotvel"] = _array_from_key(sim, context.minpot_vel, "velocity")

    component_specs = {
        "total": (context.mass_all, context.pos_all, context.vel_all),
        "baryon": (
            np.concatenate([
                _raw_array(dm.mass[context.gas_global], dtype=np.float64) if context.gas_global.size > 0 else np.empty(0, dtype=np.float64),
                _raw_array(dm.mass[context.star_global], dtype=np.float64) if context.star_global.size > 0 else np.empty(0, dtype=np.float64),
            ]),
            np.concatenate([
                _raw_array(dm.pos[context.gas_global], dtype=np.float64) if context.gas_global.size > 0 else np.empty((0, 3), dtype=np.float64),
                _raw_array(dm.pos[context.star_global], dtype=np.float64) if context.star_global.size > 0 else np.empty((0, 3), dtype=np.float64),
            ], axis=0) if context.gas_global.size or context.star_global.size else np.empty((0, 3), dtype=np.float64),
            np.concatenate([
                _raw_array(dm.vel[context.gas_global], dtype=np.float64) if context.gas_global.size > 0 else np.empty((0, 3), dtype=np.float64),
                _raw_array(dm.vel[context.star_global], dtype=np.float64) if context.star_global.size > 0 else np.empty((0, 3), dtype=np.float64),
            ], axis=0) if context.gas_global.size or context.star_global.size else np.empty((0, 3), dtype=np.float64),
        ),
        "gas": (
            _raw_array(dm.mass[context.gas_global], dtype=np.float64) if context.gas_global.size > 0 else np.empty(0, dtype=np.float64),
            _raw_array(dm.pos[context.gas_global], dtype=np.float64) if context.gas_global.size > 0 else np.empty((0, 3), dtype=np.float64),
            _raw_array(dm.vel[context.gas_global], dtype=np.float64) if context.gas_global.size > 0 else np.empty((0, 3), dtype=np.float64),
        ),
        "stellar": (
            _raw_array(dm.mass[context.star_global], dtype=np.float64) if context.star_global.size > 0 else np.empty(0, dtype=np.float64),
            _raw_array(dm.pos[context.star_global], dtype=np.float64) if context.star_global.size > 0 else np.empty((0, 3), dtype=np.float64),
            _raw_array(dm.vel[context.star_global], dtype=np.float64) if context.star_global.size > 0 else np.empty((0, 3), dtype=np.float64),
        ),
        "dm": (
            _raw_array(dm.mass[context.dm_global], dtype=np.float64) if context.dm_global.size > 0 else np.empty(0, dtype=np.float64),
            _raw_array(dm.pos[context.dm_global], dtype=np.float64) if context.dm_global.size > 0 else np.empty((0, 3), dtype=np.float64),
            _raw_array(dm.vel[context.dm_global], dtype=np.float64) if context.dm_global.size > 0 else np.empty((0, 3), dtype=np.float64),
        ),
    }

    for name, (comp_mass, comp_pos, comp_vel) in component_specs.items():
        if comp_mass.size == 0 or np.sum(comp_mass) <= 0.0:
            radii[name] = _quantity(sim, 0.0, context.len_unit)
            radii[f"{name}_r20"] = _quantity(sim, 0.0, context.len_unit)
            radii[f"{name}_half_mass"] = _quantity(sim, 0.0, context.len_unit)
            radii[f"{name}_r80"] = _quantity(sim, 0.0, context.len_unit)
            velocity_dispersions[name] = _quantity(sim, 0.0, context.vel_unit)
            continue
        comp_r = np.linalg.norm(_periodic_delta(comp_pos, context.pos_center, context.boxsize), axis=1)
        full_r, r20, half_r, r80 = _radius_summary(comp_r, comp_mass)
        radii[name] = _quantity(sim, full_r, context.len_unit)
        radii[f"{name}_r20"] = _quantity(sim, r20, context.len_unit)
        radii[f"{name}_half_mass"] = _quantity(sim, half_r, context.len_unit)
        radii[f"{name}_r80"] = _quantity(sim, r80, context.len_unit)
        velocity_dispersions[name] = _quantity(sim, _velocity_dispersion(comp_vel, comp_mass), context.vel_unit)

    velocity_dispersions["all"] = velocity_dispersions["total"]


def _apply_gas_property_pass(context, state: Dict[str, object], masses: Dict[str, object], metallicities: Dict[str, object], temperatures: Dict[str, object]) -> None:
    sim = context.sim
    dm = context.dm

    gas_sfr = _raw_array(dm.gsfr[context.gas_local], dtype=np.float64) if context.gas_local.size > 0 and hasattr(dm, "gsfr") else np.empty(0, dtype=np.float64)
    gas_Z = _raw_array(dm.gZ[context.gas_local], dtype=np.float64) if context.gas_local.size > 0 and hasattr(dm, "gZ") else np.empty(0, dtype=np.float64)
    gas_T = _raw_array(dm.gT[context.gas_local], dtype=np.float64) if context.gas_local.size > 0 and hasattr(dm, "gT") else np.empty(0, dtype=np.float64)
    gas_mass_arr = _raw_array(dm.mass[context.gas_global], dtype=np.float64) if context.gas_global.size > 0 else np.empty(0, dtype=np.float64)
    gas_mass_sum = float(np.sum(gas_mass_arr)) if gas_mass_arr.size > 0 else 0.0
    gas_sfr_sum = float(np.sum(gas_sfr)) if gas_sfr.size > 0 else 0.0
    state["sfr"] = _quantity(sim, gas_sfr_sum, f"{sim.units['mass']}/{sim.units['time']}")
    if gas_mass_sum > 0.0 and gas_Z.size > 0:
        metallicities["mass_weighted"] = _quantity(sim, float(np.sum(gas_Z * gas_mass_arr) / gas_mass_sum), "")
        temperatures["mass_weighted"] = _quantity_from_key(sim, float(np.sum(gas_T * gas_mass_arr) / gas_mass_sum), "temperature")
    else:
        metallicities["mass_weighted"] = _quantity(sim, 0.0, "")
        temperatures["mass_weighted"] = _quantity_from_key(sim, 0.0, "temperature")
    if gas_sfr_sum > 0.0 and gas_Z.size > 0:
        metallicities["sfr_weighted"] = _quantity(sim, float(np.sum(gas_Z * gas_sfr) / gas_sfr_sum), "")
        temperatures["sfr_weighted"] = _quantity_from_key(sim, float(np.sum(gas_T * gas_sfr) / gas_sfr_sum), "temperature")
    else:
        metallicities["sfr_weighted"] = _quantity(sim, 0.0, "")
        temperatures["sfr_weighted"] = _quantity_from_key(sim, 0.0, "temperature")

    hi_mass = 0.0
    h2_mass = 0.0
    if context.gas_local.size > 0 and hasattr(dm, "gfHI") and hasattr(dm, "gfH2"):
        hi_frac, h2_frac = _sanitized_hydrogen_fractions(sim, context.gas_local)
        hi_mass = float(np.sum(gas_mass_arr * hi_frac) * float(getattr(sim.simulation, "XH", 0.76)))
        h2_mass = float(np.sum(gas_mass_arr * h2_frac) * float(getattr(sim.simulation, "XH", 0.76)))
    masses["HI"] = _quantity(sim, hi_mass, context.mass_unit)
    masses["H2"] = _quantity(sim, h2_mass, context.mass_unit)


def _apply_star_property_pass(context, state: Dict[str, object], metallicities: Dict[str, object]) -> None:
    sim = context.sim
    dm = context.dm

    star_mass_arr = _raw_array(dm.mass[context.star_global], dtype=np.float64) if context.star_global.size > 0 else np.empty(0, dtype=np.float64)
    if context.star_local.size > 0 and hasattr(dm, "sZ"):
        star_Z = _raw_array(dm.sZ[context.star_local], dtype=np.float64)
        zsum = float(np.sum(star_Z * star_mass_arr))
        metallicities["stellar"] = _quantity(sim, zsum / max(float(np.sum(star_mass_arr)), 1.0e-30), "")
    else:
        metallicities["stellar"] = _quantity(sim, 0.0, "")

    if context.star_local.size > 0 and hasattr(dm, "age"):
        star_age = _raw_array(dm.age[context.star_local], dtype=np.float64)
        if star_mass_arr.size > 0 and np.sum(star_mass_arr) > 0.0:
            age_mass = float(np.sum(star_age * star_mass_arr) / np.sum(star_mass_arr))
            if "stellar" in metallicities and np.sum(star_mass_arr * np.maximum(_raw_array(dm.sZ[context.star_local], dtype=np.float64), 0.0)) > 0.0 and hasattr(dm, "sZ"):
                star_Z = _raw_array(dm.sZ[context.star_local], dtype=np.float64)
                zweight = np.sum(star_age * star_mass_arr * star_Z)
                ztot = np.sum(star_mass_arr * star_Z)
                age_metal = float(zweight / ztot) if ztot > 0.0 else age_mass
            else:
                age_metal = age_mass
            state["ages"] = {
                "mass_weighted": _quantity(sim, age_mass, "Gyr"),
                "metal_weighted": _quantity(sim, age_metal, "Gyr"),
            }
            state["sfr_100"] = _quantity(sim, float(np.sum(star_mass_arr[star_age < 0.1]) / 1.0e8), "Msun/yr")


def _apply_bh_property_pass(context, state: Dict[str, object]) -> None:
    from astropy import constants as const

    sim = context.sim
    dm = context.dm
    if context.bh_local.size == 0:
        return
    bhmdot_arr = _raw_array(dm.bhmdot[context.bh_local], dtype=np.float64) if hasattr(dm, "bhmdot") else np.empty(0, dtype=np.float64)
    bhmass_arr = _raw_array(dm.bhmass[context.bh_local], dtype=np.float64) if hasattr(dm, "bhmass") else _raw_array(dm.mass[context.bh_global], dtype=np.float64)
    if bhmass_arr.size > 0:
        imax = int(np.argmax(bhmass_arr))
        bhmdot = float(bhmdot_arr[imax]) if bhmdot_arr.size > imax else 0.0
        state["bhmdot"] = _quantity(sim, bhmdot, "Msun/yr")
        FRAD = 0.1
        edd_factor = (4 * np.pi * const.G * const.m_p / (FRAD * const.c * const.sigma_T)).to("1/yr").value
        state["bh_fedd"] = _quantity(sim, bhmdot / (edd_factor * max(float(bhmass_arr[imax]), 1.0e-30)), "")
    else:
        state["bhmdot"] = _quantity(sim, 0.0, "Msun/yr")
        state["bh_fedd"] = _quantity(sim, 0.0, "")


def _apply_rotation_and_virial_pass(context, state: Dict[str, object], masses: Dict[str, object], radii: Dict[str, object], velocity_dispersions: Dict[str, object], temperatures: Dict[str, object], rotation: Dict[str, object], virial_quantities: Dict[str, object]) -> None:
    sim = context.sim
    if context.rel_pos_center.size == 0 or context.mass_all.size == 0:
        radii["r200"] = _quantity(sim, 0.0, context.len_unit)
        temperatures["virial"] = _quantity(sim, 0.0, "K").to(sim.units["temperature"])
        virial_quantities["r200"] = radii["r200"]
        virial_quantities["r200c"] = radii.get("r200c", _quantity(sim, 0.0, context.len_unit))
        virial_quantities["r500c"] = radii.get("r500c", _quantity(sim, 0.0, context.len_unit))
        virial_quantities["r2500c"] = radii.get("r2500c", _quantity(sim, 0.0, context.len_unit))
        virial_quantities["circular_velocity"] = _quantity(sim, 0.0, "km/s").to(sim.units["velocity"])
        virial_quantities["temperature"] = temperatures["virial"]
        virial_quantities["spin_param"] = _quantity(sim, 0.0, "")
        return

    length_to_kpc = _unit_factor(sim, context.len_unit, "kpc")
    mass_to_msun = _unit_factor(sim, context.mass_unit, "Msun")
    pos_rel_kpc = context.rel_pos_center * length_to_kpc
    radii_minpot_kpc = context.radii_minpot * length_to_kpc
    vel_rel = context.vel_all - context.vel_center
    vel_kms = vel_rel * _unit_factor(sim, context.vel_unit, "km/s")
    mass_msun = context.mass_all * mass_to_msun

    if pos_rel_kpc.size > 0 and mass_msun.size > 0:
        momentum = mass_msun[:, None] * vel_kms
        Lvec = np.sum(np.cross(pos_rel_kpc, momentum), axis=0)
        Lnorm = float(np.linalg.norm(Lvec))
        state["angular_momentum_vector"] = _quantity(sim, Lvec, "Msun*km**2/s")
        if Lnorm > 0.0:
            ALPHA = float(np.arctan2(Lvec[1], Lvec[2]))
            p = np.asarray([
                np.sin(np.arccos(Lvec[2] / Lnorm)) * np.cos(np.arctan2(Lvec[1], Lvec[0])),
                np.sin(np.arccos(Lvec[2] / Lnorm)) * np.sin(np.arctan2(Lvec[1], Lvec[0])),
                np.cos(np.arccos(Lvec[2] / Lnorm)),
            ], dtype=np.float64)
            p = np.asarray([p[0], p[1] * np.cos(ALPHA) - p[2] * np.sin(ALPHA), p[1] * np.sin(ALPHA) + p[2] * np.cos(ALPHA)], dtype=np.float64)
            BETA = float(np.arctan2(p[0], p[2]))
            state["rotation_angles"] = {"ALPHA": ALPHA, "BETA": BETA}
            rxy = np.sqrt(pos_rel_kpc[:, 0] ** 2 + pos_rel_kpc[:, 1] ** 2)
            valid = rxy > 0.0
            if np.any(valid):
                vphi = (vel_kms[valid, 0] * -pos_rel_kpc[valid, 1] + vel_kms[valid, 1] * pos_rel_kpc[valid, 0]) / rxy[valid]
                vr = (vel_kms[valid, 0] * pos_rel_kpc[valid, 0] + vel_kms[valid, 1] * pos_rel_kpc[valid, 1]) / rxy[valid]
                state["max_vphi"] = _quantity_from_key(sim, float(np.max(vphi)), "velocity")
                state["max_vr"] = _quantity_from_key(sim, float(np.max(vr)), "velocity")
                l_part = np.cross(pos_rel_kpc, momentum)
                ldot = np.einsum("ij,j->i", l_part, Lvec)
                state["BoverT"] = _quantity(sim, float(2.0 * np.sum(mass_msun[ldot < 0.0]) / max(context.total_mass * mass_to_msun, 1.0e-30)), "")
                krot = 0.5 * (ldot[valid] / np.maximum(rxy[valid], 1.0e-30)) ** 2 / np.maximum(mass_msun[valid], 1.0e-30)
                ktot = 0.5 * np.sum(vel_kms[valid] ** 2, axis=1) * mass_msun[valid]
                rotation["kappa_rot_total"] = _quantity(sim, float(np.sum(krot) / max(np.sum(ktot), 1.0e-30)), "")

    if radii_minpot_kpc.size > 0 and context.total_mass > 0.0:
        order = np.argsort(radii_minpot_kpc)
        r_sorted = radii_minpot_kpc[order]
        m_sorted = np.cumsum(mass_msun[order])
        valid = r_sorted > 0.0
        if np.any(valid):
            rhocrit = float(getattr(sim.simulation.critical_density.to("Msun/kpc**3"), "value", sim.simulation.critical_density.to("Msun/kpc**3").d))
            overdensity = np.full_like(r_sorted, np.nan, dtype=np.float64)
            if rhocrit > 0.0:
                overdensity[valid] = m_sorted[valid] / (
                    (4.0 / 3.0) * np.pi * r_sorted[valid] ** 3 * rhocrit
                )
            for factor in (200, 500, 2500):
                mask = overdensity >= float(factor)
                if np.any(mask):
                    ridx = np.where(mask)[0][-1]
                    rkpc = float(r_sorted[ridx])
                    mmsun = float(m_sorted[ridx])
                else:
                    rkpc = 0.0
                    mmsun = 0.0
                radii[f"r{factor}c"] = _quantity(sim, sim.yt_dataset.quan(rkpc, "kpc").to(context.len_unit).value, context.len_unit)
                masses[f"m{factor}c"] = _quantity(sim, sim.yt_dataset.quan(mmsun, "Msun").to(context.mass_unit).value, context.mass_unit)

    try:
        total_mass_q = _quantity(sim, context.total_mass, context.mass_unit).to("Msun")
        Om_z = float(getattr(sim.simulation, "Om_z", 0.0))
        if Om_z > 0.0:
            r200 = (sim.simulation.G.to("kpc**3/(Msun*s**2)") * total_mass_q / (100.0 * Om_z * sim.simulation.H_z.to("1/s") ** 2)) ** (1.0 / 3.0)
            vc = np.sqrt(sim.simulation.G.to("kpc**3/(Msun*s**2)") * total_mass_q / r200).to("km/s")
            vT = _quantity(sim, 3.6e5 * (float(vc.value) / 100.0) ** 2, "K")
        else:
            r200 = _quantity(sim, 0.0, "kpc")
            vc = _quantity(sim, 0.0, "km/s")
            vT = _quantity(sim, 0.0, "K")
    except Exception:
        r200 = _quantity(sim, 0.0, "kpc")
        vc = _quantity(sim, 0.0, "km/s")
        vT = _quantity(sim, 0.0, "K")
        total_mass_q = _quantity(sim, 0.0, "Msun")
    radii["r200"] = r200.to(context.len_unit)
    temperatures["virial"] = vT.to(sim.units["temperature"])
    virial_quantities["r200"] = r200.to(context.len_unit)
    virial_quantities["r200c"] = radii.get("r200c", _quantity(sim, 0.0, context.len_unit))
    virial_quantities["r500c"] = radii.get("r500c", _quantity(sim, 0.0, context.len_unit))
    virial_quantities["r2500c"] = radii.get("r2500c", _quantity(sim, 0.0, context.len_unit))
    virial_quantities["circular_velocity"] = vc.to(sim.units["velocity"])
    virial_quantities["temperature"] = temperatures["virial"]
    if "angular_momentum_vector" in state and float(getattr(r200.to("km"), "value", r200.to("km").d)) > 0.0 and context.total_mass > 0.0 and float(getattr(vc.to("km/s"), "value", vc.to("km/s").d)) > 0.0:
        Lmag = np.linalg.norm(_raw_array(state["angular_momentum_vector"].to("Msun*km**2/s")))
        spin = Lmag / (np.sqrt(2.0) * float(total_mass_q.value) * float(vc.to("km/s").value) * float(r200.to("km").value))
    else:
        spin = 0.0
    virial_quantities["spin_param"] = _quantity(sim, spin, "")


def _apply_galaxy_aperture_pass(context, masses: Dict[str, object], radii: Dict[str, object], velocity_dispersions: Dict[str, object]) -> None:
    sim = context.sim
    if context.group_type != "galaxy":
        return
    aperture = float(sim._kwargs.get("aperture", 30.0))
    masses_30, sigmas_30 = _group_aperture_properties(sim, context.group, aperture, gal_pos_raw=context.pos_center)
    masses.update(masses_30)
    velocity_dispersions.update(sigmas_30)
    if "half_stellar_radius_property" in sim._kwargs:
        stellar_half = _raw_scalar(radii.get("stellar_half_mass", _quantity(sim, 0.0, context.len_unit)))
        masses_hmr, sigmas_hmr = _group_aperture_properties(sim, context.group, stellar_half, gal_pos_raw=context.pos_center)
        renamed_masses = {}
        renamed_sigmas = {}
        for key, value in masses_hmr.items():
            if key.endswith("kpc"):
                renamed_masses[key[:-3] + "stellar_half_mass_radius"] = value
            else:
                renamed_masses[key] = value
        for key, value in sigmas_hmr.items():
            if key.endswith("kpc"):
                renamed_sigmas[key[:-3] + "stellar_half_mass_radius"] = value
            else:
                renamed_sigmas[key] = value
        masses.update(renamed_masses)
        velocity_dispersions.update(renamed_sigmas)


def _compute_single_group_properties_selfcontained(sim, group, group_type: str) -> Dict[str, object]:
    context = _build_group_property_context(sim, group, group_type)
    state: Dict[str, object] = {}
    masses: Dict[str, object] = {}
    radii: Dict[str, object] = {}
    velocity_dispersions: Dict[str, object] = {}
    metallicities: Dict[str, object] = {}
    temperatures: Dict[str, object] = {}
    rotation: Dict[str, object] = {}
    virial_quantities: Dict[str, object] = {}

    _apply_common_property_pass(context, state, masses, radii, velocity_dispersions)
    _apply_gas_property_pass(context, state, masses, metallicities, temperatures)
    _apply_star_property_pass(context, state, metallicities)
    _apply_bh_property_pass(context, state)
    _apply_rotation_and_virial_pass(context, state, masses, radii, velocity_dispersions, temperatures, rotation, virial_quantities)
    _apply_galaxy_aperture_pass(context, masses, radii, velocity_dispersions)

    state["masses"] = masses
    state["radii"] = radii
    state["velocity_dispersions"] = velocity_dispersions
    state["metallicities"] = metallicities
    state["temperatures"] = temperatures
    state["rotation"] = rotation
    state["virial_quantities"] = virial_quantities
    return state


def _compute_group_properties_subset(sim, *, group_type: str, groups: Sequence) -> None:
    if not groups:
        return
    max_workers = max(1, min(int(getattr(sim, "nproc", 1)), len(groups)))
    if max_workers > 1 and len(groups) > 8:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_compute_single_group_properties_selfcontained, sim, group, group_type): group
                for group in groups
            }
            for fut in as_completed(futures):
                state = fut.result()
                group = futures[fut]
                for key, value in state.items():
                    setattr(group, key, value)
    else:
        for group in groups:
            state = _compute_single_group_properties_selfcontained(sim, group, group_type)
            for key, value in state.items():
                setattr(group, key, value)

    if str(group_type) == "galaxy":
        _apply_host_local_hydrogen_assignment(sim, groups)


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


def _build_task_input_payload(
    sim,
    *,
    task: AHFSubhaloTask,
    membership_arrays: Dict[int, np.ndarray],
    pid_maps_sel,
    fof_nHlim: float,
    fof_Tlim: float,
    fof_use_sfr_gate: bool,
) -> Optional[Dict[str, object]]:
    gas_sel_dense, star_sel, bh_sel, dm_sel = _resolve_node_membership(
        node_id=int(task.node_id),
        membership_arrays=membership_arrays,
        pid_maps_sel=pid_maps_sel,
        sim=sim,
        fof_nHlim=float(fof_nHlim),
        fof_Tlim=float(fof_Tlim),
        fof_use_sfr_gate=bool(fof_use_sfr_gate),
    )

    gas_concat = sim.data_manager.selected_to_concat("gas", np.asarray(gas_sel_dense, dtype=np.int64))
    star_concat = sim.data_manager.selected_to_concat("star", np.asarray(star_sel, dtype=np.int64))
    if bh_sel.size > 0:
        bh_concat = sim.data_manager.selected_to_concat("bh", np.asarray(bh_sel, dtype=np.int64))
    else:
        bh_concat = np.empty(0, dtype=np.int64)

    eligible_concat = np.concatenate((gas_concat, star_concat, bh_concat), axis=None).astype(np.int64, copy=False)
    if eligible_concat.size == 0:
        return None

    return {
        "task": _serialize_task(task),
        "gas_sel": np.asarray(gas_sel_dense, dtype=np.int32),
        "star_sel": np.asarray(star_sel, dtype=np.int32),
        "bh_sel": np.asarray(bh_sel, dtype=np.int32),
        "dm_sel": np.asarray(dm_sel, dtype=np.int32),
        "ng": int(gas_sel_dense.size),
        "ns": int(star_sel.size),
        "nb": int(bh_sel.size),
        "eligible_pos": np.asarray(sim.data_manager.pos[eligible_concat]),
        "eligible_vel": np.asarray(sim.data_manager.vel[eligible_concat]),
    }


def _task_pool_from_payload(
    task_payload: Dict[str, object],
    *,
    blocked_star_idx: Optional[np.ndarray] = None,
) -> Optional[Dict[str, object]]:
    task = _deserialize_task(task_payload["task"])
    gas_sel = np.asarray(task_payload["gas_sel"], dtype=np.int32)
    star_sel = np.asarray(task_payload["star_sel"], dtype=np.int32)
    bh_sel = np.asarray(task_payload["bh_sel"], dtype=np.int32)
    dm_sel = np.asarray(task_payload["dm_sel"], dtype=np.int32)
    pos = np.asarray(task_payload["eligible_pos"])
    vel = np.asarray(task_payload["eligible_vel"])
    ng = int(task_payload["ng"])
    ns = int(task_payload["ns"])
    nb = int(task_payload["nb"])

    gas_pos = pos[:ng]
    gas_vel = vel[:ng]
    star_pos = pos[ng:ng + ns]
    star_vel = vel[ng:ng + ns]
    bh_pos = pos[ng + ns:ng + ns + nb]
    bh_vel = vel[ng + ns:ng + ns + nb]

    if blocked_star_idx is not None and np.asarray(blocked_star_idx).size > 0 and star_sel.size > 0:
        keep = ~np.isin(star_sel.astype(np.int64, copy=False), np.asarray(blocked_star_idx, dtype=np.int64))
        star_sel = star_sel[keep]
        star_pos = star_pos[keep]
        star_vel = star_vel[keep]

    out_pos_blocks = []
    out_vel_blocks = []
    if gas_pos.size > 0:
        out_pos_blocks.append(gas_pos)
        out_vel_blocks.append(gas_vel)
    if star_pos.size > 0:
        out_pos_blocks.append(star_pos)
        out_vel_blocks.append(star_vel)
    if bh_pos.size > 0:
        out_pos_blocks.append(bh_pos)
        out_vel_blocks.append(bh_vel)
    if not out_pos_blocks:
        return None

    return {
        "task": task,
        "gas_sel": gas_sel,
        "star_sel": star_sel,
        "bh_sel": bh_sel,
        "dm_sel": dm_sel,
        "ng": int(gas_sel.size),
        "ns": int(star_sel.size),
        "nb": int(bh_sel.size),
        "eligible_pos": np.concatenate(out_pos_blocks, axis=0),
        "eligible_vel": np.concatenate(out_vel_blocks, axis=0),
    }


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
        cc_backend=_preferred_cc_backend(cc_backend, backend=backend),
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


def _fof_on_task_payload(
    *,
    task_payload: Dict[str, object],
    min_stars: int,
    fof_ll: float,
    fof_vel_ll: Optional[float],
    backend: str,
    cc_backend: str,
    max_pairs_per_batch: int,
    blocked_star_idx: Optional[np.ndarray] = None,
    device_id: Optional[int] = None,
) -> List[Dict[str, np.ndarray | int]]:
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
            raise RuntimeError("GPU payload task requested but CuPy/CUDA is unavailable")
        cp.cuda.Device(int(device_id)).use()

    pool = _task_pool_from_payload(task_payload, blocked_star_idx=blocked_star_idx)
    if pool is None:
        return []

    task = pool["task"]
    gas_sel = np.asarray(pool["gas_sel"], dtype=np.int32)
    star_sel = np.asarray(pool["star_sel"], dtype=np.int32)
    bh_sel = np.asarray(pool["bh_sel"], dtype=np.int32)
    dm_sel = np.asarray(pool["dm_sel"], dtype=np.int32)
    eligible_pos = np.asarray(pool["eligible_pos"])
    eligible_vel = np.asarray(pool["eligible_vel"])
    ng = int(pool["ng"])
    ns = int(pool["ns"])
    nb = int(pool["nb"])

    if int(star_sel.size) < int(min_stars) or eligible_pos.shape[0] < int(min_stars):
        return []

    tags, _ = fof6d_on_pool_graph(
        eligible_pos,
        eligible_vel,
        fof_ll=float(fof_ll),
        vel_ll=fof_vel_ll,
        mingrp=int(min_stars),
        periodic=False,
        kernel="caesar_table",
        backend=resolved_backend,
        cc_backend=_preferred_cc_backend(cc_backend, backend=resolved_backend),
        max_pairs_per_batch=int(max_pairs_per_batch),
    )

    tags = np.asarray(tags, dtype=np.int64)
    valid_tags = np.unique(tags[tags >= 0])
    if valid_tags.size == 0:
        return []

    allow_dm = bool(valid_tags.size == 1)
    out: List[Dict[str, np.ndarray | int]] = []
    for gid in valid_tags:
        mask = tags == int(gid)
        gsub = gas_sel[mask[:ng]] if ng > 0 else np.empty(0, dtype=np.int32)
        ssub = star_sel[mask[ng:ng + ns]] if ns > 0 else np.empty(0, dtype=np.int32)
        bsub = bh_sel[mask[ng + ns:ng + ns + nb]] if nb > 0 else np.empty(0, dtype=np.int32)
        if int(ssub.size) < int(min_stars):
            continue
        out.append(
            _candidate_payload(
                task=task,
                star_sel=ssub,
                gas_sel=gsub,
                bh_sel=bsub,
                dm_sel=dm_sel if allow_dm else np.empty(0, dtype=np.int32),
            )
        )
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
        cc_backend=_preferred_cc_backend(cc_backend, backend=resolved_backend),
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


def _fof_on_batch_payload(
    *,
    batch_payload: Dict[str, object],
    min_stars: int,
    fof_ll: float,
    fof_vel_ll: Optional[float],
    backend: str,
    cc_backend: str,
    max_pairs_per_batch: int,
    device_id: Optional[int] = None,
) -> Dict[int, List[Dict[str, np.ndarray | int]]]:
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
            raise RuntimeError("GPU payload batch requested but CuPy/CUDA is unavailable")
        cp.cuda.Device(int(device_id)).use()

    task_payloads = list(batch_payload.get("task_payloads", []))
    out: Dict[int, List[Dict[str, np.ndarray | int]]] = {}
    entry_data = []
    concat_pos_blocks = []
    concat_vel_blocks = []
    owner_blocks = []

    for owner_idx, task_payload in enumerate(task_payloads):
        pool = _task_pool_from_payload(task_payload)
        task = pool["task"] if pool is not None else _deserialize_task(task_payload["task"])
        out[int(task.node_id)] = []
        if pool is None:
            continue
        star_sel = np.asarray(pool["star_sel"], dtype=np.int32)
        eligible_pos = np.asarray(pool["eligible_pos"])
        eligible_vel = np.asarray(pool["eligible_vel"])
        if int(star_sel.size) < int(min_stars) or eligible_pos.shape[0] < int(min_stars):
            continue

        entry = {
            "task": task,
            "gas_sel": np.asarray(pool["gas_sel"], dtype=np.int32),
            "star_sel": star_sel,
            "bh_sel": np.asarray(pool["bh_sel"], dtype=np.int32),
            "dm_sel": np.asarray(pool["dm_sel"], dtype=np.int32),
            "ng": int(pool["ng"]),
            "ns": int(pool["ns"]),
            "nb": int(pool["nb"]),
            "offset": int(sum(block.shape[0] for block in concat_pos_blocks)),
            "size": int(eligible_pos.shape[0]),
            "owner_idx": int(owner_idx),
        }
        entry_data.append(entry)
        concat_pos_blocks.append(eligible_pos)
        concat_vel_blocks.append(eligible_vel)
        owner_blocks.append(np.full(int(eligible_pos.shape[0]), int(owner_idx), dtype=np.int32))

    if not concat_pos_blocks:
        return out

    combined_pos = np.concatenate(concat_pos_blocks, axis=0)
    combined_vel = np.concatenate(concat_vel_blocks, axis=0)
    combined_owner = np.concatenate(owner_blocks, axis=0)

    tags, _ = fof6d_on_pool_graph(
        combined_pos,
        combined_vel,
        fof_ll=float(fof_ll),
        vel_ll=fof_vel_ll,
        mingrp=int(min_stars),
        periodic=False,
        kernel="caesar_table",
        backend=resolved_backend,
        cc_backend=_preferred_cc_backend(cc_backend, backend=resolved_backend),
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
            comp_task_ids = [int(_deserialize_task(task_payloads[int(i)]["task"]).node_id) for i in owners.tolist()]
            raise AssertionError(
                "AHF-subhalo invariant violated: graph component spans multiple batched halos "
                f"{comp_task_ids}"
            )
        owner_idx = int(owners[0])
        comps_by_owner.setdefault(owner_idx, []).append(int(comp))

    for entry in entry_data:
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
        gas_sel = np.asarray(entry["gas_sel"], dtype=np.int32)
        star_sel = np.asarray(entry["star_sel"], dtype=np.int32)
        bh_sel = np.asarray(entry["bh_sel"], dtype=np.int32)
        dm_sel = np.asarray(entry["dm_sel"], dtype=np.int32)
        allow_dm = len(comps) == 1

        local_tags = tags[offset:offset + size]
        for comp in comps:
            local_mask = local_tags == int(comp)
            gsub = gas_sel[local_mask[:ng]] if ng > 0 else np.empty(0, dtype=np.int32)
            ssub = star_sel[local_mask[ng:ng + ns]] if ns > 0 else np.empty(0, dtype=np.int32)
            bsub = bh_sel[local_mask[ng + ns:ng + ns + nb]] if nb > 0 else np.empty(0, dtype=np.int32)
            if int(ssub.size) < int(min_stars):
                continue
            out[int(task.node_id)].append(
                _candidate_payload(
                    task=task,
                    star_sel=ssub,
                    gas_sel=gsub,
                    bh_sel=bsub,
                    dm_sel=dm_sel if allow_dm else np.empty(0, dtype=np.int32),
                )
            )
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


def _reconcile_root_payload(
    *,
    tasks: Sequence[AHFSubhaloTask],
    initial_candidates_by_node: Dict[int, List[Dict[str, np.ndarray | int]]],
    task_payloads_by_node: Dict[int, Dict[str, object]],
    min_stars: int,
    fof_ll: float,
    fof_vel_ll: Optional[float],
    backend: str,
    cc_backend: str,
    max_pairs_per_batch: int,
    device_id: Optional[int] = None,
) -> List[Dict[str, np.ndarray | int]]:
    tasks_desc = sorted(tasks, key=lambda t: (int(t.depth), int(t.node_id)), reverse=True)
    carry_claimed: Dict[int, np.ndarray] = {}
    final_candidates: List[Dict[str, np.ndarray | int]] = []

    for task in tasks_desc:
        descendant_claimed = np.asarray(
            carry_claimed.pop(int(task.node_id), np.empty(0, dtype=np.int64)),
            dtype=np.int64,
        )
        if descendant_claimed.size == 0:
            node_candidates = list(initial_candidates_by_node.get(int(task.node_id), []))
        else:
            node_candidates = _fof_on_task_payload(
                task_payload=task_payloads_by_node[int(task.node_id)],
                min_stars=int(min_stars),
                fof_ll=float(fof_ll),
                fof_vel_ll=fof_vel_ll,
                backend=backend,
                cc_backend=cc_backend,
                max_pairs_per_batch=int(max_pairs_per_batch),
                blocked_star_idx=descendant_claimed,
                device_id=device_id,
            )

        node_claimed = np.asarray(descendant_claimed, dtype=np.int64)
        for gal in node_candidates:
            node_claimed = _array_union(node_claimed, np.asarray(gal["slist"], dtype=np.int64))
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
    compute_missing_halo_properties: bool = True,
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
        _ensure_missing_ahf_halos(
            sim,
            missing_top_ids,
            ahf_particles_file,
            pid_maps_sel,
            recompute_properties=bool(compute_missing_halo_properties),
        )
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


def _populate_caesar_ahf_lineage_indexes(sim) -> None:
    ahf_to_groupid: Dict[int, int] = {}
    for halo in getattr(sim, "halo_list", []):
        try:
            ahf_to_groupid[int(getattr(halo, "AHF_haloID", -1))] = int(getattr(halo, "GroupID", -1))
        except Exception:
            continue

    for halo in getattr(sim, "halo_list", []):
        try:
            parent_ahf = int(getattr(halo, "AHF_parent_haloID", -1))
        except Exception:
            parent_ahf = -1
        try:
            top_ahf = int(getattr(halo, "AHF_top_haloID", getattr(halo, "AHF_haloID", -1)))
        except Exception:
            top_ahf = -1
        halo.caesar_parent_halo_index = int(ahf_to_groupid.get(parent_ahf, -1))
        halo.caesar_top_halo_index = int(ahf_to_groupid.get(top_ahf, -1))

    for gal in getattr(sim, "galaxy_list", []):
        try:
            parent_ahf = int(getattr(gal, "AHF_parent_haloID", -1))
        except Exception:
            parent_ahf = -1
        try:
            top_ahf = int(getattr(gal, "AHF_top_haloID", -1))
        except Exception:
            top_ahf = -1
        gal.caesar_parent_halo_index = int(ahf_to_groupid.get(parent_ahf, -1))
        gal.caesar_top_halo_index = int(ahf_to_groupid.get(top_ahf, -1))


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
    _populate_caesar_ahf_lineage_indexes(sim)

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


def _direct_group_global_indexes(sim, group) -> np.ndarray:
    dm = sim.data_manager
    blocks = []
    for group_attr, dm_attr in (
        ("glist", "glist"),
        ("slist", "slist"),
        ("dmlist", "dmlist"),
        ("bhlist", "bhlist"),
        ("dlist", "dlist"),
    ):
        if not hasattr(group, group_attr) or not hasattr(dm, dm_attr):
            continue
        local = np.asarray(getattr(group, group_attr), dtype=np.int64)
        if local.size == 0:
            continue
        blocks.append(np.asarray(getattr(dm, dm_attr)[local], dtype=np.int64))
    if not blocks:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(blocks).astype(np.int64, copy=False)


def _build_direct_stage3_runtime(
    state,
    *,
    galaxy_payloads: Sequence[Dict[str, np.ndarray | int]],
    nproc: int,
    kwargs: Optional[Dict[str, object]] = None,
    host_filter_ids: Optional[Set[int]] = None,
):
    import os
    from types import SimpleNamespace

    from caesar.group import create_new_group
    from caesar.main import CAESAR
    from caesar.property_manager import ptype_ints

    sim = CAESAR()
    sim._kwargs = dict(kwargs or {})
    sim.units = dict(state.snapshot.units)
    sim.load_pot = True
    sim.load_haloid = False
    sim.skip_hash_check = True
    sim.nproc = int(max(1, nproc))
    sim._ds = _ShardYTUnitHelper(redshift=float(state.snapshot.redshift))

    z = float(state.snapshot.redshift)
    a = float(state.snapshot.scale_factor)
    om0 = float(state.snapshot.omega_matter)
    ol0 = float(state.snapshot.omega_lambda)
    ok0 = float(state.snapshot.omega_curvature)
    ez = float(np.sqrt(ol0 + ok0 * (1.0 + z) ** 2 + om0 * (1.0 + z) ** 3))
    omz = float(om0 * (1.0 + z) ** 3 / (ez * ez)) if ez > 0.0 else float(om0)
    h0_s = float(state.snapshot.hubble_constant) * 100.0 * 3.24077929e-20
    hz_s = float(h0_s * ez)

    sim.simulation.cosmological_simulation = True
    sim.simulation.XH = 0.76
    sim.simulation.redshift = z
    sim.simulation.scale_factor = a
    sim.simulation.omega_matter = om0
    sim.simulation.omega_lambda = ol0
    sim.simulation.omega_baryon = 0.0
    sim.simulation.hubble_constant = float(state.snapshot.hubble_constant)
    sim.simulation.Om_z = omz
    sim.simulation.E_z = ez
    sim.simulation.fullpath = os.path.dirname(str(state.snapshot.snapshot_file))
    sim.simulation.basename = os.path.basename(str(state.snapshot.snapshot_file))
    sim.simulation.parameters = {}
    sim.simulation.ds_type = "GadgetHDF5Dataset"
    sim.simulation.time = sim.yt_dataset.quan(float(state.snapshot.time_gyr), "Gyr")
    sim.simulation.H_z = sim.yt_dataset.quan(hz_s, "1/s")
    sim.simulation.G = sim.yt_dataset.quan(4.51691362044e-39, "kpc**3/(Msun * s**2)")
    sim.simulation.boxsize = sim.yt_dataset.quan(float(state.snapshot.boxsize), sim.units["length"])
    sim.simulation.boxsize_units = str(sim.simulation.boxsize.units)
    sim.simulation.search_radius = sim.yt_dataset.arr([300.0, 1000.0, 3000.0], sim.units["length"])
    sim.simulation.critical_density = sim.yt_dataset.quan(
        float(state.snapshot.critical_density_msun_kpc3),
        "Msun/kpc**3",
    )
    sim.simulation.Densities = sim.yt_dataset.arr(
        np.asarray(
            [
                200.0 * float(state.snapshot.critical_density_msun_kpc3),
                500.0 * float(state.snapshot.critical_density_msun_kpc3),
                2500.0 * float(state.snapshot.critical_density_msun_kpc3),
            ],
            dtype=np.float64,
        ),
        "Msun/kpc**3",
    )
    sim.simulation.ngas = int(state.snapshot.particle_counts.get("gas", 0))
    sim.simulation.nstar = int(state.snapshot.particle_counts.get("star", 0))
    sim.simulation.nbh = int(state.snapshot.particle_counts.get("bh", 0))
    sim.simulation.ndust = int(state.snapshot.particle_counts.get("dust", 0))
    sim.simulation.ndm = int(state.snapshot.particle_counts.get("dm", 0))
    sim.simulation.ndm2 = int(state.snapshot.particle_counts.get("dm2", 0))
    sim.simulation.ndm3 = int(state.snapshot.particle_counts.get("dm3", 0))
    sim.simulation.ntot = int(sum(int(v) for v in state.snapshot.particle_counts.values()))
    sim.simulation.baryons_present = bool(sim.simulation.ngas > 0 or sim.simulation.nstar > 0)
    sim.simulation.unbind_halos = False
    sim.simulation.effective_resolution = int(_effective_resolution_from_ndm(sim.simulation.ndm))
    sim.simulation.mean_interparticle_separation = sim.yt_dataset.quan(
        _mean_interparticle_separation_from_boxsize(float(state.snapshot.boxsize), sim.simulation.ndm),
        sim.units["length"],
    )

    ordered_ptypes = [
        ptype for ptype in ("gas", "dm", "dm2", "dm3", "star", "bh", "dust")
        if state.particles.has_ptype(ptype)
    ]
    offsets: Dict[str, int] = {}
    pos_blocks = []
    vel_blocks = []
    mass_blocks = []
    pot_blocks = []
    ptype_blocks = []

    dm = SimpleNamespace()
    dm.ptypes = list(ordered_ptypes)
    dm.blackholes = "bh" in ordered_ptypes
    dm.dust = "dust" in ordered_ptypes

    offset = 0
    for ptype in ordered_ptypes:
        table = state.particles.table(ptype)
        count = len(table)
        offsets[ptype] = int(offset)
        setattr(dm, f"{ptype}list" if ptype != "star" else "slist", np.arange(count, dtype=np.int64) + int(offset))
        if ptype == "gas":
            dm.glist = np.arange(count, dtype=np.int64) + int(offset)
        elif ptype == "dm":
            dm.dmlist = np.arange(count, dtype=np.int64) + int(offset)
        elif ptype == "bh":
            dm.bhlist = np.arange(count, dtype=np.int64) + int(offset)
        elif ptype == "dust":
            dm.dlist = np.arange(count, dtype=np.int64) + int(offset)

        pos_blocks.append(np.asarray(table.get("pos"), dtype=np.float32))
        vel_blocks.append(np.asarray(table.get("vel"), dtype=np.float32))
        mass_blocks.append(np.asarray(table.get("mass"), dtype=np.float32))
        pot_blocks.append(np.asarray(table.get("pot", np.zeros(count, dtype=np.float32)), dtype=np.float32))
        ptype_blocks.append(np.full(count, ptype_ints[ptype], dtype=np.int32))
        for attr in ("gnh", "gsfr", "gZ", "gT", "gfH2", "gfHI", "dustmass", "sZ", "age", "bhmass", "bhmdot"):
            if table.has(attr):
                setattr(dm, attr, np.asarray(table.get(attr)))
        offset += count

    dm.pos = np.concatenate(pos_blocks, axis=0) if pos_blocks else np.empty((0, 3), dtype=np.float32)
    dm.vel = np.concatenate(vel_blocks, axis=0) if vel_blocks else np.empty((0, 3), dtype=np.float32)
    dm.mass = np.concatenate(mass_blocks, axis=0) if mass_blocks else np.empty(0, dtype=np.float32)
    dm.pot = np.concatenate(pot_blocks, axis=0) if pot_blocks else np.empty(0, dtype=np.float32)
    dm.ptype = np.concatenate(ptype_blocks, axis=0) if ptype_blocks else np.empty(0, dtype=np.int32)
    dm.indexes = np.arange(dm.mass.size, dtype=np.int64)
    sim._dm = dm
    sim._ds_type = _ShardDatasetType(ptypes=dm.ptypes, data_manager_attrs=set(dm.__dict__.keys()))
    sim.group_types = ["halo", "galaxy"]
    setattr(sim, "_ahf_matched", True)
    setattr(sim, "_include_dm_in_galaxies", True)
    # Save the full CAESAR schema, but stream global reverse maps at write time
    # instead of materializing them all in RAM on rank 0.
    setattr(sim, "_ahf_subhalo_streaming_save", True)

    host_filter = None if host_filter_ids is None else {int(v) for v in host_filter_ids}

    def _keep_halo(node_index: int, required_host_ids: set[int]) -> bool:
        halo_id = int(state.nodes.halo_id[node_index])
        parent_id = int(state.nodes.parent_halo_id[node_index])
        top_id = int(state.nodes.top_halo_id[node_index])
        if host_filter is not None and int(top_id) not in host_filter:
            return False
        dm_count = int(state.nodes.dm_count[node_index])
        min_dm = int(MINIMUM_DM_PER_TOPLEVEL_AHF_HALO) if parent_id <= 0 else int(MINIMUM_DM_PER_AHF_SUBHALO)
        return dm_count >= min_dm or halo_id in required_host_ids

    required_host_ids = {int(rec["AHF_top_haloID"]) for rec in galaxy_payloads}
    ahf_to_halo_index: Dict[int, int] = {}
    halo_list = []
    for node_index in range(len(state.nodes)):
        if not _keep_halo(node_index, required_host_ids):
            continue
        halo = create_new_group(sim, "halo")
        halo.AHF_haloID = int(state.nodes.halo_id[node_index])
        halo.AHF_parent_haloID = int(state.nodes.parent_halo_id[node_index])
        halo.AHF_top_haloID = int(state.nodes.top_halo_id[node_index])
        halo.AHF_depth = int(state.nodes.depth[node_index])
        halo.AHF_ancestor_haloIDs = np.asarray(state.nodes.ancestors_for(node_index), dtype=np.int64)
        halo.glist = np.asarray(state.nodes.members_for(node_index, "gas"), dtype=np.int32)
        halo.slist = np.asarray(state.nodes.members_for(node_index, "star"), dtype=np.int32)
        halo.dmlist = np.asarray(state.nodes.members_for(node_index, "dm"), dtype=np.int32)
        halo.bhlist = np.asarray(state.nodes.members_for(node_index, "bh"), dtype=np.int32)
        halo.dlist = np.asarray(state.nodes.members_for(node_index, "dust"), dtype=np.int32)
        halo.galaxy_index_list = np.empty(0, dtype=np.int32)
        halo.global_indexes = _direct_group_global_indexes(sim, halo)
        ahf_to_halo_index[int(halo.AHF_haloID)] = len(halo_list)
        halo_list.append(halo)

    galaxy_list = []
    for payload in galaxy_payloads:
        gal = create_new_group(sim, "galaxy")
        gal.AHF_haloID = int(payload["AHF_haloID"])
        gal.AHF_parent_haloID = int(payload.get("AHF_parent_haloID", -1))
        gal.AHF_top_haloID = int(payload.get("AHF_top_haloID", -1))
        gal.AHF_depth = int(payload.get("AHF_depth", 0))
        gal.AHF_ancestor_haloIDs = np.asarray(payload.get("AHF_ancestor_haloIDs", []), dtype=np.int64)
        gal.glist = np.asarray(payload.get("glist", []), dtype=np.int32)
        gal.slist = np.asarray(payload.get("slist", []), dtype=np.int32)
        gal.dmlist = np.asarray(payload.get("dmlist", []), dtype=np.int32)
        gal.bhlist = np.asarray(payload.get("bhlist", []), dtype=np.int32)
        gal.dlist = np.asarray(payload.get("dlist", []), dtype=np.int32)
        gal.parent_halo_index = int(ahf_to_halo_index.get(int(gal.AHF_top_haloID), -1))
        gal._ahf_host_halo_index = int(gal.parent_halo_index)
        if "_merge_id" in payload:
            gal._merge_id = int(payload["_merge_id"])
        gal.global_indexes = _direct_group_global_indexes(sim, gal)
        galaxy_list.append(gal)

    for halo in halo_list:
        halo.galaxy_index_list = []
    for gi, gal in enumerate(galaxy_list):
        host_index = int(getattr(gal, "parent_halo_index", -1))
        if 0 <= host_index < len(halo_list):
            halo_list[host_index].galaxy_index_list.append(int(gi))
    for halo in halo_list:
        halo.galaxy_index_list = np.asarray(getattr(halo, "galaxy_index_list", []), dtype=np.int32)

    sim.halo_list = halo_list
    sim.halos = halo_list
    sim.nhalos = len(halo_list)
    sim.galaxy_list = galaxy_list
    sim.galaxies = galaxy_list
    sim.ngalaxies = len(galaxy_list)
    sim._ahf_galaxy_hosts = [int(getattr(gal, "parent_halo_index", -1)) for gal in galaxy_list]
    sim._ahf_galaxy_ahf_ids = [int(getattr(gal, "AHF_haloID", -1)) for gal in galaxy_list]
    sim._ahf_galaxy_top_ahf_ids = [int(getattr(gal, "AHF_top_haloID", -1)) for gal in galaxy_list]
    for gal in galaxy_list:
        idx = int(getattr(gal, "parent_halo_index", -1))
        gal.halo = halo_list[idx] if 0 <= idx < len(halo_list) else None
    return sim


def _complete_finalization_after_properties_direct(sim):
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
    _populate_caesar_ahf_lineage_indexes(sim)

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


def _adopt_caesar_runtime(target, source) -> None:
    original_args = getattr(target, "_args", ())
    original_kwargs = dict(getattr(target, "_kwargs", {}))
    target.__dict__.clear()
    target.__dict__.update(source.__dict__)
    target._args = original_args
    target._kwargs = original_kwargs

    if hasattr(target, "global_particle_lists") and hasattr(target.global_particle_lists, "obj"):
        target.global_particle_lists.obj = target

    for halo in getattr(target, "halo_list", []):
        halo.obj = target
    for gal in getattr(target, "galaxy_list", []):
        gal.obj = target
        host_index = int(getattr(gal, "parent_halo_index", -1))
        if 0 <= host_index < len(getattr(target, "halo_list", [])):
            gal.halo = target.halo_list[host_index]
        else:
            gal.halo = None


def _run_ahf_subhalo_direct(
    snapshot_file: str,
    ahf_particles_file: str,
    *,
    kwargs: Optional[Dict[str, object]] = None,
    nproc: int = 1,
    min_stars: int,
):
    from caesar.ahf_subhalo_hdf5 import build_direct_state, build_task_payload

    state = build_direct_state(snapshot_file, ahf_particles_file)
    tasks, tasks_by_root = _build_task_manifest_from_direct_state(state, min_stars=int(min_stars))

    fof_ll = _direct_fof_linking_length(state.snapshot, kwargs=kwargs)
    fof_vel_ll = 1.0
    try:
        _vel_env = os.environ.get("CAESAR_FOF6D_VEL_LL")
        if _vel_env not in (None, ""):
            fof_vel_ll = float(_vel_env)
    except Exception:
        pass
    if os.environ.get("CAESAR_FOF6D_DISABLE_VEL", "0") == "1":
        fof_vel_ll = None

    fof_nHlim = _env_float("CAESAR_AHF_FAST_FOF_NHLIM", 0.13)
    fof_Tlim = _env_float("CAESAR_AHF_FAST_FOF_TLIM", 1.0e5)
    fof_use_sfr_gate = os.environ.get("CAESAR_AHF_FAST_FOF_USE_SFR", "1") == "1"
    backend = _env_str("CAESAR_AHF_SUBHALO_BACKEND", "auto").lower()
    if backend == "cpu":
        backend = "numpy"
    elif backend == "gpu":
        backend = "cupy"
    cc_backend = _env_str("CAESAR_AHF_SUBHALO_CC_BACKEND", "auto").lower()
    max_pairs_per_batch = max(1, _env_int("CAESAR_AHF_SUBHALO_MAX_PAIRS_PER_BATCH", 5_000_000))
    tiny_star_threshold = max(int(min_stars), _env_int("CAESAR_AHF_SUBHALO_TINY_STARS", 32))
    tiny_max_nodes_per_batch = max(1, _env_int("CAESAR_AHF_SUBHALO_TINY_MAX_NODES", 64))
    tiny_max_fof_candidates_per_batch = max(
        1,
        _env_int("CAESAR_AHF_SUBHALO_TINY_MAX_FOF_CANDIDATES", 250_000),
    )

    task_payloads_by_node: Dict[int, Dict[str, object]] = {}
    filtered_tasks: List[AHFSubhaloTask] = []
    filtered_tasks_by_root: Dict[int, List[AHFSubhaloTask]] = {}
    for task in tasks:
        payload = build_task_payload(
            state,
            task=task,
            fof_nHlim=float(fof_nHlim),
            fof_Tlim=float(fof_Tlim),
            fof_use_sfr_gate=bool(fof_use_sfr_gate),
        )
        if payload is None or int(np.asarray(payload["star_sel"]).size) < int(min_stars):
            continue
        payload["task"] = _serialize_task(task)
        task_payloads_by_node[int(task.node_id)] = payload
        filtered_tasks.append(task)
        filtered_tasks_by_root.setdefault(int(task.top_id), []).append(task)

    tiny_tasks = [task for task in filtered_tasks if int(task.star_count) <= int(tiny_star_threshold)]
    regular_tasks = [task for task in filtered_tasks if int(task.star_count) > int(tiny_star_threshold)]
    work_batches = [
        AHFSubhaloBatch(tasks=(task,), is_tiny_batch=False, estimated_cost=int(task.fof_candidates))
        for task in sorted(regular_tasks, key=lambda task: int(task.fof_candidates), reverse=True)
    ]
    work_batches.extend(
        _build_tiny_batches(
            tiny_tasks,
            max_nodes_per_batch=int(tiny_max_nodes_per_batch),
            max_fof_candidates_per_batch=int(tiny_max_fof_candidates_per_batch),
        )
    )

    initial_candidates_by_node: Dict[int, List[Dict[str, np.ndarray | int]]] = {
        int(task.node_id): [] for task in filtered_tasks
    }
    for batch in work_batches:
        batch_payload = {
            "batch": _serialize_batch(batch),
            "task_payloads": [task_payloads_by_node[int(task.node_id)] for task in batch.tasks],
        }
        batch_results = _fof_on_batch_payload(
            batch_payload=batch_payload,
            min_stars=int(min_stars),
            fof_ll=float(fof_ll),
            fof_vel_ll=fof_vel_ll,
            backend=backend,
            cc_backend=cc_backend,
            max_pairs_per_batch=int(max_pairs_per_batch),
        )
        for node_id, records in batch_results.items():
            initial_candidates_by_node[int(node_id)].extend(list(records))

    final_galaxies: List[Dict[str, np.ndarray | int]] = []
    for root_id, root_tasks in sorted(filtered_tasks_by_root.items(), key=lambda kv: int(kv[0])):
        final_galaxies.extend(
            _reconcile_root_payload(
                tasks=root_tasks,
                initial_candidates_by_node=initial_candidates_by_node,
                task_payloads_by_node=task_payloads_by_node,
                min_stars=int(min_stars),
                fof_ll=float(fof_ll),
                fof_vel_ll=fof_vel_ll,
                backend=backend,
                cc_backend=cc_backend,
                max_pairs_per_batch=int(max_pairs_per_batch),
            )
        )

    sim = _build_direct_stage3_runtime(
        state,
        galaxy_payloads=final_galaxies,
        nproc=int(max(1, nproc)),
        kwargs=dict(kwargs or {}),
    )
    _compute_group_properties_subset(sim, group_type="halo", groups=list(sim.halo_list))
    _compute_group_properties_subset(sim, group_type="galaxy", groups=list(sim.galaxy_list))
    _complete_finalization_after_properties_direct(sim)
    return sim


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
    node_ndm = _build_node_dm_counts(membership_arrays)

    host_to_nodes, _ = _group_nodes_by_root(parent_of)
    tasks, tasks_by_root = _build_task_manifest(
        parent_of=parent_of,
        host_to_nodes=host_to_nodes,
        node_npart=node_npart,
        node_nstar=node_nstar,
        node_ndm=node_ndm,
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

    if not obj._kwargs.get("haloid_file"):
        raise ValueError("AHF-subhalo requires an AHF_particles file via haloid_file.")

    from caesar.group import get_min_stars

    ms = get_min_stars(obj)
    snapshot_file = _snapshot_file_from_obj(obj)
    runtime = _run_ahf_subhalo_direct(
        snapshot_file,
        obj._kwargs["haloid_file"],
        kwargs=dict(getattr(obj, "_kwargs", {})),
        nproc=int(obj.nproc),
        min_stars=int(ms),
    )
    _adopt_caesar_runtime(obj, runtime)
