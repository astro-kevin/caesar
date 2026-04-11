from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


try:
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover
    MPI = None


from caesar.AHF_FAST_halos import build_halos_from_ahf_fast
from caesar.AHF_FAST_loader import load_ahf_halos_dataframe, load_ahf_hierarchy
from caesar.AHF_subhalo import (
    AHFSubhaloBatch,
    AHFSubhaloTask,
    _available_gpu_device_ids,
    _build_node_dm_counts,
    _build_stage3_property_payload,
    _build_stage3_property_runtime,
    _build_task_input_payload,
    _build_task_manifest,
    _build_tiny_batches,
    _complete_finalization_after_properties,
    _compute_group_properties_subset,
    _deserialize_candidate_group,
    _deserialize_task,
    _env_float,
    _env_int,
    _env_str,
    _fof_on_batch_payload,
    _fof_on_task_payload,
    _prepare_final_subhalo_galaxies,
    _reconcile_root_payload,
    _serialize_group_state,
    _serialize_candidate_group,
    _serialize_task,
    _apply_group_state,
)
from caesar.fubar import get_b as _get_b
from caesar.fubar import get_mean_interparticle_separation as _get_mean_interparticle_separation
from caesar.group import get_min_stars
from caesar.halo_matching import _build_selected_pid_maps, _group_nodes_by_root


MSG_REGISTER = 10
MSG_WORK = 11
MSG_RESULT = 12
MSG_STOP = 13


@dataclass(frozen=True)
class WorkerCapability:
    rank: int
    role: str
    hostname: str
    local_rank: int
    gpu_device: Optional[int]
    threads: int


@dataclass
class _StageQueue:
    iterator: object
    total: int
    dispatched: int = 0
    exhausted: bool = False

    def next_item(self):
        if self.exhausted:
            return None
        try:
            item = next(self.iterator)
        except StopIteration:
            self.exhausted = True
            return None
        self.dispatched += 1
        return item

    @property
    def remaining(self) -> int:
        return max(0, int(self.total) - int(self.dispatched))


def _serialize_batch(batch: AHFSubhaloBatch) -> Dict[str, object]:
    return {
        "tasks": [_serialize_task(task) for task in batch.tasks],
        "is_tiny_batch": bool(batch.is_tiny_batch),
        "target_backend": str(batch.target_backend),
        "target_device": batch.target_device,
        "estimated_cost": int(batch.estimated_cost),
    }


def _deserialize_batch(payload: Dict[str, object]) -> AHFSubhaloBatch:
    return AHFSubhaloBatch(
        tasks=tuple(_deserialize_task(task) for task in payload["tasks"]),
        is_tiny_batch=bool(payload.get("is_tiny_batch", False)),
        target_backend=str(payload.get("target_backend", "cpu")),
        target_device=payload.get("target_device"),
        estimated_cost=int(payload.get("estimated_cost", 0)),
    )


def _local_rank() -> int:
    for key in ("SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"):
        raw = os.environ.get(key)
        if raw not in (None, ""):
            try:
                return int(raw)
            except Exception:
                pass
    return 0


def _gpu_device_for_rank(
    *,
    rank: int,
    role: str,
    world_layout: Sequence[Dict[str, object]],
) -> Optional[int]:
    if str(role) != "gpu_worker":
        return None
    devices = _available_gpu_device_ids()
    if not devices:
        return None

    mine = next((entry for entry in world_layout if int(entry["rank"]) == int(rank)), None)
    if mine is None:
        return None
    hostname = str(mine["hostname"])
    gpu_workers = [
        entry
        for entry in world_layout
        if str(entry["role"]) == "gpu_worker" and str(entry["hostname"]) == hostname
    ]
    gpu_workers.sort(key=lambda entry: (int(entry["local_rank"]), int(entry["rank"])))
    gpu_index = next(
        (idx for idx, entry in enumerate(gpu_workers) if int(entry["rank"]) == int(rank)),
        None,
    )
    if gpu_index is None:
        return None
    if 0 <= int(gpu_index) < len(devices):
        return int(devices[int(gpu_index)])
    return None


def _worker_capability(
    rank: int,
    *,
    role: str,
    threads: int,
    world_layout: Sequence[Dict[str, object]],
) -> WorkerCapability:
    hostname = os.uname().nodename
    local_rank = _local_rank()
    gpu_device = _gpu_device_for_rank(rank=rank, role=role, world_layout=world_layout)
    return WorkerCapability(
        rank=int(rank),
        role=str(role),
        hostname=str(hostname),
        local_rank=int(local_rank),
        gpu_device=gpu_device,
        threads=int(max(1, threads)),
    )


def _dump_pickle(path: Path, payload) -> None:
    with path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def _stage1_shard_payload(results_by_node: Dict[int, List]) -> Dict[int, List[dict]]:
    out = {}
    for node_id, gals in results_by_node.items():
        out[int(node_id)] = [
            gal if isinstance(gal, dict) else _serialize_candidate_group(gal)
            for gal in gals
        ]
    return out


def _stage2_shard_payload(galaxies: Sequence) -> List[dict]:
    return [gal if isinstance(gal, dict) else _serialize_candidate_group(gal) for gal in galaxies]


def _prepare_manifest(
    *,
    ahf_particles_file: str,
    node_ndm: Optional[Dict[int, int]] = None,
    min_stars: int,
) -> Tuple[List[AHFSubhaloTask], Dict[int, List[AHFSubhaloTask]], Dict[int, int], Dict[int, int]]:
    hier = load_ahf_hierarchy(ahf_particles_file)
    halos_df = load_ahf_halos_dataframe(ahf_particles_file)
    parent_of = hier.parent_of
    host_to_nodes, _ = _group_nodes_by_root(parent_of)
    node_npart = {
        int(hid): int(npart)
        for hid, npart in zip(halos_df["hid"].to_numpy(), halos_df["npart"].to_numpy())
    }
    node_nstar = {
        int(hid): max(0, int(nstar))
        for hid, nstar in zip(halos_df["hid"].to_numpy(), halos_df["n_star"].to_numpy())
    }
    tasks, tasks_by_root = _build_task_manifest(
        parent_of=parent_of,
        host_to_nodes=host_to_nodes,
        node_npart=node_npart,
        node_nstar=node_nstar,
        node_ndm=node_ndm,
        min_stars=int(min_stars),
    )
    return tasks, tasks_by_root, node_npart, node_nstar


def _classify_stage1_batches(
    *,
    tasks: Sequence[AHFSubhaloTask],
    gpu_worker_count: int,
    cpu_worker_count: int,
) -> Tuple[List[AHFSubhaloBatch], List[AHFSubhaloBatch]]:
    tiny_star_threshold = max(_env_int("CAESAR_AHF_SUBHALO_TINY_STARS", 32), 1)
    tiny_max_nodes_per_batch = max(1, _env_int("CAESAR_AHF_SUBHALO_TINY_MAX_NODES", 64))
    tiny_max_fof_candidates_per_batch = max(
        1,
        _env_int("CAESAR_AHF_SUBHALO_TINY_MAX_FOF_CANDIDATES", 250_000),
    )
    gpu_min_fof = max(1, _env_int("CAESAR_AHF_SUBHALO_GPU_MIN_FOF", 50_000))

    tiny_tasks = [task for task in tasks if int(task.star_count) <= int(tiny_star_threshold)]
    regular_tasks = [task for task in tasks if int(task.star_count) > int(tiny_star_threshold)]
    tiny_batches = _build_tiny_batches(
        tiny_tasks,
        max_nodes_per_batch=int(tiny_max_nodes_per_batch),
        max_fof_candidates_per_batch=int(tiny_max_fof_candidates_per_batch),
    )

    gpu_regular = [task for task in regular_tasks if int(task.fof_candidates) >= int(gpu_min_fof)]
    cpu_regular = [task for task in regular_tasks if int(task.fof_candidates) < int(gpu_min_fof)]

    gpu_items = [
        AHFSubhaloBatch(tasks=(task,), is_tiny_batch=False, target_backend="gpu", estimated_cost=int(task.fof_candidates))
        for task in sorted(gpu_regular, key=lambda t: int(t.fof_candidates), reverse=True)
    ]
    cpu_items = [
        AHFSubhaloBatch(tasks=(task,), is_tiny_batch=False, target_backend="cpu", estimated_cost=int(task.fof_candidates))
        for task in sorted(cpu_regular, key=lambda t: int(t.fof_candidates), reverse=True)
    ]
    cpu_items.extend(tiny_batches)

    if gpu_worker_count <= 0:
        cpu_items = gpu_items + cpu_items
        gpu_items = []
    if cpu_worker_count <= 0 and cpu_items:
        gpu_items.extend(
            AHFSubhaloBatch(
                tasks=item.tasks,
                is_tiny_batch=bool(item.is_tiny_batch),
                target_backend="gpu",
                estimated_cost=int(item.estimated_cost),
            )
            for item in cpu_items
        )
        cpu_items = []

    return gpu_items, cpu_items


def _classify_stage2_roots(
    *,
    tasks_by_root: Dict[int, List[AHFSubhaloTask]],
    gpu_worker_count: int,
    cpu_worker_count: int,
) -> Tuple[List[int], List[int]]:
    gpu_min_root_fof = max(1, _env_int("CAESAR_AHF_SUBHALO_GPU_MIN_ROOT_FOF", 250_000))
    gpu_roots = []
    cpu_roots = []
    for root_id, tasks in tasks_by_root.items():
        total = int(sum(int(task.fof_candidates) for task in tasks))
        if gpu_worker_count > 0 and total >= int(gpu_min_root_fof):
            gpu_roots.append(int(root_id))
        else:
            cpu_roots.append(int(root_id))
    if gpu_worker_count <= 0:
        cpu_roots = gpu_roots + cpu_roots
        gpu_roots = []
    if cpu_worker_count <= 0 and cpu_roots:
        gpu_roots.extend(cpu_roots)
        cpu_roots = []
    return gpu_roots, cpu_roots


def _rank0_log(message: str) -> None:
    print(f"[AHF-subhalo][rank0] {message}", flush=True)


def _stage_queue(source, *, total: Optional[int] = None) -> _StageQueue:
    resolved_total = int(total) if total is not None else -1
    if resolved_total < 0:
        try:
            resolved_total = int(len(source))
        except Exception:
            resolved_total = 0
    return _StageQueue(iterator=iter(source), total=int(resolved_total))


def _cpu_local_workers(cap: WorkerCapability, *, stage_name: str) -> int:
    if str(cap.role) != "cpu_worker" or str(stage_name) not in {"stage1", "stage2"}:
        return 1
    default = max(1, int(cap.threads))
    configured = max(
        1,
        _env_int(f"CAESAR_AHF_SUBHALO_MPI_{str(stage_name).upper()}_LOCAL_WORKERS", default),
    )
    return int(min(default, configured))


def _take_cpu_bundle(queue: _StageQueue, *, max_items: int):
    items = []
    for _ in range(max(1, int(max_items))):
        item = queue.next_item()
        if item is None:
            break
        items.append(item)
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    bundle = dict(items[0])
    bundle["items"] = items
    bundle["completed_count"] = len(items)
    return bundle


def _dispatch_stage(
    comm,
    *,
    worker_caps: Sequence[WorkerCapability],
    gpu_queue,
    cpu_queue,
    stage_name: str,
    gpu_total: Optional[int] = None,
    cpu_total: Optional[int] = None,
):
    gpu_state = _stage_queue(gpu_queue, total=gpu_total)
    cpu_state = _stage_queue(cpu_queue, total=cpu_total)
    total_items = int(gpu_state.total) + int(cpu_state.total)
    active = 0
    results = []
    completed = 0
    progress_every = max(1, _env_int("CAESAR_AHF_SUBHALO_MPI_PROGRESS_EVERY", 100))
    progress_seconds = max(5.0, _env_float("CAESAR_AHF_SUBHALO_MPI_PROGRESS_SECONDS", 60.0))
    start_time = time.monotonic()
    last_heartbeat = start_time
    last_completion_log = 0

    def log_progress(prefix: str) -> None:
        elapsed = time.monotonic() - start_time
        _rank0_log(
            f"{stage_name}: {prefix}; completed={completed}/{total_items}, "
            f"active={active}, dispatched_gpu={gpu_state.dispatched}/{gpu_state.total}, "
            f"dispatched_cpu={cpu_state.dispatched}/{cpu_state.total}, "
            f"remaining_gpu={gpu_state.remaining}, remaining_cpu={cpu_state.remaining}, "
            f"elapsed={elapsed:.1f}s"
        )

    def assign_next(rank: int, cap: WorkerCapability):
        nonlocal active
        item = None
        if cap.role == "gpu_worker" and cap.gpu_device is not None:
            item = gpu_state.next_item()
        elif cap.role == "cpu_worker":
            item = _take_cpu_bundle(
                cpu_state,
                max_items=_cpu_local_workers(cap, stage_name=stage_name),
            )
        elif cap.role == "prop_worker":
            item = cpu_state.next_item()
        if item is None and cap.role == "gpu_worker":
            item = gpu_state.next_item()
        if item is None and cap.role == "cpu_worker":
            item = _take_cpu_bundle(
                cpu_state,
                max_items=_cpu_local_workers(cap, stage_name=stage_name),
            )
        if item is None and cap.role == "prop_worker":
            item = cpu_state.next_item()

        if item is None:
            return False

        payload = {"type": "work", "stage": stage_name, "item": item}
        comm.send(payload, dest=rank, tag=MSG_WORK)
        active += 1
        return True

    log_progress("starting")
    for cap in worker_caps:
        assign_next(cap.rank, cap)
    log_progress("initial assignments queued")

    while active > 0:
        if not comm.iprobe(source=MPI.ANY_SOURCE, tag=MSG_RESULT):
            now = time.monotonic()
            if now - last_heartbeat >= progress_seconds:
                log_progress("heartbeat")
                last_heartbeat = now
            time.sleep(0.5)
            continue
        status = MPI.Status()
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MSG_RESULT, status=status)
        src = int(status.Get_source())
        active -= 1
        results.append(msg)
        completed += int(msg.get("completed_count", 1))
        cap = next(cap for cap in worker_caps if int(cap.rank) == int(src))
        assign_next(src, cap)
        if completed == total_items or completed - last_completion_log >= progress_every:
            log_progress("progress")
            last_completion_log = completed
            last_heartbeat = time.monotonic()

    log_progress("complete")
    return results


def _stop_workers(comm, *, worker_caps: Sequence[WorkerCapability]) -> None:
    for cap in worker_caps:
        comm.send({"type": "stop"}, dest=int(cap.rank), tag=MSG_STOP)


def _rank0_init_runtime(snapshot_file: str, ahf_particles_file: str, *, nproc: int):
    import yt
    import caesar

    ds = yt.load(snapshot_file)
    sim = caesar.CAESAR(ds)
    sim.nproc = int(nproc)
    sim.load_haloid = False
    build_halos_from_ahf_fast(sim, ahf_particles_file, compute_properties=False)
    pid_maps_sel = _build_selected_pid_maps(sim)
    fof_mis = float(_get_mean_interparticle_separation(sim).d)
    fof_ll = float(fof_mis * _get_b(sim, "galaxy"))

    fof_vel_ll = 1.0
    try:
        _vel_env = os.environ.get("CAESAR_FOF6D_VEL_LL")
        if _vel_env not in (None, ""):
            fof_vel_ll = float(_vel_env)
    except Exception:
        pass
    if os.environ.get("CAESAR_FOF6D_DISABLE_VEL", "0") == "1":
        fof_vel_ll = None

    return sim, pid_maps_sel, fof_ll, fof_vel_ll


def _rank0_prepare_stage12(
    *,
    snapshot_file: str,
    ahf_particles_file: str,
    shard_root: Path,
    nproc: int,
    min_stars: Optional[int],
):
    sim, pid_maps_sel, fof_ll, fof_vel_ll = _rank0_init_runtime(
        snapshot_file,
        ahf_particles_file,
        nproc=int(nproc),
    )
    membership_arrays = getattr(sim, "_ahf_fast_memberships")
    ms = get_min_stars(sim, override=min_stars)
    tasks, tasks_by_root, _, _ = _prepare_manifest(
        ahf_particles_file=ahf_particles_file,
        node_ndm=_build_node_dm_counts(membership_arrays),
        min_stars=int(ms),
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

    fof_nHlim = _env_float("CAESAR_AHF_FAST_FOF_NHLIM", 0.13)
    fof_Tlim = _env_float("CAESAR_AHF_FAST_FOF_TLIM", 1.0e5)
    fof_use_sfr_gate = os.environ.get("CAESAR_AHF_FAST_FOF_USE_SFR", "1") == "1"

    task_payloads_by_node: Dict[int, Dict[str, object]] = {}
    filtered_tasks: List[AHFSubhaloTask] = []
    for task in tasks:
        payload = _build_task_input_payload(
            sim,
            task=task,
            membership_arrays=membership_arrays,
            pid_maps_sel=pid_maps_sel,
            fof_nHlim=float(fof_nHlim),
            fof_Tlim=float(fof_Tlim),
            fof_use_sfr_gate=bool(fof_use_sfr_gate),
        )
        if payload is None or int(np.asarray(payload["star_sel"]).size) < int(ms):
            continue
        task_payloads_by_node[int(task.node_id)] = payload
        filtered_tasks.append(task)

    filtered_ids = {int(task.node_id) for task in filtered_tasks}
    tasks = filtered_tasks
    tasks_by_root = {
        int(root): [task for task in root_tasks if int(task.node_id) in filtered_ids]
        for root, root_tasks in tasks_by_root.items()
    }
    tasks_by_root = {int(root): root_tasks for root, root_tasks in tasks_by_root.items() if root_tasks}

    return sim, pid_maps_sel, int(ms), float(fof_ll), fof_vel_ll, tasks, tasks_by_root, task_payloads_by_node


def _iter_stage1_batch_payloads(
    *,
    shard_root: Path,
    stage_batches: Sequence[AHFSubhaloBatch],
    prefix: str,
    task_payloads_by_node: Dict[int, Dict[str, object]],
    fof_ll: float,
    fof_vel_ll: Optional[float],
    min_stars: int,
):
    for idx, batch in enumerate(stage_batches):
        payload_path = shard_root / f"stage1_input_{prefix}_{idx:06d}.pkl"
        payload = {
            "batch": _serialize_batch(batch),
            "task_payloads": [task_payloads_by_node[int(task.node_id)] for task in batch.tasks],
        }
        _dump_pickle(payload_path, payload)
        yield {
            "batch": _serialize_batch(batch),
            "payload_path": str(payload_path),
            "backend": str(batch.target_backend),
            "device_id": batch.target_device,
            "fof_ll": float(fof_ll),
            "fof_vel_ll": fof_vel_ll,
            "min_stars": int(min_stars),
        }


def _write_stage2_root_payloads(
    *,
    shard_root: Path,
    tasks_by_root: Dict[int, List[AHFSubhaloTask]],
    task_payloads_by_node: Dict[int, Dict[str, object]],
) -> Dict[int, str]:
    root_payload_paths: Dict[int, str] = {}
    for root_id, root_tasks in tasks_by_root.items():
        payload_path = shard_root / f"stage2_input_root{int(root_id)}.pkl"
        payload = {
            "root_id": int(root_id),
            "tasks": [_serialize_task(task) for task in root_tasks],
            "task_payloads_by_node": {
                int(task.node_id): task_payloads_by_node[int(task.node_id)]
                for task in root_tasks
            },
        }
        _dump_pickle(payload_path, payload)
        root_payload_paths[int(root_id)] = str(payload_path)
    return root_payload_paths


def _worker_run_stage1(
    *,
    item: Dict,
    fof_ll: float,
    fof_vel_ll: Optional[float],
    min_stars: int,
    shard_dir: Path,
    nproc: int = 1,
) -> Dict:
    items = list(item.get("items", [item]))
    fof_use_sfr_gate = os.environ.get("CAESAR_AHF_FAST_FOF_USE_SFR", "1") == "1"
    cc_backend = _env_str("CAESAR_AHF_SUBHALO_CC_BACKEND", "auto").lower()
    max_pairs_per_batch = max(1, _env_int("CAESAR_AHF_SUBHALO_MAX_PAIRS_PER_BATCH", 5_000_000))

    def run_one(one_item: Dict):
        payload = _load_pickle(Path(one_item["payload_path"]))
        batch = _deserialize_batch(payload["batch"])
        results = _fof_on_batch_payload(
            batch_payload=payload,
            min_stars=int(min_stars),
            fof_ll=float(fof_ll),
            fof_vel_ll=fof_vel_ll,
            backend="cupy" if str(one_item.get("backend", batch.target_backend)) == "gpu" else "numpy",
            cc_backend=cc_backend if str(one_item.get("backend", batch.target_backend)) == "gpu" else "cpu",
            max_pairs_per_batch=int(max_pairs_per_batch),
            device_id=one_item.get("device_id", batch.target_device),
        )
        return {
            "results": _stage1_shard_payload(results),
            "node_ids": [int(t.node_id) for t in batch.tasks],
            "root_ids": sorted({int(t.top_id) for t in batch.tasks}),
            "input_payload_path": str(one_item["payload_path"]),
        }

    max_workers = max(1, min(int(nproc), len(items)))
    if max_workers > 1:
        outputs = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one, one_item) for one_item in items]
            for fut in as_completed(futures):
                outputs.append(fut.result())
    else:
        outputs = [run_one(one_item) for one_item in items]

    combined_payload = {}
    node_ids = []
    root_ids = set()
    input_payload_paths = []
    for out in outputs:
        combined_payload.update(out["results"])
        node_ids.extend(int(v) for v in out["node_ids"])
        root_ids.update(int(v) for v in out["root_ids"])
        input_payload_paths.append(str(out["input_payload_path"]))

    shard_path = shard_dir / f"stage1_rank{os.getpid()}_{abs(hash(tuple(sorted(node_ids))))}.pkl"
    _dump_pickle(shard_path, combined_payload)
    return {
        "stage": "stage1",
        "shard_path": str(shard_path),
        "node_ids": sorted(set(int(v) for v in node_ids)),
        "root_ids": sorted(root_ids),
        "input_payload_paths": input_payload_paths,
        "completed_count": len(items),
    }


def _worker_run_stage2(
    *,
    item: Dict,
    fof_ll: float,
    fof_vel_ll: Optional[float],
    min_stars: int,
    shard_dir: Path,
    nproc: int = 1,
) -> Dict:
    items = list(item.get("items", [item]))
    cc_backend = _env_str("CAESAR_AHF_SUBHALO_CC_BACKEND", "auto").lower()
    max_pairs_per_batch = max(1, _env_int("CAESAR_AHF_SUBHALO_MAX_PAIRS_PER_BATCH", 5_000_000))

    def run_one(one_item: Dict):
        root_id = int(one_item["root_id"])
        root_payload = _load_pickle(Path(one_item["root_payload_path"]))
        root_tasks = [_deserialize_task(task) for task in root_payload["tasks"]]
        task_payloads_by_node = {
            int(node_id): payload
            for node_id, payload in root_payload["task_payloads_by_node"].items()
        }
        initial_candidates_by_node: Dict[int, List[dict]] = {int(task.node_id): [] for task in root_tasks}
        wanted = set(initial_candidates_by_node.keys())
        for path in one_item["shard_paths"]:
            payload = _load_pickle(Path(path))
            for node_id, records in payload.items():
                node_int = int(node_id)
                if node_int not in wanted:
                    continue
                initial_candidates_by_node[node_int].extend(list(records))

        backend = "cupy" if str(one_item.get("backend", "cpu")) == "gpu" else "numpy"
        device_id = one_item.get("device_id")
        galaxies = _reconcile_root_payload(
            tasks=root_tasks,
            initial_candidates_by_node=initial_candidates_by_node,
            task_payloads_by_node=task_payloads_by_node,
            min_stars=int(min_stars),
            fof_ll=float(fof_ll),
            fof_vel_ll=fof_vel_ll,
            backend=backend,
            cc_backend=cc_backend if backend == "cupy" else "cpu",
            max_pairs_per_batch=int(max_pairs_per_batch),
            device_id=device_id if backend == "cupy" else None,
        )
        return {
            "root_id": int(root_id),
            "galaxies": _stage2_shard_payload(galaxies),
            "count": int(len(galaxies)),
        }

    max_workers = max(1, min(int(nproc), len(items)))
    if max_workers > 1:
        outputs = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one, one_item) for one_item in items]
            for fut in as_completed(futures):
                outputs.append(fut.result())
    else:
        outputs = [run_one(one_item) for one_item in items]

    galaxies_payload = []
    root_ids = []
    total_count = 0
    for out in outputs:
        galaxies_payload.extend(out["galaxies"])
        root_ids.append(int(out["root_id"]))
        total_count += int(out["count"])

    shard_path = shard_dir / f"stage2_rank{os.getpid()}_{abs(hash(tuple(sorted(root_ids))))}.pkl"
    _dump_pickle(shard_path, galaxies_payload)
    return {
        "stage": "stage2",
        "shard_path": str(shard_path),
        "root_ids": sorted(root_ids),
        "count": int(total_count),
        "completed_count": len(items),
    }


def _split_even(seq: Sequence, parts: int) -> List[List]:
    buckets = [[] for _ in range(max(1, int(parts)))]
    for idx, item in enumerate(seq):
        buckets[idx % len(buckets)].append(item)
    return buckets


def _stage3_galaxy_work_payload(groups: Sequence) -> List[dict]:
    return [_serialize_candidate_group(gal) for gal in groups]


def _stage3_halo_work_payload(halos: Sequence) -> List[int]:
    return [int(getattr(halo, "AHF_haloID", -1)) for halo in halos]


def _worker_run_stage3(
    *,
    item: Dict,
    shard_dir: Path,
    nproc: int,
) -> Dict:
    payload = _load_pickle(Path(item["payload_path"]))
    sim = _build_stage3_property_runtime(payload, nproc=int(nproc))

    _compute_group_properties_subset(sim, group_type="halo", groups=list(sim.halo_list))
    _compute_group_properties_subset(sim, group_type="galaxy", groups=list(sim.galaxy_list))

    shard_path = shard_dir / f"stage3_rank{os.getpid()}_{abs(hash(str(item.get('payload_path'))))}.pkl"
    _dump_pickle(
        shard_path,
        {
            "halos": [_serialize_group_state(group, id_key="AHF_haloID") for group in sim.halo_list],
            "galaxies": [_serialize_group_state(group, id_key="_merge_id") for group in sim.galaxy_list],
        },
    )
    return {
        "stage": "stage3",
        "shard_path": str(shard_path),
        "count_halos": int(len(sim.halo_list)),
        "count_galaxies": int(len(sim.galaxy_list)),
    }


def _rank0_build_final_sim(
    *,
    snapshot_file: str,
    ahf_particles_file: str,
    final_shards: Sequence[str],
    nproc: int,
):
    import yt
    import caesar

    ds = yt.load(snapshot_file)
    sim = caesar.CAESAR(ds)
    sim.nproc = int(nproc)
    build_halos_from_ahf_fast(sim, ahf_particles_file, compute_properties=False)
    pid_maps_sel = _build_selected_pid_maps(sim)

    final_galaxies = []
    for path in final_shards:
        payload = _load_pickle(Path(path))
        final_galaxies.extend(_deserialize_candidate_group(sim, rec) for rec in payload)

    _prepare_final_subhalo_galaxies(
        sim,
        ahf_particles_file=ahf_particles_file,
        galaxy_list=final_galaxies,
        pid_maps_sel=pid_maps_sel,
    )
    return sim


def _rank0_run_stage3_and_save(
    comm,
    *,
    worker_caps: Sequence[WorkerCapability],
    snapshot_file: str,
    ahf_particles_file: str,
    output_file: str,
    final_shards: Sequence[str],
    nproc: int,
    shard_root: Path,
):
    _rank0_log("stage3: rebuilding final reconciled catalogue")
    sim = _rank0_build_final_sim(
        snapshot_file=snapshot_file,
        ahf_particles_file=ahf_particles_file,
        final_shards=final_shards,
        nproc=int(nproc),
    )

    for idx, halo in enumerate(sim.halo_list):
        halo._merge_id = int(idx)
    for idx, gal in enumerate(sim.galaxy_list):
        gal._merge_id = int(idx)

    stage3_cpu_items = []
    halo_batches = [batch for batch in _split_even(list(sim.halo_list), len(worker_caps)) if batch]
    _rank0_log(
        f"stage3: prepared property batches count={len(halo_batches)} halos={len(sim.halo_list)} galaxies={len(sim.galaxy_list)}"
    )
    for ibatch, batch in enumerate(halo_batches):
        payload_path = shard_root / f"stage3_input_{ibatch:05d}.pkl"
        _dump_pickle(payload_path, _build_stage3_property_payload(sim, batch))
        stage3_cpu_items.append({"payload_path": str(payload_path)})

    stage3_results = _dispatch_stage(
        comm,
        worker_caps=worker_caps,
        gpu_queue=[],
        cpu_queue=stage3_cpu_items,
        stage_name="stage3",
        gpu_total=0,
        cpu_total=len(stage3_cpu_items),
    )

    halo_by_id = {int(getattr(halo, "AHF_haloID", -1)): halo for halo in sim.halo_list}
    gal_by_id = {int(getattr(gal, "_merge_id", -1)): gal for gal in sim.galaxy_list}

    for result in stage3_results:
        shard = _load_pickle(Path(result["shard_path"]))
        for state in shard.get("halos", []):
            halo = halo_by_id[int(state["id"])]
            _apply_group_state(halo, state)
        for state in shard.get("galaxies", []):
            gal = gal_by_id[int(state["id"])]
            _apply_group_state(gal, state)

    _complete_finalization_after_properties(sim)
    sim.save(output_file)
    _rank0_log(f"stage3: saved catalogue to {output_file}")


def _stage3_manifest_path(shard_root: Path) -> Path:
    return shard_root / "stage3_manifest.pkl"


def _write_stage3_manifest(
    *,
    shard_root: Path,
    snapshot_file: str,
    ahf_particles_file: str,
    output_file: str,
    final_shards: Sequence[str],
) -> Path:
    path = _stage3_manifest_path(shard_root)
    _dump_pickle(
        path,
        {
            "snapshot_file": str(snapshot_file),
            "ahf_particles_file": str(ahf_particles_file),
            "output_file": str(output_file),
            "final_shards": [str(v) for v in final_shards],
        },
    )
    return path


def _load_stage3_manifest(shard_root: Path) -> Dict[str, object]:
    path = _stage3_manifest_path(shard_root)
    if not path.is_file():
        raise RuntimeError(f"Stage-3 manifest not found: {path}")
    payload = _load_pickle(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid stage-3 manifest payload: {path}")
    return payload


def run_mpi(
    *,
    snapshot_file: str,
    ahf_particles_file: str,
    output_file: str,
    nproc: int = 1,
    role: str = "auto",
    min_stars: Optional[int] = None,
    shard_dir: Optional[str] = None,
    phase: str = "stage12",
) -> None:
    if MPI is None:
        raise RuntimeError("mpi4py is required for AHF-subhalo MPI execution")

    phase = str(phase)
    if phase not in {"stage12", "stage3"}:
        raise ValueError(f"Unsupported AHF-subhalo MPI phase: {phase}")

    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    size = int(comm.Get_size())
    if size < 2:
        raise RuntimeError("AHF-subhalo MPI requires at least 2 ranks (1 coordinator + workers)")

    role = str(role)
    if phase == "stage12":
        valid_roles = {"coordinator", "gpu_worker", "cpu_worker", "auto"}
    else:
        valid_roles = {"coordinator", "prop_worker", "auto"}
    if role not in valid_roles:
        raise ValueError(f"Unsupported AHF-subhalo MPI role for {phase}: {role}")

    declared_roles = comm.allgather(role)
    if "coordinator" in declared_roles:
        if declared_roles.count("coordinator") != 1:
            raise RuntimeError("Exactly one MPI rank must use --role coordinator")
        coordinator_rank = int(declared_roles.index("coordinator"))
    else:
        coordinator_rank = 0

    effective_role = role
    if role == "auto":
        if int(rank) == int(coordinator_rank):
            effective_role = "coordinator"
        elif phase == "stage12":
            effective_role = "gpu_worker" if _available_gpu_device_ids() else "cpu_worker"
        else:
            effective_role = "prop_worker"

    world_layout = comm.allgather(
        {
            "rank": int(rank),
            "role": str(effective_role),
            "hostname": os.uname().nodename,
            "local_rank": int(_local_rank()),
        }
    )

    shard_root = Path(shard_dir) if shard_dir else Path(tempfile.mkdtemp(prefix="ahf_subhalo_mpi_"))
    if int(rank) == int(coordinator_rank):
        shard_root.mkdir(parents=True, exist_ok=True)
    shard_root = Path(comm.bcast(str(shard_root), root=coordinator_rank))

    if phase == "stage12":
        if effective_role == "coordinator":
            _rank0_log(f"stage12: coordinator starting on shard_root={shard_root}")
            worker_ranks = [i for i in range(size) if int(i) != int(coordinator_rank)]
            worker_caps = [comm.recv(source=i, tag=MSG_REGISTER) for i in worker_ranks]
            worker_caps = [WorkerCapability(**cap) if isinstance(cap, dict) else cap for cap in worker_caps]
            gpu_worker_count = sum(1 for cap in worker_caps if cap.gpu_device is not None)
            cpu_worker_count = len(worker_caps) - gpu_worker_count
            _rank0_log(
                "stage12: workers registered "
                f"gpu={gpu_worker_count}, cpu={cpu_worker_count}, total={len(worker_caps)}"
            )

            _rank0_log("stage12: preparing runtime, task manifest, and per-node payloads")
            (
                _sim,
                _pid_maps_sel,
                min_stars_val,
                fof_ll,
                fof_vel_ll,
                tasks,
                tasks_by_root,
                task_payloads_by_node,
            ) = _rank0_prepare_stage12(
                snapshot_file=snapshot_file,
                ahf_particles_file=ahf_particles_file,
                shard_root=shard_root,
                nproc=int(nproc),
                min_stars=min_stars,
            )
            _rank0_log(
                "stage12: preparation complete "
                f"tasks={len(tasks)}, roots={len(tasks_by_root)}, min_stars={int(min_stars_val)}"
            )

            gpu_batches, cpu_batches = _classify_stage1_batches(
                tasks=tasks,
                gpu_worker_count=int(gpu_worker_count),
                cpu_worker_count=int(cpu_worker_count),
            )
            _rank0_log(
                "stage1: classified batches "
                f"gpu={len(gpu_batches)}, cpu={len(cpu_batches)}, total={len(gpu_batches) + len(cpu_batches)}"
            )
            stage1_gpu_items = _iter_stage1_batch_payloads(
                shard_root=shard_root,
                stage_batches=gpu_batches,
                prefix="gpu",
                task_payloads_by_node=task_payloads_by_node,
                fof_ll=float(fof_ll),
                fof_vel_ll=fof_vel_ll,
                min_stars=int(min_stars_val),
            )
            stage1_cpu_items = _iter_stage1_batch_payloads(
                shard_root=shard_root,
                stage_batches=cpu_batches,
                prefix="cpu",
                task_payloads_by_node=task_payloads_by_node,
                fof_ll=float(fof_ll),
                fof_vel_ll=fof_vel_ll,
                min_stars=int(min_stars_val),
            )

            stage1_results = _dispatch_stage(
                comm,
                worker_caps=worker_caps,
                gpu_queue=stage1_gpu_items,
                cpu_queue=stage1_cpu_items,
                stage_name="stage1",
                gpu_total=len(gpu_batches),
                cpu_total=len(cpu_batches),
            )

            root_to_shards: Dict[int, List[str]] = {int(root): [] for root in tasks_by_root}
            for result in stage1_results:
                for root_id in result["root_ids"]:
                    root_to_shards[int(root_id)].append(str(result["shard_path"]))
            _rank0_log(
                "stage1: result aggregation complete "
                f"roots_with_shards={sum(1 for paths in root_to_shards.values() if paths)}"
            )

            _rank0_log(f"stage2: writing root payloads for {len(tasks_by_root)} roots")
            root_payload_paths = _write_stage2_root_payloads(
                shard_root=shard_root,
                tasks_by_root=tasks_by_root,
                task_payloads_by_node=task_payloads_by_node,
            )

            gpu_root_queue, cpu_root_queue = _classify_stage2_roots(
                tasks_by_root=tasks_by_root,
                gpu_worker_count=int(gpu_worker_count),
                cpu_worker_count=int(cpu_worker_count),
            )
            _rank0_log(
                "stage2: classified roots "
                f"gpu={len(gpu_root_queue)}, cpu={len(cpu_root_queue)}, total={len(gpu_root_queue) + len(cpu_root_queue)}"
            )
            stage2_gpu_items = [
                {
                    "root_id": int(root_id),
                    "shard_paths": root_to_shards[int(root_id)],
                    "root_payload_path": root_payload_paths[int(root_id)],
                    "backend": "gpu",
                    "fof_ll": float(fof_ll),
                    "fof_vel_ll": fof_vel_ll,
                    "min_stars": int(min_stars_val),
                }
                for root_id in gpu_root_queue
            ]
            stage2_cpu_items = [
                {
                    "root_id": int(root_id),
                    "shard_paths": root_to_shards[int(root_id)],
                    "root_payload_path": root_payload_paths[int(root_id)],
                    "backend": "cpu",
                    "fof_ll": float(fof_ll),
                    "fof_vel_ll": fof_vel_ll,
                    "min_stars": int(min_stars_val),
                }
                for root_id in cpu_root_queue
            ]
            stage2_results = _dispatch_stage(
                comm,
                worker_caps=worker_caps,
                gpu_queue=stage2_gpu_items,
                cpu_queue=stage2_cpu_items,
                stage_name="stage2",
                gpu_total=len(stage2_gpu_items),
                cpu_total=len(stage2_cpu_items),
            )
            _stop_workers(comm, worker_caps=worker_caps)
            final_shards = [str(result["shard_path"]) for result in stage2_results]
            _rank0_log(f"stage2: complete; final_shards={len(final_shards)}")
            _write_stage3_manifest(
                shard_root=shard_root,
                snapshot_file=snapshot_file,
                ahf_particles_file=ahf_particles_file,
                output_file=output_file,
                final_shards=final_shards,
            )
            _rank0_log("stage12: wrote stage3 manifest")
            return

        cap = _worker_capability(
            rank,
            role=effective_role,
            threads=int(nproc),
            world_layout=world_layout,
        )
        comm.send(cap.__dict__, dest=coordinator_rank, tag=MSG_REGISTER)

        while True:
            status = MPI.Status()
            msg = comm.recv(source=coordinator_rank, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag() == MSG_STOP or msg.get("type") == "stop":
                break
            if msg.get("type") != "work":
                raise RuntimeError(f"Unexpected MPI message: {msg}")

            stage = str(msg["stage"])
            item = msg["item"]
            if stage == "stage1":
                result = _worker_run_stage1(
                    item=item,
                    fof_ll=float(item["fof_ll"]),
                    fof_vel_ll=item.get("fof_vel_ll"),
                    min_stars=int(item["min_stars"]),
                    shard_dir=shard_root,
                    nproc=int(cap.threads),
                )
            elif stage == "stage2":
                result = _worker_run_stage2(
                    item=item,
                    fof_ll=float(item["fof_ll"]),
                    fof_vel_ll=item.get("fof_vel_ll"),
                    min_stars=int(item["min_stars"]),
                    shard_dir=shard_root,
                    nproc=int(cap.threads),
                )
            else:
                raise RuntimeError(f"Unknown MPI stage: {stage}")

            comm.send(result, dest=coordinator_rank, tag=MSG_RESULT)
        return

    if effective_role == "coordinator":
        worker_ranks = [i for i in range(size) if int(i) != int(coordinator_rank)]
        worker_caps = [comm.recv(source=i, tag=MSG_REGISTER) for i in worker_ranks]
        worker_caps = [WorkerCapability(**cap) if isinstance(cap, dict) else cap for cap in worker_caps]
        manifest = _load_stage3_manifest(shard_root)
        _rank0_run_stage3_and_save(
            comm,
            worker_caps=worker_caps,
            snapshot_file=str(manifest.get("snapshot_file", snapshot_file)),
            ahf_particles_file=str(manifest.get("ahf_particles_file", ahf_particles_file)),
            output_file=str(manifest.get("output_file", output_file)),
            final_shards=[str(v) for v in manifest.get("final_shards", [])],
            nproc=int(nproc),
            shard_root=shard_root,
        )
        _stop_workers(comm, worker_caps=worker_caps)
        return

    cap = _worker_capability(
        rank,
        role=effective_role,
        threads=int(nproc),
        world_layout=world_layout,
    )
    comm.send(cap.__dict__, dest=coordinator_rank, tag=MSG_REGISTER)

    while True:
        status = MPI.Status()
        msg = comm.recv(source=coordinator_rank, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == MSG_STOP or msg.get("type") == "stop":
            break
        if msg.get("type") != "work":
            raise RuntimeError(f"Unexpected MPI message: {msg}")
        if str(msg["stage"]) != "stage3":
            raise RuntimeError(f"Unexpected property-stage MPI message: {msg}")
        result = _worker_run_stage3(item=msg["item"], shard_dir=shard_root, nproc=int(cap.threads))
        comm.send(result, dest=coordinator_rank, tag=MSG_RESULT)


def main():
    parser = argparse.ArgumentParser(description="MPI runtime for CAESAR AHF-subhalo mode")
    parser.add_argument("snapshot", type=str, help="Path to snapshot file")
    parser.add_argument("ahf", type=str, help="Path to AHF_particles file")
    parser.add_argument("out", type=str, help="Path to output CAESAR file")
    parser.add_argument(
        "--phase",
        type=str,
        default="stage12",
        choices=("stage12", "stage3"),
        help="MPI phase to run: non-uniform stage12 or uniform property stage3",
    )
    parser.add_argument(
        "--role",
        type=str,
        default="auto",
        choices=("auto", "coordinator", "gpu_worker", "cpu_worker", "prop_worker"),
        help="MPI rank role; stage12 uses coordinator/gpu_worker/cpu_worker, stage3 uses coordinator/prop_worker",
    )
    parser.add_argument("--nproc", type=int, default=1, help="Per-rank CAESAR nproc/thread budget")
    parser.add_argument("--min-stars", type=int, default=None, help="Minimum stars per galaxy")
    parser.add_argument("--shard-dir", type=str, default=None, help="Directory for intermediate shard files")
    args = parser.parse_args()

    run_mpi(
        snapshot_file=args.snapshot,
        ahf_particles_file=args.ahf,
        output_file=args.out,
        nproc=int(args.nproc),
        role=str(args.role),
        min_stars=args.min_stars,
        shard_dir=args.shard_dir,
        phase=str(args.phase),
    )


if __name__ == "__main__":
    main()
