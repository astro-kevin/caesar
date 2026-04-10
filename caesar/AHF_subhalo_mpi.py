from __future__ import annotations

import argparse
import os
import pickle
import tempfile
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
    _assign_batches_to_resources,
    _available_gpu_device_ids,
    _build_task_manifest,
    _build_tiny_batches,
    _complete_finalization_after_properties,
    _compute_group_properties_subset,
    _deserialize_candidate_group,
    _env_float,
    _env_int,
    _env_str,
    _prepare_final_subhalo_galaxies,
    _fof_for_batch,
    _reconcile_root,
    _serialize_group_state,
    _serialize_candidate_group,
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


def _local_rank() -> int:
    for key in ("SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"):
        raw = os.environ.get(key)
        if raw not in (None, ""):
            try:
                return int(raw)
            except Exception:
                pass
    return 0


def _worker_capability(rank: int, *, role: str, threads: int, comm) -> WorkerCapability:
    hostname = os.uname().nodename
    local_rank = _local_rank()
    gpu_device = None
    if str(role) == "gpu_worker":
        devices = _available_gpu_device_ids()
        try:
            node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, key=rank)
            gpu_index = int(node_comm.Get_rank())
        except Exception:
            gpu_index = int(local_rank)
        if 0 <= gpu_index < len(devices):
            gpu_device = int(devices[gpu_index])
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
    return {
        int(node_id): [_serialize_candidate_group(gal) for gal in gals]
        for node_id, gals in results_by_node.items()
    }


def _stage2_shard_payload(galaxies: Sequence) -> List[dict]:
    return [_serialize_candidate_group(gal) for gal in galaxies]


def _prepare_manifest(
    *,
    ahf_particles_file: str,
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


def _next_item(queue: List, idx: int):
    if idx >= len(queue):
        return None, idx
    return queue[idx], idx + 1


def _dispatch_stage(
    comm,
    *,
    worker_caps: Sequence[WorkerCapability],
    gpu_queue: List,
    cpu_queue: List,
    stage_name: str,
):
    gpu_idx = 0
    cpu_idx = 0
    active = 0
    results = []

    def assign_next(rank: int, cap: WorkerCapability):
        nonlocal gpu_idx, cpu_idx, active
        item = None
        if cap.role == "gpu_worker" and cap.gpu_device is not None and gpu_idx < len(gpu_queue):
            item, gpu_idx = _next_item(gpu_queue, gpu_idx)
        elif cap.role == "cpu_worker" and cpu_idx < len(cpu_queue):
            item, cpu_idx = _next_item(cpu_queue, cpu_idx)
        elif cap.role == "gpu_worker" and gpu_idx < len(gpu_queue):
            item, gpu_idx = _next_item(gpu_queue, gpu_idx)
        elif cap.role == "cpu_worker" and cpu_idx < len(cpu_queue):
            item, cpu_idx = _next_item(cpu_queue, cpu_idx)

        if item is None:
            comm.send({"type": "stop", "stage": stage_name}, dest=rank, tag=MSG_STOP)
            return False

        payload = {"type": "work", "stage": stage_name, "item": item}
        comm.send(payload, dest=rank, tag=MSG_WORK)
        active += 1
        return True

    for cap in worker_caps:
        assign_next(cap.rank, cap)

    while active > 0:
        status = MPI.Status()
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MSG_RESULT, status=status)
        src = int(status.Get_source())
        active -= 1
        results.append(msg)
        cap = next(cap for cap in worker_caps if int(cap.rank) == int(src))
        assign_next(src, cap)

    return results


def _worker_init_runtime(snapshot_file: str, ahf_particles_file: str, *, nproc: int):
    import yt
    import caesar

    ds = yt.load(snapshot_file)
    sim = caesar.CAESAR(ds)
    sim.nproc = int(nproc)
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


def _worker_run_stage1(
    *,
    sim,
    batch: AHFSubhaloBatch,
    ahf_particles_file: str,
    pid_maps_sel,
    fof_ll: float,
    fof_vel_ll: Optional[float],
    min_stars: int,
    shard_dir: Path,
) -> Dict:
    membership_arrays = getattr(sim, "_ahf_fast_memberships")
    fof_nHlim = _env_float("CAESAR_AHF_FAST_FOF_NHLIM", 0.13)
    fof_Tlim = _env_float("CAESAR_AHF_FAST_FOF_TLIM", 1.0e5)
    fof_use_sfr_gate = os.environ.get("CAESAR_AHF_FAST_FOF_USE_SFR", "1") == "1"
    cc_backend = _env_str("CAESAR_AHF_SUBHALO_CC_BACKEND", "auto").lower()
    max_pairs_per_batch = max(1, _env_int("CAESAR_AHF_SUBHALO_MAX_PAIRS_PER_BATCH", 5_000_000))

    results = _fof_for_batch(
        sim,
        batch=batch,
        membership_arrays=membership_arrays,
        pid_maps_sel=pid_maps_sel,
        min_stars=int(min_stars),
        fof_ll=float(fof_ll),
        fof_vel_ll=fof_vel_ll,
        fof_nHlim=float(fof_nHlim),
        fof_Tlim=float(fof_Tlim),
        fof_use_sfr_gate=bool(fof_use_sfr_gate),
        backend="cupy" if batch.target_backend == "gpu" else "numpy",
        cc_backend=cc_backend if batch.target_backend == "gpu" else "cpu",
        max_pairs_per_batch=int(max_pairs_per_batch),
        device_id=batch.target_device,
    )

    shard_path = shard_dir / f"stage1_rank{os.getpid()}_{abs(hash(tuple(int(t.node_id) for t in batch.tasks)))}.pkl"
    _dump_pickle(shard_path, _stage1_shard_payload(results))
    return {
        "stage": "stage1",
        "shard_path": str(shard_path),
        "node_ids": [int(t.node_id) for t in batch.tasks],
        "root_ids": sorted({int(t.top_id) for t in batch.tasks}),
    }


def _worker_run_stage2(
    *,
    sim,
    root_id: int,
    root_tasks: Sequence[AHFSubhaloTask],
    shard_paths: Sequence[str],
    pid_maps_sel,
    fof_ll: float,
    fof_vel_ll: Optional[float],
    min_stars: int,
    shard_dir: Path,
    backend: str,
    device_id: Optional[int],
) -> Dict:
    membership_arrays = getattr(sim, "_ahf_fast_memberships")
    fof_nHlim = _env_float("CAESAR_AHF_FAST_FOF_NHLIM", 0.13)
    fof_Tlim = _env_float("CAESAR_AHF_FAST_FOF_TLIM", 1.0e5)
    fof_use_sfr_gate = os.environ.get("CAESAR_AHF_FAST_FOF_USE_SFR", "1") == "1"
    cc_backend = _env_str("CAESAR_AHF_SUBHALO_CC_BACKEND", "auto").lower()
    max_pairs_per_batch = max(1, _env_int("CAESAR_AHF_SUBHALO_MAX_PAIRS_PER_BATCH", 5_000_000))

    initial_candidates_by_node: Dict[int, List] = {int(task.node_id): [] for task in root_tasks}
    wanted = set(initial_candidates_by_node.keys())
    for path in shard_paths:
        payload = _load_pickle(Path(path))
        for node_id, records in payload.items():
            node_int = int(node_id)
            if node_int not in wanted:
                continue
            initial_candidates_by_node[node_int].extend(
                _deserialize_candidate_group(sim, rec) for rec in records
            )

    galaxies = _reconcile_root(
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
        cc_backend=cc_backend if backend == "cupy" else "cpu",
        max_pairs_per_batch=int(max_pairs_per_batch),
    )
    shard_path = shard_dir / f"stage2_rank{os.getpid()}_root{int(root_id)}.pkl"
    _dump_pickle(shard_path, _stage2_shard_payload(galaxies))
    return {
        "stage": "stage2",
        "shard_path": str(shard_path),
        "root_id": int(root_id),
        "count": int(len(galaxies)),
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
    sim,
    item: Dict,
    shard_dir: Path,
) -> Dict:
    group_type = str(item["group_type"])
    if group_type == "halo":
        wanted = set(int(v) for v in item["halo_ids"])
        groups = [halo for halo in sim.halo_list if int(getattr(halo, "AHF_haloID", -1)) in wanted]
        _compute_group_properties_subset(sim, group_type="halo", groups=groups)
        payload = [_serialize_group_state(group, id_key="AHF_haloID") for group in groups]
    elif group_type == "galaxy":
        groups = [_deserialize_candidate_group(sim, rec) for rec in item["galaxies"]]
        _compute_group_properties_subset(sim, group_type="galaxy", groups=groups)
        payload = [_serialize_group_state(group, id_key="_merge_id") for group in groups]
    else:
        raise RuntimeError(f"Unknown stage3 group_type: {group_type}")

    shard_path = shard_dir / f"stage3_rank{os.getpid()}_{group_type}_{abs(hash(str(item)[:128]))}.pkl"
    _dump_pickle(shard_path, {"group_type": group_type, "payload": payload})
    return {"stage": "stage3", "group_type": group_type, "shard_path": str(shard_path), "count": int(len(payload))}


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
):
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

    galaxy_workers = max(1, len(worker_caps))
    halo_batches = [batch for batch in _split_even(list(sim.halo_list), len(worker_caps)) if batch]
    galaxy_batches = [batch for batch in _split_even(list(sim.galaxy_list), galaxy_workers) if batch]

    stage3_cpu_items = []
    for batch in halo_batches:
        stage3_cpu_items.append({"group_type": "halo", "halo_ids": _stage3_halo_work_payload(batch)})
    for batch in galaxy_batches:
        stage3_cpu_items.append({"group_type": "galaxy", "galaxies": _stage3_galaxy_work_payload(batch)})

    stage3_results = _dispatch_stage(
        comm,
        worker_caps=worker_caps,
        gpu_queue=[],
        cpu_queue=stage3_cpu_items,
        stage_name="stage3",
    )

    halo_by_id = {int(getattr(halo, "AHF_haloID", -1)): halo for halo in sim.halo_list}
    gal_by_id = {int(getattr(gal, "_merge_id", -1)): gal for gal in sim.galaxy_list}

    for result in stage3_results:
        shard = _load_pickle(Path(result["shard_path"]))
        group_type = str(shard["group_type"])
        payload = shard["payload"]
        if group_type == "halo":
            for state in payload:
                halo = halo_by_id[int(state["id"])]
                _apply_group_state(halo, state)
        elif group_type == "galaxy":
            for state in payload:
                gal = gal_by_id[int(state["id"])]
                _apply_group_state(gal, state)
        else:
            raise RuntimeError(f"Unexpected stage3 shard group_type: {group_type}")

    _complete_finalization_after_properties(sim)
    sim.save(output_file)


def run_mpi(
    *,
    snapshot_file: str,
    ahf_particles_file: str,
    output_file: str,
    nproc: int = 1,
    role: str = "auto",
    min_stars: Optional[int] = None,
    shard_dir: Optional[str] = None,
) -> None:
    if MPI is None:
        raise RuntimeError("mpi4py is required for AHF-subhalo MPI execution")

    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    size = int(comm.Get_size())
    if size < 2:
        raise RuntimeError("AHF-subhalo MPI requires at least 2 ranks (1 coordinator + workers)")

    role = str(role)
    if role not in {"coordinator", "gpu_worker", "cpu_worker", "auto"}:
        raise ValueError(f"Unsupported AHF-subhalo MPI role: {role}")

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
        else:
            effective_role = "gpu_worker" if _available_gpu_device_ids() else "cpu_worker"

    shard_root = Path(shard_dir) if shard_dir else Path(tempfile.mkdtemp(prefix="ahf_subhalo_mpi_"))
    if int(rank) == int(coordinator_rank):
        shard_root.mkdir(parents=True, exist_ok=True)
    shard_root = Path(comm.bcast(str(shard_root), root=coordinator_rank))

    if effective_role == "coordinator":
        if min_stars is None:
            min_stars_val = 16
        else:
            min_stars_val = int(min_stars)
        tasks, tasks_by_root, _, _ = _prepare_manifest(
            ahf_particles_file=ahf_particles_file,
            min_stars=int(min_stars_val),
        )
        worker_ranks = [i for i in range(size) if int(i) != int(coordinator_rank)]
        worker_caps = [comm.recv(source=i, tag=MSG_REGISTER) for i in worker_ranks]
        worker_caps = [WorkerCapability(**cap) if isinstance(cap, dict) else cap for cap in worker_caps]

        gpu_worker_count = sum(1 for cap in worker_caps if cap.gpu_device is not None)
        cpu_worker_count = len(worker_caps) - gpu_worker_count
        gpu_queue, cpu_queue = _classify_stage1_batches(
            tasks=tasks,
            gpu_worker_count=int(gpu_worker_count),
            cpu_worker_count=int(cpu_worker_count),
        )
        stage1_results = _dispatch_stage(
            comm,
            worker_caps=worker_caps,
            gpu_queue=list(gpu_queue),
            cpu_queue=list(cpu_queue),
            stage_name="stage1",
        )

        root_to_shards: Dict[int, List[str]] = {int(root): [] for root in tasks_by_root}
        for result in stage1_results:
            for root_id in result["root_ids"]:
                root_to_shards[int(root_id)].append(str(result["shard_path"]))

        gpu_root_queue, cpu_root_queue = _classify_stage2_roots(
            tasks_by_root=tasks_by_root,
            gpu_worker_count=int(gpu_worker_count),
            cpu_worker_count=int(cpu_worker_count),
        )
        stage2_gpu_items = [
            {"root_id": int(root_id), "shard_paths": root_to_shards[int(root_id)], "backend": "gpu"}
            for root_id in gpu_root_queue
        ]
        stage2_cpu_items = [
            {"root_id": int(root_id), "shard_paths": root_to_shards[int(root_id)], "backend": "cpu"}
            for root_id in cpu_root_queue
        ]
        stage2_results = _dispatch_stage(
            comm,
            worker_caps=worker_caps,
            gpu_queue=stage2_gpu_items,
            cpu_queue=stage2_cpu_items,
            stage_name="stage2",
        )
        final_shards = [result["shard_path"] for result in stage2_results]
        _rank0_run_stage3_and_save(
            comm,
            worker_caps=worker_caps,
            snapshot_file=snapshot_file,
            ahf_particles_file=ahf_particles_file,
            output_file=output_file,
            final_shards=final_shards,
            nproc=int(nproc),
        )
        return

    cap = _worker_capability(rank, role=effective_role, threads=int(nproc), comm=comm)
    comm.send(cap.__dict__, dest=coordinator_rank, tag=MSG_REGISTER)

    sim, pid_maps_sel, fof_ll, fof_vel_ll = _worker_init_runtime(
        snapshot_file,
        ahf_particles_file,
        nproc=int(cap.threads),
    )
    ms = get_min_stars(sim, override=min_stars)
    tasks_by_root = None
    manifest_cache = None

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
            batch = item
            if isinstance(batch, dict):
                batch = AHFSubhaloBatch(
                    tasks=tuple(AHFSubhaloTask(**task) if isinstance(task, dict) else task for task in batch["tasks"]),
                    is_tiny_batch=bool(batch.get("is_tiny_batch", False)),
                    target_backend=str(batch.get("target_backend", "cpu")),
                    target_device=batch.get("target_device"),
                    estimated_cost=int(batch.get("estimated_cost", 0)),
                )
            result = _worker_run_stage1(
                sim=sim,
                batch=batch,
                ahf_particles_file=ahf_particles_file,
                pid_maps_sel=pid_maps_sel,
                fof_ll=float(fof_ll),
                fof_vel_ll=fof_vel_ll,
                min_stars=int(ms),
                shard_dir=shard_root,
            )
        elif stage == "stage2":
            if tasks_by_root is None:
                manifest_cache = _prepare_manifest(ahf_particles_file=ahf_particles_file, min_stars=int(ms))
                _, tasks_by_root, _, _ = manifest_cache
            root_id = int(item["root_id"])
            backend = "cupy" if str(item.get("backend", "cpu")) == "gpu" and cap.gpu_device is not None else "numpy"
            result = _worker_run_stage2(
                sim=sim,
                root_id=int(root_id),
                root_tasks=tasks_by_root[int(root_id)],
                shard_paths=item["shard_paths"],
                pid_maps_sel=pid_maps_sel,
                fof_ll=float(fof_ll),
                fof_vel_ll=fof_vel_ll,
                min_stars=int(ms),
                shard_dir=shard_root,
                backend=backend,
                device_id=cap.gpu_device if backend == "cupy" else None,
            )
        elif stage == "stage3":
            result = _worker_run_stage3(
                sim=sim,
                item=item,
                shard_dir=shard_root,
            )
        else:
            raise RuntimeError(f"Unknown MPI stage: {stage}")

        comm.send(result, dest=coordinator_rank, tag=MSG_RESULT)


def main():
    parser = argparse.ArgumentParser(description="MPI runtime for CAESAR AHF-subhalo mode")
    parser.add_argument("snapshot", type=str, help="Path to snapshot file")
    parser.add_argument("ahf", type=str, help="Path to AHF_particles file")
    parser.add_argument("out", type=str, help="Path to output CAESAR file")
    parser.add_argument(
        "--role",
        type=str,
        default="auto",
        choices=("auto", "coordinator", "gpu_worker", "cpu_worker"),
        help="MPI rank role; use explicit roles with MPMD launch for non-uniform worker layouts",
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
    )


if __name__ == "__main__":
    main()
