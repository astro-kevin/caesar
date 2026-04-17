from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
import gc
import os
import pickle
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    _tqdm = None


try:
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover
    MPI = None


from caesar.AHF_FAST_halos import build_halos_from_ahf_fast
from caesar.AHF_FAST_loader import load_ahf_halos_dataframe, load_ahf_hierarchy
from caesar.AHF_subhalo import (
    AHFSubhaloBatch,
    AHFSubhaloTask,
    MINIMUM_DM_PER_AHF_SUBHALO,
    MINIMUM_DM_PER_TOPLEVEL_AHF_HALO,
    _available_gpu_device_ids,
    _build_node_dm_counts,
    _build_calculating_properties_payload_from_particle_store,
    _build_calculating_properties_runtime,
    _build_task_input_payload,
    _build_task_manifest,
    _build_tiny_batches,
    _complete_finalization_after_properties,
    _complete_finalization_after_properties_direct,
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
from caesar.ahf_subhalo_hdf5 import (
    AHFSubhaloDirectState,
    build_direct_state,
    build_halo_record as _build_store_halo_record,
    build_store_for_roots,
    load_node_store,
    load_particle_store,
    load_snapshot_meta_store,
    write_node_store,
    write_particle_store,
    write_snapshot_meta_store,
    build_task_payload as _build_direct_task_payload,
)
from caesar.ahf_subhalo_tables import (
    candidate_records_from_table_payload,
    candidate_records_to_table,
)
from caesar.ahf_subhalo_export import (
    build_group_column_payload,
    compute_global_properties_from_property_shards,
    write_catalogue_from_property_shards,
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
    visible_cores: int = 0


@dataclass(frozen=True)
class ProgressMetric:
    label: str
    result_key: str
    total: Optional[int] = None


@dataclass
class _StageQueue:
    items: List[object]
    total: int
    front_index: int = 0
    back_index: int = -1
    front_assigned: int = 0
    back_assigned: int = 0

    def __post_init__(self):
        self.items = list(self.items)
        if int(self.total) <= 0:
            self.total = len(self.items)
        self.back_index = len(self.items) - 1

    def next_front(self):
        if self.front_index > self.back_index:
            return None
        item = self.items[self.front_index]
        self.front_index += 1
        self.front_assigned += 1
        return item

    def next_back(self):
        if self.front_index > self.back_index:
            return None
        item = self.items[self.back_index]
        self.back_index -= 1
        self.back_assigned += 1
        return item

    @property
    def remaining(self) -> int:
        return max(0, int(self.back_index) - int(self.front_index) + 1)

    @property
    def dispatched(self) -> int:
        return int(self.front_assigned) + int(self.back_assigned)


_RANK0_LOG_TOTAL_START: Optional[float] = None
_RANK0_LOG_STAGE_STARTS: Dict[str, float] = {}


@dataclass
class _BufferedPreparedQueue:
    specs: List[object]
    prepare_fn: object
    total: int
    front_target: int = 0
    back_target: int = 0
    max_workers: int = 2
    front_index: int = 0
    back_index: int = -1
    front_assigned: int = 0
    back_assigned: int = 0
    ready_front: deque = field(default_factory=deque)
    ready_back: deque = field(default_factory=deque)
    _pending_front: List[object] = field(default_factory=list)
    _pending_back: List[object] = field(default_factory=list)
    _executor: Optional[ThreadPoolExecutor] = None

    def __post_init__(self):
        self.specs = list(self.specs)
        if int(self.total) <= 0:
            self.total = len(self.specs)
        self.back_index = len(self.specs) - 1
        self.front_target = max(0, int(self.front_target))
        self.back_target = max(0, int(self.back_target))
        self.max_workers = max(1, int(self.max_workers))
        if len(self.specs) > 0 and (self.front_target > 0 or self.back_target > 0):
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
            self._top_up()
            self._wait_until_ready()

    @property
    def remaining(self) -> int:
        return max(0, int(self.back_index) - int(self.front_index) + 1)

    @property
    def dispatched(self) -> int:
        return int(self.front_assigned) + int(self.back_assigned)

    def _drain_completed(self) -> None:
        for pending, ready in (
            (self._pending_front, self.ready_front),
            (self._pending_back, self.ready_back),
        ):
            keep = []
            for fut in pending:
                if fut.done():
                    ready.append(fut.result())
                else:
                    keep.append(fut)
            pending[:] = keep

    def _submit_front(self) -> bool:
        if self._executor is None or self.front_index > self.back_index:
            return False
        spec = self.specs[self.front_index]
        self.front_index += 1
        self._pending_front.append(self._executor.submit(self.prepare_fn, spec, "front"))
        return True

    def _submit_back(self) -> bool:
        if self._executor is None or self.front_index > self.back_index:
            return False
        spec = self.specs[self.back_index]
        self.back_index -= 1
        self._pending_back.append(self._executor.submit(self.prepare_fn, spec, "back"))
        return True

    def _top_up(self) -> None:
        self._drain_completed()
        while (
            self.front_index <= self.back_index
            and len(self.ready_front) + len(self._pending_front) < self.front_target
        ):
            if not self._submit_front():
                break
        while (
            self.front_index <= self.back_index
            and len(self.ready_back) + len(self._pending_back) < self.back_target
        ):
            if not self._submit_back():
                break
        self._drain_completed()

    def _wait_until_ready(self) -> None:
        while True:
            front_need = self.front_target > 0 and len(self.ready_front) == 0 and len(self._pending_front) > 0
            back_need = self.back_target > 0 and len(self.ready_back) == 0 and len(self._pending_back) > 0
            if not (front_need or back_need):
                break
            futures = list(self._pending_front) + list(self._pending_back)
            if not futures:
                break
            wait(futures, return_when=FIRST_COMPLETED)
            self._drain_completed()
            self._top_up()

    def next_front(self):
        self._drain_completed()
        self._top_up()
        if not self.ready_front and self._pending_front:
            self._wait_until_ready()
        if self.ready_front:
            self.front_assigned += 1
            item = self.ready_front.popleft()
            self._top_up()
            return item
        return None

    def next_back(self):
        self._drain_completed()
        self._top_up()
        if not self.ready_back and self._pending_back:
            self._wait_until_ready()
        if self.ready_back:
            self.back_assigned += 1
            item = self.ready_back.popleft()
            self._top_up()
            return item
        return None

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


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


def _parse_first_int(raw: Optional[str]) -> Optional[int]:
    if raw in (None, ""):
        return None
    try:
        return int(str(raw).split(",")[0].split("(")[0].strip())
    except Exception:
        return None


def _local_visible_cores() -> int:
    for key in ("SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE"):
        parsed = _parse_first_int(os.environ.get(key))
        if parsed is not None and parsed > 0:
            return int(parsed)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


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
        visible_cores=int(_local_visible_cores()),
    )


def _distribute_even_threads(ranks: Sequence[int], total_threads: int) -> Dict[int, int]:
    ordered = [int(rank) for rank in ranks]
    if not ordered:
        return {}
    total_threads = max(int(total_threads), len(ordered))
    base = int(total_threads) // len(ordered)
    remainder = int(total_threads) % len(ordered)
    out: Dict[int, int] = {}
    for idx, rank in enumerate(sorted(ordered)):
        out[int(rank)] = int(base + (1 if idx < remainder else 0))
    return out


def _host_core_budgets(
    *,
    world_layout: Sequence[Dict[str, object]],
    coordinator_rank: int,
    coordinator_reserve: int = 1,
) -> Dict[str, int]:
    host_cores: Dict[str, int] = {}
    for entry in world_layout:
        host = str(entry["hostname"])
        host_cores[host] = max(host_cores.get(host, 0), int(entry.get("visible_cores", 0) or 0))

    coord_host = next(
        (str(entry["hostname"]) for entry in world_layout if int(entry["rank"]) == int(coordinator_rank)),
        None,
    )
    if coord_host is not None:
        host_cores[coord_host] = max(1, int(host_cores.get(coord_host, 1)) - int(max(0, coordinator_reserve)))
    return host_cores


def _build_finding_galaxies_thread_map(
    *,
    worker_caps: Sequence[WorkerCapability],
    world_layout: Sequence[Dict[str, object]],
    coordinator_rank: int,
) -> Dict[int, int]:
    host_budgets = _host_core_budgets(world_layout=world_layout, coordinator_rank=coordinator_rank)
    out: Dict[int, int] = {}
    caps_by_host: Dict[str, List[WorkerCapability]] = {}
    for cap in worker_caps:
        caps_by_host.setdefault(str(cap.hostname), []).append(cap)

    for host, caps in caps_by_host.items():
        gpu_caps = [cap for cap in caps if cap.gpu_device is not None]
        cpu_caps = [cap for cap in caps if cap.gpu_device is None]

        gpu_threads = 0
        for cap in gpu_caps:
            nthreads = max(1, int(cap.threads))
            out[int(cap.rank)] = nthreads
            gpu_threads += nthreads

        remaining = max(0, int(host_budgets.get(host, len(cpu_caps))) - gpu_threads)
        if cpu_caps:
            if remaining < len(cpu_caps):
                remaining = len(cpu_caps)
            out.update(_distribute_even_threads([cap.rank for cap in cpu_caps], remaining))
    return out


def _build_uniform_stage_thread_map(
    *,
    worker_caps: Sequence[WorkerCapability],
    world_layout: Sequence[Dict[str, object]],
    coordinator_rank: int,
) -> Dict[int, int]:
    host_budgets = _host_core_budgets(world_layout=world_layout, coordinator_rank=coordinator_rank)
    out: Dict[int, int] = {}
    caps_by_host: Dict[str, List[WorkerCapability]] = {}
    for cap in worker_caps:
        caps_by_host.setdefault(str(cap.hostname), []).append(cap)

    for host, caps in caps_by_host.items():
        budget = max(len(caps), int(host_budgets.get(host, len(caps))))
        out.update(_distribute_even_threads([cap.rank for cap in caps], budget))
    return out


def _log_thread_map(stage_name: str, worker_caps: Sequence[WorkerCapability], thread_map: Dict[int, int]) -> None:
    if not thread_map:
        return
    threads = [int(thread_map.get(int(cap.rank), 1)) for cap in worker_caps]
    gpu_threads = [int(thread_map.get(int(cap.rank), 1)) for cap in worker_caps if cap.gpu_device is not None]
    cpu_threads = [int(thread_map.get(int(cap.rank), 1)) for cap in worker_caps if cap.gpu_device is None]
    parts = [f"workers={len(threads)}", f"threads_total={sum(threads)}", f"min={min(threads)}", f"max={max(threads)}"]
    if gpu_threads:
        parts.append(f"gpu_threads_total={sum(gpu_threads)}")
    if cpu_threads:
        parts.append(f"cpu_threads_total={sum(cpu_threads)}")
    _rank0_log(f"{stage_name}: thread budget " + ", ".join(parts))


def _eligible_halo_node_ids_by_root(state: AHFSubhaloDirectState) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for node_index in range(len(state.nodes)):
        halo_id = int(state.nodes.halo_id[node_index])
        parent_id = int(state.nodes.parent_halo_id[node_index])
        top_id = int(state.nodes.top_halo_id[node_index])
        dm_count = int(state.nodes.dm_count[node_index])
        min_dm = int(MINIMUM_DM_PER_TOPLEVEL_AHF_HALO) if parent_id <= 0 else int(MINIMUM_DM_PER_AHF_SUBHALO)
        if dm_count < min_dm and halo_id != top_id:
            continue
        out.setdefault(int(top_id), []).append(int(halo_id))
    return out


def _build_halo_record_from_state(state: AHFSubhaloDirectState, *, node_index: int) -> Dict[str, object]:
    return _build_store_halo_record(state, node_id=int(state.nodes.halo_id[node_index]))


def _group_record_member_cost(record: Mapping[str, object]) -> int:
    total = 0
    for name in ("glist", "slist", "dmlist", "bhlist", "dlist"):
        total += int(np.asarray(record.get(name, []), dtype=np.int64).size)
    return int(max(1, total))


def _dump_pickle(path: Path, payload) -> None:
    with path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def _snapshot_meta_store_path(shard_root: Path) -> Path:
    return Path(shard_root) / "snapshot_meta_store.pkl"


def _store_manifest_path(shard_root: Path) -> Path:
    return Path(shard_root) / "store_manifest.pkl"


def _particle_store_path(shard_root: Path, *, store_id: str, kind: str) -> Path:
    safe_id = str(store_id)
    return Path(shard_root) / f"particle_store_{str(kind)}{safe_id}.h5"


def _node_store_path(shard_root: Path, *, store_id: str, kind: str) -> Path:
    safe_id = str(store_id)
    return Path(shard_root) / f"node_store_{str(kind)}{safe_id}.pkl"


def _write_store_manifest(shard_root: Path, payload: Mapping[str, object]) -> Path:
    path = _store_manifest_path(shard_root)
    _dump_pickle(path, dict(payload))
    return path

def _galaxy_finding_shard_payload(results_by_node: Dict[int, List]) -> Dict[int, List[dict]]:
    out = {}
    for node_id, gals in results_by_node.items():
        out[int(node_id)] = [
            gal if isinstance(gal, dict) else _serialize_candidate_group(gal)
            for gal in gals
        ]
    return out


def _reconciled_galaxy_shard_payload(galaxies: Sequence) -> List[dict]:
    records = [gal if isinstance(gal, dict) else _serialize_candidate_group(gal) for gal in galaxies]
    return candidate_records_to_table(records).to_payload()


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


def _root_store_summaries(
    state: AHFSubhaloDirectState,
    *,
    tasks_by_root: Mapping[int, Sequence[AHFSubhaloTask]],
) -> Dict[int, Dict[str, object]]:
    root_node_indexes: Dict[int, List[int]] = {}
    for node_index in range(len(state.nodes)):
        root_id = int(state.nodes.top_halo_id[node_index])
        if int(root_id) not in tasks_by_root:
            continue
        root_node_indexes.setdefault(int(root_id), []).append(int(node_index))

    summaries: Dict[int, Dict[str, object]] = {}
    for root_id, node_indexes in root_node_indexes.items():
        particle_rows_by_ptype: Dict[str, int] = {}
        particle_row_total = 0
        for ptype in state.particles.ptypes:
            rows = [
                np.asarray(state.nodes.members_for(int(node_index), str(ptype), dtype=np.int64), dtype=np.int64)
                for node_index in node_indexes
            ]
            merged = np.unique(np.concatenate(rows).astype(np.int64, copy=False)) if rows else np.empty(0, dtype=np.int64)
            particle_rows_by_ptype[str(ptype)] = int(merged.size)
            particle_row_total += int(merged.size)

        summaries[int(root_id)] = {
            "root_id": int(root_id),
            "node_count": int(len(node_indexes)),
            "task_count": int(len(tasks_by_root.get(int(root_id), ()))),
            "particle_row_total": int(particle_row_total),
            "particle_rows_by_ptype": particle_rows_by_ptype,
            "fof_candidates": int(sum(int(task.fof_candidates) for task in tasks_by_root.get(int(root_id), ()))),
        }
    return summaries


def _bucket_root_stores(
    root_summaries: Mapping[int, Mapping[str, object]],
) -> List[Dict[str, object]]:
    if not root_summaries:
        return []

    max_roots_per_bucket = max(1, _env_int("CAESAR_AHF_SUBHALO_STORE_MAX_ROOTS_PER_BUCKET", 128))
    max_particle_rows = max(1, _env_int("CAESAR_AHF_SUBHALO_STORE_MAX_PARTICLE_ROWS", 1_000_000))
    max_nodes = max(1, _env_int("CAESAR_AHF_SUBHALO_STORE_MAX_NODES", 1024))
    max_fof_candidates = max(1, _env_int("CAESAR_AHF_SUBHALO_STORE_MAX_FOF_CANDIDATES", 500_000))

    ordered = [
        dict(root_summaries[int(root_id)])
        for root_id in sorted(
            root_summaries.keys(),
            key=lambda rid: (
                int(root_summaries[int(rid)].get("fof_candidates", 0)),
                int(root_summaries[int(rid)].get("particle_row_total", 0)),
                int(root_summaries[int(rid)].get("node_count", 0)),
                int(rid),
            ),
        )
    ]

    buckets: List[Dict[str, object]] = []
    current: Dict[str, object] = {
        "root_ids": [],
        "node_count": 0,
        "particle_row_total": 0,
        "fof_candidates": 0,
    }

    def _flush_current() -> None:
        nonlocal current
        if current["root_ids"]:
            buckets.append(
                {
                    "root_ids": tuple(int(v) for v in current["root_ids"]),
                    "node_count": int(current["node_count"]),
                    "particle_row_total": int(current["particle_row_total"]),
                    "fof_candidates": int(current["fof_candidates"]),
                }
            )
        current = {
            "root_ids": [],
            "node_count": 0,
            "particle_row_total": 0,
            "fof_candidates": 0,
        }

    for rec in ordered:
        root_id = int(rec["root_id"])
        add_roots = int(len(current["root_ids"])) + 1
        add_nodes = int(current["node_count"]) + int(rec.get("node_count", 0))
        add_rows = int(current["particle_row_total"]) + int(rec.get("particle_row_total", 0))
        add_candidates = int(current["fof_candidates"]) + int(rec.get("fof_candidates", 0))
        if current["root_ids"] and (
            add_roots > int(max_roots_per_bucket)
            or add_nodes > int(max_nodes)
            or add_rows > int(max_particle_rows)
            or add_candidates > int(max_fof_candidates)
        ):
            _flush_current()
        current["root_ids"].append(int(root_id))
        current["node_count"] = int(current["node_count"]) + int(rec.get("node_count", 0))
        current["particle_row_total"] = int(current["particle_row_total"]) + int(rec.get("particle_row_total", 0))
        current["fof_candidates"] = int(current["fof_candidates"]) + int(rec.get("fof_candidates", 0))
    _flush_current()

    out: List[Dict[str, object]] = []
    bucket_index = 0
    for bucket in buckets:
        root_ids = tuple(int(v) for v in bucket["root_ids"])
        if len(root_ids) == 1:
            root_id = int(root_ids[0])
            out.append(
                {
                    "store_id": str(root_id),
                    "kind": "root",
                    "root_ids": root_ids,
                    "node_count": int(bucket["node_count"]),
                    "particle_row_total": int(bucket["particle_row_total"]),
                    "fof_candidates": int(bucket["fof_candidates"]),
                }
            )
        else:
            bucket_index += 1
            out.append(
                {
                    "store_id": f"{bucket_index:06d}",
                    "kind": "bucket",
                    "root_ids": root_ids,
                    "node_count": int(bucket["node_count"]),
                    "particle_row_total": int(bucket["particle_row_total"]),
                    "fof_candidates": int(bucket["fof_candidates"]),
                }
            )
    return out


def _build_store_artifacts(
    *,
    state: AHFSubhaloDirectState,
    tasks_by_root: Mapping[int, Sequence[AHFSubhaloTask]],
    shard_root: Path,
    log_fn=None,
    log_label: str = "finding galaxies",
) -> Dict[str, object]:
    snapshot_meta_path = _snapshot_meta_store_path(shard_root)
    write_snapshot_meta_store(snapshot_meta_path, state.snapshot)
    root_summaries = _root_store_summaries(state, tasks_by_root=tasks_by_root)
    store_specs = _bucket_root_stores(root_summaries)

    stores: Dict[str, Dict[str, object]] = {}
    roots: Dict[int, Dict[str, object]] = {}
    for spec in store_specs:
        store_id = str(spec["store_id"])
        kind = str(spec["kind"])
        root_ids = tuple(int(v) for v in spec["root_ids"])
        particle_store, node_store, store_summary = build_store_for_roots(state, root_ids=root_ids)
        particle_store_path = _particle_store_path(shard_root, store_id=store_id, kind=kind)
        node_store_path = _node_store_path(shard_root, store_id=store_id, kind=kind)
        write_particle_store(particle_store_path, particle_store)
        write_node_store(
            node_store_path,
            node_store,
            root_ids=root_ids,
            particle_rows_by_ptype=store_summary.get("particle_rows_by_ptype", {}),
        )
        store_rec = {
            "store_id": store_id,
            "kind": kind,
            "root_ids": root_ids,
            "particle_store_path": str(particle_store_path),
            "node_store_path": str(node_store_path),
            "node_count": int(spec.get("node_count", 0)),
            "particle_row_total": int(spec.get("particle_row_total", 0)),
            "fof_candidates": int(spec.get("fof_candidates", 0)),
            "particle_rows_by_ptype": {
                str(k): int(v)
                for k, v in dict(store_summary.get("particle_rows_by_ptype", {})).items()
            },
        }
        stores[store_id] = store_rec
        for root_id in root_ids:
            root_summary = dict(root_summaries.get(int(root_id), {}))
            roots[int(root_id)] = {
                "root_id": int(root_id),
                "store_id": store_id,
                "kind": kind,
                "particle_store_path": str(particle_store_path),
                "node_store_path": str(node_store_path),
                "node_count": int(root_summary.get("node_count", 0)),
                "task_count": int(root_summary.get("task_count", 0)),
                "particle_row_total": int(root_summary.get("particle_row_total", 0)),
                "fof_candidates": int(root_summary.get("fof_candidates", 0)),
            }

    manifest = {
        "snapshot_meta_store_path": str(snapshot_meta_path),
        "stores": stores,
        "roots": roots,
    }
    _write_store_manifest(shard_root, manifest)
    if log_fn is not None:
        bucket_count = sum(1 for rec in stores.values() if str(rec.get("kind")) == "bucket")
        log_fn(
            f"{log_label}: immutable stores ready; stores={len(stores)}, buckets={bucket_count}, roots={len(roots)}"
        )
    return manifest


def _classify_galaxy_finding_batches(
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

    tiny_tasks = [task for task in tasks if int(task.star_count) <= int(tiny_star_threshold)]
    regular_tasks = [task for task in tasks if int(task.star_count) > int(tiny_star_threshold)]
    tiny_batches = _build_tiny_batches(
        tiny_tasks,
        max_nodes_per_batch=int(tiny_max_nodes_per_batch),
        max_fof_candidates_per_batch=int(tiny_max_fof_candidates_per_batch),
    )
    regular_items = [
        AHFSubhaloBatch(tasks=(task,), is_tiny_batch=False, target_backend="cpu", estimated_cost=int(task.fof_candidates))
        for task in sorted(regular_tasks, key=lambda t: int(t.fof_candidates), reverse=True)
    ]
    small_items = sorted(
        tiny_batches,
        key=lambda batch: (
            int(batch.estimated_cost),
            int(len(batch.tasks)),
            int(batch.tasks[0].node_id) if batch.tasks else -1,
        ),
    )

    if gpu_worker_count <= 0:
        small_items.extend(
            AHFSubhaloBatch(tasks=(task,), is_tiny_batch=False, target_backend="cpu", estimated_cost=int(task.fof_candidates))
            for task in sorted(regular_tasks, key=lambda t: int(t.fof_candidates))
        )
        regular_items = []
    if cpu_worker_count <= 0 and small_items:
        regular_items.extend(
            AHFSubhaloBatch(
                tasks=item.tasks,
                is_tiny_batch=bool(item.is_tiny_batch),
                target_backend="gpu",
                estimated_cost=int(item.estimated_cost),
            )
            for item in sorted(small_items, key=lambda batch: int(batch.estimated_cost), reverse=True)
        )
        small_items = []

    return regular_items, small_items


def _classify_reconciling_subhalos_roots(
    *,
    root_costs: Dict[int, int],
    gpu_worker_count: int,
    cpu_worker_count: int,
) -> Tuple[List[int], List[int]]:
    ordered_roots = [
        int(root_id)
        for root_id, _cost in sorted(
            root_costs.items(),
            key=lambda kv: int(kv[1]),
            reverse=True,
        )
    ]
    if gpu_worker_count <= 0:
        return ordered_roots, []
    return ordered_roots, []


def _reset_rank0_log_context(*, now: Optional[float] = None) -> None:
    global _RANK0_LOG_TOTAL_START, _RANK0_LOG_STAGE_STARTS
    _RANK0_LOG_TOTAL_START = float(time.monotonic() if now is None else now)
    _RANK0_LOG_STAGE_STARTS = {}


def _rank0_log_stage_name(message: str) -> Optional[str]:
    if ":" not in str(message):
        return None
    token = str(message).split(":", 1)[0].strip()
    if not token:
        return None
    return token


def _rank0_log(message: str) -> None:
    global _RANK0_LOG_TOTAL_START, _RANK0_LOG_STAGE_STARTS
    now = float(time.monotonic())
    if _RANK0_LOG_TOTAL_START is None:
        _reset_rank0_log_context(now=now)
    total_elapsed = now - float(_RANK0_LOG_TOTAL_START)
    parts = [f"[AHF-subhalo][rank0][total={total_elapsed:.1f}s]"]
    stage_name = _rank0_log_stage_name(message)
    if stage_name is not None:
        stage_start = _RANK0_LOG_STAGE_STARTS.get(stage_name)
        if stage_start is None:
            stage_start = now
            _RANK0_LOG_STAGE_STARTS[stage_name] = stage_start
        parts.append(f"[{stage_name}={now - float(stage_start):.1f}s]")
    print("".join(parts) + f" {message}", flush=True)


def _progress_style() -> str:
    style = _env_str("CAESAR_AHF_SUBHALO_PROGRESS_STYLE", "text").strip().lower()
    if style in {"bar", "bars", "progress", "progressbar"}:
        return "bar"
    return "text"


def _progress_bar(completed: int, total: int) -> str:
    width = max(10, _env_int("CAESAR_AHF_SUBHALO_PROGRESS_BAR_WIDTH", 28))
    total_int = max(0, int(total))
    done_int = max(0, int(completed))
    if total_int <= 0:
        filled = 0
    else:
        frac = min(1.0, max(0.0, float(done_int) / float(total_int)))
        filled = int(round(frac * float(width)))
    filled = max(0, min(int(width), int(filled)))
    return "[" + "#" * filled + "-" * (int(width) - filled) + "]"


def _progress_metric_text(label: str, value: int, total: Optional[int] = None) -> str:
    if total is None:
        return f"{label}={int(value)}"
    return f"{label}={int(value)}/{int(total)}"


def _imbalance_ratio(values: Sequence[float]) -> str:
    resolved = [float(v) for v in values]
    if not resolved:
        return "n/a"
    max_value = max(resolved)
    min_value = min(resolved)
    if max_value <= 0.0 and min_value <= 0.0:
        return "1.00"
    if min_value <= 0.0:
        return "inf"
    return f"{max_value / min_value:.2f}"


def _progress_message(
    *,
    label: str,
    completed: int,
    total: int,
    unit: str,
    elapsed: float,
    metrics: Optional[Sequence[Tuple[str, int, Optional[int]]]] = None,
    active: Optional[int] = None,
    remaining: Optional[int] = None,
    detail: Optional[str] = None,
) -> str:
    metric_parts = [_progress_metric_text(name, value, total_value) for name, value, total_value in (metrics or [])]
    if _progress_style() == "bar":
        total_int = max(0, int(total))
        done_int = max(0, int(completed))
        if _tqdm is not None and total_int > 0:
            meter = _tqdm.format_meter(
                n=done_int,
                total=total_int,
                elapsed=max(0.0, float(elapsed)),
                prefix=f"{label} ({unit}) ",
                ascii=True,
                unit=str(unit),
                bar_format="{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
            )
        else:
            meter = f"{label} {_progress_bar(done_int, total_int)} {done_int}/{total_int} {unit}"
        parts = [meter]
        parts.extend(metric_parts)
        if active is not None:
            parts.append(f"active={int(active)}")
        if remaining is not None:
            parts.append(f"remaining={int(remaining)}")
        if detail:
            parts.append(str(detail))
        parts.append(f"elapsed={float(elapsed):.1f}s")
        return " | ".join(parts)

    parts = [f"{label}; completed={int(completed)}/{int(total)} {unit}"]
    if metric_parts:
        parts.append(", ".join(metric_parts))
    if active is not None:
        parts.append(f"active={int(active)}")
    if remaining is not None:
        parts.append(f"remaining={int(remaining)}")
    if detail:
        parts.append(str(detail))
    parts.append(f"elapsed={float(elapsed):.1f}s")
    return ", ".join(parts)


def _stage_queue(source, *, total: Optional[int] = None) -> _StageQueue:
    if hasattr(source, "next_front") and hasattr(source, "next_back"):
        return source
    resolved_total = int(total) if total is not None else -1
    return _StageQueue(items=list(source), total=int(resolved_total))


def _cpu_local_workers(cap: WorkerCapability, *, stage_name: str, threads: Optional[int] = None) -> int:
    if str(cap.role) != "cpu_worker" or str(stage_name) not in {"finding_galaxies", "reconciling_subhalos"}:
        return max(1, int(threads if threads is not None else cap.threads))
    default = max(1, int(threads if threads is not None else cap.threads))
    configured = max(
        1,
        _env_int(f"CAESAR_AHF_SUBHALO_MPI_{str(stage_name).upper()}_LOCAL_WORKERS", default),
    )
    return int(min(default, configured))


def _bundle_items(items: Sequence[Dict]) -> Optional[Dict]:
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    bundle = dict(items[0])
    bundle["items"] = list(items)
    bundle["completed_count"] = len(items)
    return bundle


def _take_cpu_bundle(
    *,
    small_queue: _StageQueue,
    regular_queue: _StageQueue,
    max_items: int,
    cap: WorkerCapability,
    prepare_small,
    prepare_regular,
    use_regular_back: bool,
):
    prepared: List[Dict] = []
    for _ in range(max(1, int(max_items))):
        raw_small = small_queue.next_front()
        if raw_small is not None:
            prepared.append(prepare_small(raw_small, cap=cap, direction="small"))
            continue
        raw_regular = regular_queue.next_back() if use_regular_back else regular_queue.next_front()
        if raw_regular is None:
            break
        prepared.append(
            prepare_regular(
                raw_regular,
                cap=cap,
                direction="back" if use_regular_back else "front",
            )
        )
    return _bundle_items(prepared)


def _dispatch_stage(
    comm,
    *,
    worker_caps: Sequence[WorkerCapability],
    regular_queue,
    small_queue,
    stage_name: str,
    display_name: Optional[str] = None,
    prepare_regular=None,
    prepare_small=None,
    regular_total: Optional[int] = None,
    small_total: Optional[int] = None,
    worker_threads: Optional[Dict[int, int]] = None,
    worker_roles: Optional[Dict[int, str]] = None,
    progress_unit: str = "items",
    progress_metrics: Optional[Sequence[ProgressMetric]] = None,
):
    regular_state = _stage_queue(regular_queue, total=regular_total)
    small_state = _stage_queue(small_queue, total=small_total)
    total_items = int(regular_state.total) + int(small_state.total)
    active = 0
    results = []
    completed = 0
    progress_every = max(1, _env_int("CAESAR_AHF_SUBHALO_MPI_PROGRESS_EVERY", 100))
    progress_seconds = max(5.0, _env_float("CAESAR_AHF_SUBHALO_MPI_PROGRESS_SECONDS", 60.0))
    start_time = time.monotonic()
    last_status = start_time
    last_completion_log = 0
    role_for = {
        int(cap.rank): str(worker_roles.get(int(cap.rank), cap.role) if worker_roles is not None else cap.role)
        for cap in worker_caps
    }
    threads_for = {
        int(cap.rank): max(1, int(worker_threads.get(int(cap.rank), cap.threads) if worker_threads is not None else cap.threads))
        for cap in worker_caps
    }
    has_gpu_workers = any(str(role_for.get(int(cap.rank), cap.role)) == "gpu_worker" for cap in worker_caps)
    has_cpu_workers = any(str(role_for.get(int(cap.rank), cap.role)) == "cpu_worker" for cap in worker_caps)
    cap_by_rank = {int(cap.rank): cap for cap in worker_caps}
    metric_defs = list(progress_metrics or [])
    metric_totals = {str(metric.label): (None if metric.total is None else int(metric.total)) for metric in metric_defs}
    metric_counts = {str(metric.label): 0 for metric in metric_defs}
    rank_work = {}
    for cap in worker_caps:
        rank_id = int(cap.rank)
        rank_work[rank_id] = {
            "role": str(role_for.get(rank_id, cap.role)),
            "busy_seconds": 0.0,
            "completed": 0,
            "metrics": {str(metric.label): 0 for metric in metric_defs},
        }

    def log_progress(prefix: str) -> None:
        elapsed = time.monotonic() - start_time
        queued_remaining = int(regular_state.remaining) + int(small_state.remaining)
        shown_name = str(display_name if display_name is not None else stage_name)
        metrics = [
            (str(metric.label), int(metric_counts[str(metric.label)]), metric_totals[str(metric.label)])
            for metric in metric_defs
        ]
        detail = (
            f"regular_front_assigned={regular_state.front_assigned}, "
            f"regular_back_assigned={regular_state.back_assigned}, "
            f"small_assigned={small_state.front_assigned}, "
            f"regular_remaining={regular_state.remaining}, "
            f"small_remaining={small_state.remaining}"
        )
        _rank0_log(
            _progress_message(
                label=f"{shown_name}: {prefix}",
                completed=completed,
                total=total_items,
                unit=progress_unit,
                elapsed=elapsed,
                metrics=metrics,
                active=active,
                remaining=queued_remaining,
                detail=detail if _progress_style() != "bar" else None,
            )
        )

    def log_workload_imbalance() -> None:
        if not rank_work:
            return
        shown_name = str(display_name if display_name is not None else stage_name)
        def _log_rank_subset(rank_ids: Sequence[int], label: str) -> None:
            resolved_ranks = [int(rank_id) for rank_id in rank_ids]
            if not resolved_ranks:
                return
            subset = [rank_work[int(rank_id)] for rank_id in resolved_ranks]
            busy_values = [float(rec["busy_seconds"]) for rec in subset]
            summary_parts = [
                f"busy_s max/min={_imbalance_ratio(busy_values)}"
            ]
            completed_values = [float(rec["completed"]) for rec in subset]
            summary_parts.append(
                f"{progress_unit} max/min={_imbalance_ratio(completed_values)}"
            )
            for metric in metric_defs:
                metric_label = str(metric.label)
                values = [float(rec["metrics"][metric_label]) for rec in subset]
                summary_parts.append(f"{metric_label} max/min={_imbalance_ratio(values)}")
            _rank0_log(f"{shown_name}: {label} workload imbalance; " + "; ".join(summary_parts))

        all_ranks = sorted(rank_work)
        if str(stage_name) == "finding_galaxies":
            gpu_ranks = [
                int(rank_id)
                for rank_id in all_ranks
                if str(rank_work[int(rank_id)]["role"]) == "gpu_worker"
            ]
            cpu_ranks = [
                int(rank_id)
                for rank_id in all_ranks
                if str(rank_work[int(rank_id)]["role"]) == "cpu_worker"
            ]
            _log_rank_subset(cpu_ranks, "CPU rank")
            if len(gpu_ranks) >= 2:
                _log_rank_subset(gpu_ranks, "GPU rank")
            return
        _log_rank_subset(all_ranks, "rank")

    def assign_next(rank: int, cap: WorkerCapability):
        nonlocal active
        item = None
        stage_role = str(role_for.get(int(rank), cap.role))
        stage_threads = int(threads_for.get(int(rank), cap.threads))
        if stage_role == "gpu_worker" and cap.gpu_device is not None:
            raw = regular_state.next_front()
            if raw is not None:
                item = prepare_regular(raw, cap=cap, direction="front") if prepare_regular else raw
            elif not has_cpu_workers:
                raw = small_state.next_front()
                if raw is not None:
                    item = prepare_small(raw, cap=cap, direction="small") if prepare_small else raw
        elif stage_role == "cpu_worker":
            item = _take_cpu_bundle(
                small_queue=small_state,
                regular_queue=regular_state,
                max_items=_cpu_local_workers(cap, stage_name=stage_name, threads=stage_threads),
                cap=cap,
                prepare_small=prepare_small or (lambda raw, **_: raw),
                prepare_regular=prepare_regular or (lambda raw, **_: raw),
                use_regular_back=bool(has_gpu_workers),
            )
        elif stage_role == "property_worker":
            raw = small_state.next_front()
            if raw is not None:
                item = prepare_small(raw, cap=cap, direction="small") if prepare_small else raw

        if item is None:
            return False

        if isinstance(item, dict):
            item = dict(item)
            item["nproc"] = int(stage_threads)
            item["worker_role"] = str(stage_role)

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
            if now - last_status >= progress_seconds:
                log_progress("status")
                last_status = now
            time.sleep(0.5)
            continue
        status = MPI.Status()
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MSG_RESULT, status=status)
        src = int(status.Get_source())
        active -= 1
        results.append(msg)
        completed_delta = int(msg.get("completed_count", 1))
        completed += completed_delta
        for metric in metric_defs:
            metric_counts[str(metric.label)] += int(msg.get(metric.result_key, 0))
        cap = cap_by_rank[int(src)]
        rank_state = rank_work[int(src)]
        rank_state["busy_seconds"] += float(msg.get("elapsed_seconds", 0.0))
        rank_state["completed"] += completed_delta
        for metric in metric_defs:
            label = str(metric.label)
            rank_state["metrics"][label] += int(msg.get(metric.result_key, 0))
        assign_next(src, cap)
        if completed == total_items or completed - last_completion_log >= progress_every:
            log_progress("progress")
            last_completion_log = completed
            last_status = time.monotonic()

    log_progress("complete")
    log_workload_imbalance()
    for queue in (regular_state, small_state):
        close_fn = getattr(queue, "close", None)
        if callable(close_fn):
            close_fn()
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


def _direct_fof_linking_length(snapshot) -> float:
    ndm = int(snapshot.particle_counts.get("dm", 0))
    if ndm <= 0:
        raise RuntimeError("AHF-subhalo direct runtime requires dark-matter particles to compute fof_ll")
    b_halo = 0.2
    b_galaxy = float(os.environ.get("CAESAR_B_GALAXY", b_halo * 0.1))
    mis = float(snapshot.boxsize) / float(ndm) ** (1.0 / 3.0)
    return float(mis * b_galaxy)


def _rank0_prepare_galaxy_finding(
    *,
    snapshot_file: str,
    ahf_particles_file: str,
    shard_root: Path,
    nproc: int,
    min_stars: Optional[int],
    log_fn=None,
    log_label: str = "finding galaxies",
):
    start_time = time.monotonic()
    if log_fn is not None:
        log_fn(f"{log_label}: loading direct-HDF5 snapshot tables and AHF node manifest")
    direct_state = build_direct_state(snapshot_file, ahf_particles_file)
    fof_ll = _direct_fof_linking_length(direct_state.snapshot)
    fof_vel_ll = 1.0
    try:
        _vel_env = os.environ.get("CAESAR_FOF6D_VEL_LL")
        if _vel_env not in (None, ""):
            fof_vel_ll = float(_vel_env)
    except Exception:
        pass
    if os.environ.get("CAESAR_FOF6D_DISABLE_VEL", "0") == "1":
        fof_vel_ll = None
    pid_maps_sel = None
    if log_fn is not None:
        log_fn(
            f"{log_label}: direct runtime ready; ptypes={','.join(direct_state.snapshot.ptypes)}, "
            f"nodes={len(direct_state.nodes)}, "
            f"elapsed={time.monotonic() - start_time:.1f}s"
        )
    ms = get_min_stars(None, override=min_stars)
    if log_fn is not None:
        log_fn(f"{log_label}: building task manifest from AHF hierarchy")
    node_ndm = {
        int(hid): int(ndm)
        for hid, ndm in zip(direct_state.nodes.halo_id.tolist(), direct_state.nodes.dm_count.tolist())
    }
    tasks, tasks_by_root, _, _ = _prepare_manifest(
        ahf_particles_file=ahf_particles_file,
        node_ndm=node_ndm,
        min_stars=int(ms),
    )
    if log_fn is not None:
        log_fn(
            f"{log_label}: task manifest ready; tasks={len(tasks)}, roots={len(tasks_by_root)}, "
            f"elapsed={time.monotonic() - start_time:.1f}s"
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
    progress_every = max(1, _env_int("CAESAR_AHF_SUBHALO_MPI_PREP_PROGRESS_EVERY", 1000))
    progress_seconds = max(5.0, _env_float("CAESAR_AHF_SUBHALO_MPI_PREP_PROGRESS_SECONDS", 60.0))
    last_status = time.monotonic()
    dropped_tasks = 0
    if log_fn is not None:
        log_fn(f"{log_label}: building per-node finding-galaxies payloads for {len(tasks)} tasks")
    for idx, task in enumerate(tasks, start=1):
        payload = _build_direct_task_payload(
            direct_state,
            task=task,
            fof_nHlim=float(fof_nHlim),
            fof_Tlim=float(fof_Tlim),
            fof_use_sfr_gate=bool(fof_use_sfr_gate),
        )
        if payload is None or int(np.asarray(payload["star_sel"]).size) < int(ms):
            dropped_tasks += 1
        else:
            payload["task"] = _serialize_task(task)
            task_payloads_by_node[int(task.node_id)] = payload
            filtered_tasks.append(task)
        now = time.monotonic()
        if (
            log_fn is not None
            and (
                idx == len(tasks)
                or idx % progress_every == 0
                or now - last_status >= progress_seconds
            )
        ):
            log_fn(
                _progress_message(
                    label=f"{log_label}: payload prep",
                    completed=idx,
                    total=len(tasks),
                    unit="tasks",
                    elapsed=now - start_time,
                    metrics=[
                        ("kept", len(filtered_tasks), None),
                        ("dropped", dropped_tasks, None),
                    ],
                )
            )
            last_status = now

    filtered_ids = {int(task.node_id) for task in filtered_tasks}
    tasks = filtered_tasks
    tasks_by_root = {
        int(root): [task for task in root_tasks if int(task.node_id) in filtered_ids]
        for root, root_tasks in tasks_by_root.items()
    }
    tasks_by_root = {int(root): root_tasks for root, root_tasks in tasks_by_root.items() if root_tasks}
    if log_fn is not None:
        log_fn(f"{log_label}: writing immutable particle/node stores for {len(tasks_by_root)} roots")
    store_manifest = _build_store_artifacts(
        state=direct_state,
        tasks_by_root=tasks_by_root,
        shard_root=shard_root,
        log_fn=log_fn,
        log_label=log_label,
    )

    return (
        direct_state,
        pid_maps_sel,
        int(ms),
        float(fof_ll),
        fof_vel_ll,
        tasks,
        tasks_by_root,
        task_payloads_by_node,
        store_manifest,
    )


def _materialize_galaxy_finding_batch_item(
    *,
    shard_root: Path,
    batch: AHFSubhaloBatch,
    prefix: str,
    batch_index: int,
    task_payloads_by_node: Dict[int, Dict[str, object]],
    fof_ll: float,
    fof_vel_ll: Optional[float],
    min_stars: int,
    backend: str,
    device_id: Optional[int],
    store_ids: Sequence[str],
):
    payload_path = shard_root / f"finding_galaxies_payload_{prefix}_{int(batch_index):06d}.pkl"
    payload = {
        "batch": _serialize_batch(batch),
        "task_payloads": [task_payloads_by_node[int(task.node_id)] for task in batch.tasks],
    }
    _dump_pickle(payload_path, payload)
    return {
        "batch": _serialize_batch(batch),
        "payload_path": str(payload_path),
        "backend": str(backend),
        "device_id": device_id,
        "fof_ll": float(fof_ll),
        "fof_vel_ll": fof_vel_ll,
        "min_stars": int(min_stars),
        "store_ids": tuple(sorted(str(v) for v in store_ids)),
    }


def _write_reconciliation_root_payloads(
    *,
    shard_root: Path,
    tasks_by_root: Dict[int, List[AHFSubhaloTask]],
    store_manifest: Mapping[str, object],
    direct_state: Optional[AHFSubhaloDirectState] = None,
    log_fn=None,
    progress_label: str = "reconciling subhalos",
) -> Dict[int, str]:
    root_payload_paths: Dict[int, str] = {}
    eligible_halo_ids_by_root = (
        _eligible_halo_node_ids_by_root(direct_state)
        if isinstance(direct_state, AHFSubhaloDirectState)
        else {}
    )
    total_roots = len(tasks_by_root)
    progress_every = max(1, _env_int("CAESAR_AHF_SUBHALO_MPI_PREP_PROGRESS_EVERY", 1000))
    progress_seconds = max(5.0, _env_float("CAESAR_AHF_SUBHALO_MPI_PREP_PROGRESS_SECONDS", 60.0))
    start_time = time.monotonic()
    last_status = start_time
    for idx, (root_id, root_tasks) in enumerate(tasks_by_root.items(), start=1):
        root_store = dict(dict(store_manifest.get("roots", {})).get(int(root_id), {}))
        if not root_store:
            raise RuntimeError(f"Missing store manifest entry for root {int(root_id)}")
        root_halo_ids: List[int] = []
        if isinstance(direct_state, AHFSubhaloDirectState):
            root_halo_ids = sorted(
                eligible_halo_ids_by_root.get(int(root_id), []),
                key=lambda halo_id: (
                    int(direct_state.nodes.depth[direct_state.nodes.index_of(int(halo_id))]),
                    int(halo_id),
                ),
            )
        payload_path = shard_root / f"reconciling_subhalos_manifest_root{int(root_id):08d}.pkl"
        payload = {
            "root_id": int(root_id),
            "tasks": [_serialize_task(task) for task in root_tasks],
            "halo_node_ids": [int(v) for v in root_halo_ids],
            "halo_count": int(len(root_halo_ids)),
            "store_id": str(root_store.get("store_id")),
            "store_kind": str(root_store.get("kind")),
            "particle_store_path": str(root_store.get("particle_store_path")),
            "node_store_path": str(root_store.get("node_store_path")),
        }
        _dump_pickle(payload_path, payload)
        root_payload_paths[int(root_id)] = str(payload_path)
        now = time.monotonic()
        if (
            log_fn is not None
            and (
                idx == total_roots
                or idx % progress_every == 0
                or now - last_status >= progress_seconds
            )
        ):
            log_fn(
                _progress_message(
                    label=f"{progress_label}: wrote root payloads",
                    completed=idx,
                    total=total_roots,
                    unit="roots",
                    elapsed=now - start_time,
                )
            )
            last_status = now
    return root_payload_paths


def _worker_run_finding_galaxies(
    *,
    item: Dict,
    fof_ll: float,
    fof_vel_ll: Optional[float],
    min_stars: int,
    shard_dir: Path,
    nproc: int = 1,
) -> Dict:
    nproc = int(item.get("nproc", nproc))
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
            "results": _galaxy_finding_shard_payload(results),
            "node_ids": [int(t.node_id) for t in batch.tasks],
            "root_ids": sorted({int(t.top_id) for t in batch.tasks}),
            "store_ids": [str(v) for v in one_item.get("store_ids", ())],
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
    store_ids = set()
    input_payload_paths = []
    for out in outputs:
        combined_payload.update(out["results"])
        node_ids.extend(int(v) for v in out["node_ids"])
        root_ids.update(int(v) for v in out["root_ids"])
        store_ids.update(str(v) for v in out.get("store_ids", []))
        input_payload_paths.append(str(out["input_payload_path"]))

    shard_path = shard_dir / f"finding_galaxies_shard_rank{os.getpid()}_{abs(hash(tuple(sorted(node_ids))))}.pkl"
    _dump_pickle(shard_path, combined_payload)
    return {
        "stage": "finding_galaxies",
        "shard_path": str(shard_path),
        "node_ids": sorted(set(int(v) for v in node_ids)),
        "count_nodes": int(len(set(int(v) for v in node_ids))),
        "root_ids": sorted(root_ids),
        "store_ids": sorted(store_ids),
        "input_payload_paths": input_payload_paths,
        "completed_count": len(items),
    }


def _worker_run_reconciling_subhalos(
    *,
    item: Dict,
    fof_ll: float,
    fof_vel_ll: Optional[float],
    min_stars: int,
    shard_dir: Path,
    nproc: int = 1,
) -> Dict:
    nproc = int(item.get("nproc", nproc))
    items = list(item.get("items", [item]))
    cc_backend = _env_str("CAESAR_AHF_SUBHALO_CC_BACKEND", "auto").lower()
    max_pairs_per_batch = max(1, _env_int("CAESAR_AHF_SUBHALO_MAX_PAIRS_PER_BATCH", 5_000_000))
    halo_overhead = max(0, _env_int("CAESAR_AHF_SUBHALO_CALCULATING_PROPERTIES_HALO_OVERHEAD", 128))
    galaxy_overhead = max(0, _env_int("CAESAR_AHF_SUBHALO_CALCULATING_PROPERTIES_GALAXY_OVERHEAD", 128))
    fof_nHlim = _env_float("CAESAR_AHF_FAST_FOF_NHLIM", 0.13)
    fof_Tlim = _env_float("CAESAR_AHF_FAST_FOF_TLIM", 1.0e5)
    fof_use_sfr_gate = os.environ.get("CAESAR_AHF_FAST_FOF_USE_SFR", "1") == "1"

    def run_one(one_item: Dict):
        root_id = int(one_item["root_id"])
        root_payload = _load_pickle(Path(one_item["root_payload_path"]))
        root_tasks = [_deserialize_task(task) for task in root_payload["tasks"]]
        particle_store = load_particle_store(Path(str(root_payload["particle_store_path"])))
        node_store = load_node_store(Path(str(root_payload["node_store_path"])))
        store_state = AHFSubhaloDirectState(
            snapshot=particle_store.meta,
            particles=particle_store,
            nodes=node_store["nodes"],
        )
        halo_records = [
            _build_store_halo_record(store_state, node_id=int(node_id))
            for node_id in root_payload.get("halo_node_ids", [])
        ]
        task_payloads_by_node = {}
        for task in root_tasks:
            task_payload = _build_direct_task_payload(
                store_state,
                task=task,
                fof_nHlim=float(fof_nHlim),
                fof_Tlim=float(fof_Tlim),
                fof_use_sfr_gate=bool(fof_use_sfr_gate),
            )
            if task_payload is None:
                raise RuntimeError(
                    f"Failed to rebuild task payload from store for root {root_id} node {int(task.node_id)}"
                )
            task_payloads_by_node[int(task.node_id)] = task_payload
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
        galaxies_payload = [gal if isinstance(gal, dict) else _serialize_candidate_group(gal) for gal in galaxies]
        particle_cost = int(sum(_group_record_member_cost(rec) for rec in halo_records))
        particle_cost += int(sum(_group_record_member_cost(rec) for rec in galaxies_payload))
        structure_cost = int(len(halo_records)) * int(halo_overhead)
        structure_cost += int(len(galaxies_payload)) * int(galaxy_overhead)
        return {
            "root_id": int(root_id),
            "galaxies": galaxies_payload,
            "count": int(len(galaxies_payload)),
            "halo_count": int(root_payload.get("halo_count", len(halo_records))),
            "property_cost": int(particle_cost + structure_cost),
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

    root_results = []
    total_count = 0
    for out in outputs:
        root_id = int(out["root_id"])
        shard_path = shard_dir / f"reconciling_subhalos_shard_root{root_id:08d}.pkl"
        _dump_pickle(shard_path, _reconciled_galaxy_shard_payload(out["galaxies"]))
        root_results.append(
            {
                "root_id": int(root_id),
                "shard_path": str(shard_path),
                "count": int(out["count"]),
                "halo_count": int(out.get("halo_count", 0)),
                "property_cost": int(out.get("property_cost", 0)),
            }
        )
        total_count += int(out["count"])
    return {
        "stage": "reconciling_subhalos",
        "root_results": sorted(root_results, key=lambda rec: int(rec["root_id"])),
        "count": int(total_count),
        "completed_count": len(items),
    }


def _split_even(seq: Sequence, parts: int) -> List[List]:
    buckets = [[] for _ in range(max(1, int(parts)))]
    for idx, item in enumerate(seq):
        buckets[idx % len(buckets)].append(item)
    return buckets


def _calculating_properties_galaxy_work_payload(groups: Sequence) -> List[dict]:
    return [_serialize_candidate_group(gal) for gal in groups]


def _calculating_properties_halo_work_payload(halos: Sequence) -> List[int]:
    return [int(getattr(halo, "AHF_haloID", -1)) for halo in halos]


def _calculating_properties_halo_cost(halo) -> int:
    try:
        return max(1, int(len(getattr(halo, "global_indexes", []))))
    except Exception:
        return 1


def _calculating_properties_galaxy_cost(galaxy) -> int:
    try:
        return max(1, int(len(getattr(galaxy, "global_indexes", []))))
    except Exception:
        return 1


def _collect_calculating_properties_root_records(root_results: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    root_records = [
        {
            "root_id": int(rec["root_id"]),
            "shard_path": str(rec["shard_path"]),
            "root_payload_path": str(rec["root_payload_path"]),
            "store_id": str(rec["store_id"]),
            "particle_store_path": str(rec["particle_store_path"]),
            "node_store_path": str(rec["node_store_path"]),
            "halo_count": int(rec.get("halo_count", 0)),
            "galaxy_count": int(rec.get("count", 0)),
            "cost": int(rec.get("property_cost", rec.get("count", 0))),
        }
        for rec in root_results
        if int(rec.get("count", 0)) > 0
    ]
    root_records.sort(
        key=lambda rec: (
            int(rec["cost"]),
            int(rec["halo_count"]),
            int(rec["galaxy_count"]),
            int(rec["root_id"]),
        ),
        reverse=True,
    )
    return root_records


def _build_calculating_properties_batches(
    root_results: Sequence[Mapping[str, object]],
    worker_count: int,
) -> List[Dict[str, object]]:
    root_records = _collect_calculating_properties_root_records(root_results)
    if not root_records:
        return []

    by_store: Dict[str, Dict[str, object]] = {}
    for rec in root_records:
        store_id = str(rec["store_id"])
        bucket = by_store.setdefault(
            store_id,
            {
                "store_id": store_id,
                "particle_store_path": str(rec["particle_store_path"]),
                "node_store_path": str(rec["node_store_path"]),
                "roots": [],
                "cost": 0,
                "halo_count": 0,
                "galaxy_count": 0,
            },
        )
        bucket["roots"].append(dict(rec))
        bucket["cost"] = int(bucket["cost"]) + int(rec["cost"])
        bucket["halo_count"] = int(bucket["halo_count"]) + int(rec["halo_count"])
        bucket["galaxy_count"] = int(bucket["galaxy_count"]) + int(rec["galaxy_count"])

    batches = list(by_store.values())
    batches.sort(
        key=lambda rec: (
            int(rec["cost"]),
            int(rec["halo_count"]),
            int(rec["galaxy_count"]),
            len(rec["roots"]),
            str(rec["store_id"]),
        ),
        reverse=True,
    )
    return batches


def _calculating_properties_state_list_data(state: Dict, name: str):
    private = f"_{name}"
    if private in state:
        return state[private]
    if name in state:
        return state[name]
    return []


def _calculating_properties_pop_list_data(state: Dict, name: str) -> np.ndarray:
    private = f"_{name}"
    if private in state:
        return np.asarray(state[private])
    if name in state:
        return np.asarray(state[name])
    return np.asarray([], dtype=np.int64)


def _build_property_list_blocks(states: Sequence[Dict], list_names: Sequence[str]) -> Dict[str, Dict[str, object]]:
    blocks: Dict[str, Dict[str, object]] = {}
    owner_ids = np.asarray([int(state["id"]) for state in states], dtype=np.int64)
    for name in list_names:
        lengths = np.zeros(len(states), dtype=np.int64)
        parts = []
        dtype_str = None
        for idx, state in enumerate(states):
            arr = _calculating_properties_pop_list_data(state, name)
            lengths[idx] = int(arr.size)
            if arr.size > 0:
                if dtype_str is None:
                    dtype_str = arr.dtype.str
                parts.append(np.asarray(arr))
        if parts:
            data = np.concatenate(parts)
        else:
            data = np.empty(0, dtype=np.int64)
        blocks[str(name)] = {
            "owner_id": owner_ids.copy(),
            "lengths": lengths,
            "data": np.asarray(data),
            "dtype": dtype_str,
        }
    return blocks


def _build_property_shard_summary(halo_columns: Mapping[str, object], galaxy_columns: Mapping[str, object]) -> Dict[str, object]:
    halo_attrs = dict(halo_columns.get("attrs", {}))
    halo_dicts = dict(halo_columns.get("dicts", {}))
    galaxy_attrs = dict(galaxy_columns.get("attrs", {}))
    galaxy_dicts = dict(galaxy_columns.get("dicts", {}))
    return {
        "halos": {
            "id": np.asarray(halo_columns.get("id", []), dtype=np.int64),
            "total_mass": np.asarray(dict(halo_dicts.get("masses", {})).get("total", np.empty(0, dtype=np.float64)), dtype=np.float64),
            "pos": np.asarray(halo_attrs.get("pos", np.empty((0, 3), dtype=np.float64)), dtype=np.float64),
            "ahf_halo_id": np.asarray(halo_attrs.get("AHF_haloID", np.empty(0, dtype=np.int64)), dtype=np.int64),
            "list_lengths": {
                str(name): np.asarray(lengths, dtype=np.int64)
                for name, lengths in dict(halo_columns.get("list_lengths", {})).items()
            },
            "list_dtypes": dict(halo_columns.get("list_dtypes", {})),
            "attr_specs": dict(halo_columns.get("attr_specs", {})),
            "dict_specs": {
                str(dict_name): dict(submap)
                for dict_name, submap in dict(halo_columns.get("dict_specs", {})).items()
            },
        },
        "galaxies": {
            "id": np.asarray(galaxy_columns.get("id", []), dtype=np.int64),
            "stellar_mass": np.asarray(dict(galaxy_dicts.get("masses", {})).get("stellar", np.empty(0, dtype=np.float64)), dtype=np.float64),
            "total_mass": np.asarray(dict(galaxy_dicts.get("masses", {})).get("total", np.empty(0, dtype=np.float64)), dtype=np.float64),
            "pos": np.asarray(galaxy_attrs.get("pos", np.empty((0, 3), dtype=np.float64)), dtype=np.float64),
            "top_halo_id": np.asarray(galaxy_attrs.get("AHF_top_haloID", np.empty(0, dtype=np.int64)), dtype=np.int64),
            "list_lengths": {
                str(name): np.asarray(lengths, dtype=np.int64)
                for name, lengths in dict(galaxy_columns.get("list_lengths", {})).items()
            },
            "list_dtypes": dict(galaxy_columns.get("list_dtypes", {})),
            "attr_specs": dict(galaxy_columns.get("attr_specs", {})),
            "dict_specs": {
                str(dict_name): dict(submap)
                for dict_name, submap in dict(galaxy_columns.get("dict_specs", {})).items()
            },
        },
    }


def _worker_run_calculating_properties(
    *,
    item: Dict,
    shard_dir: Path,
    nproc: int,
) -> Dict:
    nproc = int(item.get("nproc", nproc))
    if "roots" in item:
        if "particle_store_path" not in item or "node_store_path" not in item:
            raise RuntimeError("Calculating properties worker requires particle_store_path and node_store_path")
        particle_store = load_particle_store(Path(str(item["particle_store_path"])))
        node_store = load_node_store(Path(str(item["node_store_path"])))
        store_state = AHFSubhaloDirectState(
            snapshot=particle_store.meta,
            particles=particle_store,
            nodes=node_store["nodes"],
        )
        halo_records: List[Dict[str, object]] = []
        galaxy_records: List[Dict[str, object]] = []
        root_ids: List[int] = []
        for rec in item["roots"]:
            root_id = int(rec["root_id"])
            root_payload = _load_pickle(Path(str(rec["root_payload_path"])))
            halo_records.extend(
                [
                    _build_store_halo_record(store_state, node_id=int(node_id))
                    for node_id in root_payload.get("halo_node_ids", [])
                ]
            )
            reconciled_payload = _load_pickle(Path(str(rec["shard_path"])))
            galaxies = (
                candidate_records_from_table_payload(reconciled_payload)
                if isinstance(reconciled_payload, dict) and "ahf_halo_id" in reconciled_payload
                else list(reconciled_payload)
            )
            galaxy_records.extend([dict(v) for v in galaxies])
            root_ids.append(int(root_id))
        payload = _build_calculating_properties_payload_from_particle_store(
            particle_store,
            halo_records=halo_records,
            galaxy_records=galaxy_records,
        )
        shard_source = (str(item.get("store_id", "")), tuple(sorted(root_ids)))
    else:
        raise RuntimeError("Calculating properties worker requires assigned roots plus particle_store_path and node_store_path")
    sim = _build_calculating_properties_runtime(payload, nproc=int(nproc))

    _compute_group_properties_subset(sim, group_type="halo", groups=list(sim.halo_list))
    _compute_group_properties_subset(sim, group_type="galaxy", groups=list(sim.galaxy_list))

    halo_states = [_serialize_group_state(group, id_key="_merge_id") for group in sim.halo_list]
    galaxy_states = [_serialize_group_state(group, id_key="_merge_id") for group in sim.galaxy_list]
    halo_payload_by_id = {
        int(rec["_merge_id"]): rec
        for rec in payload.get("halos", [])
        if "_merge_id" in rec
    }
    galaxy_payload_by_id = {
        int(rec["_merge_id"]): rec
        for rec in payload.get("galaxies", [])
        if "_merge_id" in rec
    }
    for state in halo_states:
        rec = halo_payload_by_id.get(int(state["id"]))
        if rec is None:
            continue
        for name in ("glist", "slist", "dmlist", "bhlist", "dlist"):
            global_name = f"global_{name}"
            if global_name in rec:
                state[f"_{name}"] = np.asarray(rec[global_name], dtype=np.int64)
    for state in galaxy_states:
        rec = galaxy_payload_by_id.get(int(state["id"]))
        if rec is None:
            continue
        for name in ("glist", "slist", "dmlist", "bhlist", "dlist"):
            global_name = f"global_{name}"
            if global_name in rec:
                state[f"_{name}"] = np.asarray(rec[global_name], dtype=np.int64)
    halo_columns = build_group_column_payload(
        halo_states,
        list_attrs=("dmlist", "glist", "slist", "bhlist", "dlist"),
        skip_attrs={
            "id",
            "_merge_id",
            "obj",
            "halo",
            "galaxies",
            "satellite_galaxies",
            "clouds",
            "galaxy",
            "central_galaxy",
            "galaxy_index_list",
            "global_indexes",
            "AHF_ancestor_haloIDs",
            "glist",
            "slist",
            "dmlist",
            "bhlist",
            "dlist",
            "_glist",
            "_slist",
            "_dmlist",
            "_bhlist",
            "_dlist",
        },
    )
    galaxy_columns = build_group_column_payload(
        galaxy_states,
        list_attrs=("glist", "slist", "bhlist", "dlist", "cloud_index_list", "AHF_ancestor_haloIDs"),
        skip_attrs={
            "id",
            "_merge_id",
            "obj",
            "halo",
            "galaxies",
            "satellite_galaxies",
            "clouds",
            "galaxy",
            "central_galaxy",
            "parent_halo_index",
            "_ahf_host_halo_index",
            "glist",
            "slist",
            "dmlist",
            "bhlist",
            "dlist",
            "cloud_index_list",
            "AHF_ancestor_haloIDs",
            "_glist",
            "_slist",
            "_dmlist",
            "_bhlist",
            "_dlist",
        },
    )
    summary = _build_property_shard_summary(halo_columns, galaxy_columns)
    halo_list_blocks = _build_property_list_blocks(halo_states, ("dmlist", "glist", "slist", "bhlist", "dlist"))
    galaxy_list_blocks = _build_property_list_blocks(
        galaxy_states,
        ("glist", "slist", "bhlist", "dlist", "cloud_index_list", "AHF_ancestor_haloIDs"),
    )
    shard_path = shard_dir / f"calculating_properties_shard_rank{os.getpid()}_{abs(hash(shard_source))}.pkl"
    _dump_pickle(
        shard_path,
        {
            "halo_columns": halo_columns,
            "galaxy_columns": galaxy_columns,
            "halo_lists": halo_list_blocks,
            "galaxy_lists": galaxy_list_blocks,
        },
    )
    return {
        "stage": "calculating_properties",
        "shard_path": str(shard_path),
        "count_halos": int(len(sim.halo_list)),
        "count_galaxies": int(len(sim.galaxy_list)),
        "summary": summary,
    }


def _build_caesar_runtime_from_loaded_ds(ds, *, nproc: int, snapshot_hash: Optional[str] = None):
    import caesar
    from caesar.property_manager import DatasetType

    sim = caesar.CAESAR()
    sim._ds = ds
    sim._ds_type = DatasetType(ds)
    if snapshot_hash not in (None, ""):
        sim.hash = str(snapshot_hash)
    sim.nproc = int(nproc)
    sim.load_haloid = False
    sim._assign_simulation_attributes()
    return sim


def _rank0_prepare_final_sim_from_existing(
    *,
    sim,
    pid_maps_sel,
    ahf_particles_file: str,
    final_shards: Sequence[str],
):
    final_galaxies = []
    for path in final_shards:
        payload = _load_pickle(Path(path))
        records = (
            candidate_records_from_table_payload(payload)
            if isinstance(payload, dict) and "ahf_halo_id" in payload
            else list(payload)
        )
        final_galaxies.extend(_deserialize_candidate_group(sim, rec) for rec in records)

    _prepare_final_subhalo_galaxies(
        sim,
        ahf_particles_file=ahf_particles_file,
        galaxy_list=final_galaxies,
        pid_maps_sel=pid_maps_sel,
        compute_missing_halo_properties=False,
    )
    return sim


def _rank0_build_final_sim(
    *,
    snapshot_file: str,
    ahf_particles_file: str,
    final_shards: Sequence[str],
    nproc: int,
    snapshot_hash: Optional[str] = None,
):
    import yt

    ds = yt.load(snapshot_file)
    sim = _build_caesar_runtime_from_loaded_ds(
        ds,
        nproc=int(nproc),
        snapshot_hash=snapshot_hash,
    )
    build_halos_from_ahf_fast(sim, ahf_particles_file, compute_properties=False)
    pid_maps_sel = _build_selected_pid_maps(sim)

    final_galaxies = []
    for path in final_shards:
        payload = _load_pickle(Path(path))
        records = (
            candidate_records_from_table_payload(payload)
            if isinstance(payload, dict) and "ahf_halo_id" in payload
            else list(payload)
        )
        final_galaxies.extend(_deserialize_candidate_group(sim, rec) for rec in records)

    _prepare_final_subhalo_galaxies(
        sim,
        ahf_particles_file=ahf_particles_file,
        galaxy_list=final_galaxies,
        pid_maps_sel=pid_maps_sel,
        compute_missing_halo_properties=False,
    )
    return sim


def _rank0_calculate_properties(
    comm,
    *,
    worker_caps: Sequence[WorkerCapability],
    snapshot_file: str,
    ahf_particles_file: str,
    output_file: str,
    root_results: Sequence[Mapping[str, object]],
    nproc: int,
    shard_root: Path,
    snapshot_hash: Optional[str] = None,
    sim=None,
    pid_maps_sel=None,
    worker_threads: Optional[Dict[int, int]] = None,
):
    snapshot_meta_path = _snapshot_meta_store_path(shard_root)
    if not snapshot_meta_path.is_file():
        raise RuntimeError(
            f"Calculating properties requires persisted snapshot meta store: {snapshot_meta_path}"
        )
    if isinstance(sim, AHFSubhaloDirectState):
        snapshot_meta = sim.snapshot
        _rank0_log("calculating properties: using in-memory snapshot metadata and worker-side store reads")
    else:
        snapshot_meta = load_snapshot_meta_store(snapshot_meta_path)
        _rank0_log("calculating properties: using persisted snapshot metadata and worker-side store reads")
    calculating_properties_root_results = [dict(rec) for rec in root_results if int(rec.get("count", 0)) > 0]
    property_batches = _build_calculating_properties_batches(calculating_properties_root_results, len(worker_caps))
    total_halos = int(sum(int(rec.get("halo_count", 0)) for rec in calculating_properties_root_results))
    total_galaxies = int(sum(int(rec.get("count", 0)) for rec in calculating_properties_root_results))
    _rank0_log(
        f"calculating properties: prepared store batches count={len(property_batches)} "
        f"halos={total_halos} galaxies={total_galaxies}"
    )
    calculating_properties_items = [
        {
            "store_id": str(batch["store_id"]),
            "particle_store_path": str(batch["particle_store_path"]),
            "node_store_path": str(batch["node_store_path"]),
            "roots": [dict(rec) for rec in batch["roots"]],
        }
        for batch in property_batches
    ]

    calculating_properties_results = _dispatch_stage(
        comm,
        worker_caps=worker_caps,
        regular_queue=[],
        small_queue=calculating_properties_items,
        stage_name="calculating_properties",
        display_name="calculating properties",
        regular_total=0,
        small_total=len(calculating_properties_items),
        worker_threads=worker_threads,
        worker_roles={int(cap.rank): "property_worker" for cap in worker_caps},
        progress_unit="batches",
        progress_metrics=[
            ProgressMetric("halos", "count_halos", total=total_halos),
            ProgressMetric("galaxies", "count_galaxies", total=total_galaxies),
        ],
    )
    return snapshot_meta, calculating_properties_results


def _rank0_write_catalogue(
    *,
    snapshot_meta,
    property_results,
    output_file: str,
):
    global_properties = compute_global_properties_from_property_shards(
        snapshot_meta=snapshot_meta,
        property_results=property_results,
        log_fn=_rank0_log,
        stage_label="global properties",
    )
    _rank0_log("writing: coordinator beginning final export")
    write_catalogue_from_property_shards(
        snapshot_meta=snapshot_meta,
        property_results=property_results,
        output_file=output_file,
        log_fn=_rank0_log,
        stage_label="writing",
        global_properties=global_properties,
    )


def _reconciling_subhalos_manifest_path(shard_root: Path) -> Path:
    return shard_root / "reconciling_subhalos_manifest.pkl"


def _write_reconciling_subhalos_manifest(
    *,
    shard_root: Path,
    snapshot_file: str,
    ahf_particles_file: str,
    output_file: str,
    snapshot_hash: Optional[str],
    fof_ll: float,
    fof_vel_ll: Optional[float],
    min_stars: int,
    root_payload_paths: Dict[int, str],
    root_to_shards: Dict[int, List[str]],
    root_costs: Dict[int, int],
) -> Path:
    path = _reconciling_subhalos_manifest_path(shard_root)
    _dump_pickle(
        path,
        {
            "snapshot_file": str(snapshot_file),
            "ahf_particles_file": str(ahf_particles_file),
            "output_file": str(output_file),
            "snapshot_hash": None if snapshot_hash in (None, "") else str(snapshot_hash),
            "fof_ll": float(fof_ll),
            "fof_vel_ll": fof_vel_ll,
            "min_stars": int(min_stars),
            "root_payload_paths": {int(k): str(v) for k, v in root_payload_paths.items()},
            "root_to_shards": {int(k): [str(x) for x in v] for k, v in root_to_shards.items()},
            "root_costs": {int(k): int(v) for k, v in root_costs.items()},
        },
    )
    return path


def _load_reconciling_subhalos_manifest(shard_root: Path) -> Dict[str, object]:
    path = _reconciling_subhalos_manifest_path(shard_root)
    if not path.is_file():
        raise RuntimeError(f"Reconciliation manifest not found: {path}")
    payload = _load_pickle(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid reconciliation manifest payload: {path}")
    return payload


def _calculating_properties_manifest_path(shard_root: Path) -> Path:
    return shard_root / "calculating_properties_manifest.pkl"


def _write_calculating_properties_manifest(
    *,
    shard_root: Path,
    snapshot_file: str,
    ahf_particles_file: str,
    output_file: str,
    final_shards: Sequence[str],
    root_results: Optional[Sequence[Mapping[str, object]]] = None,
    snapshot_hash: Optional[str] = None,
) -> Path:
    path = _calculating_properties_manifest_path(shard_root)
    _dump_pickle(
        path,
        {
            "snapshot_file": str(snapshot_file),
            "ahf_particles_file": str(ahf_particles_file),
            "output_file": str(output_file),
            "final_shards": [str(v) for v in final_shards],
            "root_results": [
                {
                    "root_id": int(rec["root_id"]),
                    "shard_path": str(rec["shard_path"]),
                    "root_payload_path": str(rec["root_payload_path"]),
                    "store_id": str(rec.get("store_id", "")),
                    "particle_store_path": str(rec.get("particle_store_path", "")),
                    "node_store_path": str(rec.get("node_store_path", "")),
                    "count": int(rec.get("count", 0)),
                    "cost": int(rec.get("cost", 0)),
                    "halo_count": int(rec.get("halo_count", 0)),
                    "property_cost": int(rec.get("property_cost", 0)),
                }
                for rec in (root_results or [])
            ],
            "snapshot_hash": None if snapshot_hash in (None, "") else str(snapshot_hash),
        },
    )
    return path


def _load_calculating_properties_manifest(shard_root: Path) -> Dict[str, object]:
    path = _calculating_properties_manifest_path(shard_root)
    if not path.is_file():
        raise RuntimeError(f"Calculating-properties manifest not found: {path}")
    payload = _load_pickle(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid calculating-properties manifest payload: {path}")
    return payload


def _cleanup_intermediate_shards(*, shard_root: Path, log_fn=None) -> None:
    root = Path(shard_root)
    if not root.exists():
        return
    if log_fn is not None:
        log_fn(f"cleanup: removing intermediate files from {root}")
    try:
        shutil.rmtree(root)
    except Exception as exc:
        if log_fn is not None:
            log_fn(f"cleanup: warning; failed to remove intermediate files from {root}: {exc}")
        return
    if log_fn is not None:
        log_fn("cleanup: complete")


def run_mpi(
    *,
    snapshot_file: str,
    ahf_particles_file: str,
    output_file: str,
    nproc: int = 1,
    role: str = "auto",
    min_stars: Optional[int] = None,
    shard_dir: Optional[str] = None,
    phase: str = "finding_galaxies",
    cleanup: bool = True,
) -> None:
    if MPI is None:
        raise RuntimeError("mpi4py is required for AHF-subhalo MPI execution")

    phase = str(phase)
    if phase not in {"finding_galaxies", "reconciling_subhalos", "calculating_properties", "grouping_pipeline", "pipeline"}:
        raise ValueError(f"Unsupported AHF-subhalo MPI phase: {phase}")

    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    size = int(comm.Get_size())
    if size < 2:
        raise RuntimeError("AHF-subhalo MPI requires at least 2 ranks (1 coordinator + workers)")

    role = str(role)
    if phase in {"finding_galaxies", "grouping_pipeline", "pipeline"}:
        valid_roles = {"coordinator", "gpu_worker", "cpu_worker", "auto"}
    elif phase == "reconciling_subhalos":
        valid_roles = {"coordinator", "cpu_worker", "auto"}
    else:
        valid_roles = {"coordinator", "property_worker", "auto"}
    if role not in valid_roles:
        raise ValueError(f"Unsupported AHF-subhalo MPI role for {phase}: {role}")

    declared_roles = comm.allgather(role)
    if "coordinator" in declared_roles:
        if declared_roles.count("coordinator") != 1:
            raise RuntimeError("Exactly one MPI rank must use --role coordinator")
        coordinator_rank = int(declared_roles.index("coordinator"))
    else:
        coordinator_rank = 0
    if int(rank) == int(coordinator_rank):
        _reset_rank0_log_context()

    effective_role = role
    if role == "auto":
        if int(rank) == int(coordinator_rank):
            effective_role = "coordinator"
        elif phase in {"finding_galaxies", "grouping_pipeline", "pipeline"}:
            effective_role = "gpu_worker" if _available_gpu_device_ids() else "cpu_worker"
        elif phase == "reconciling_subhalos":
            effective_role = "cpu_worker"
        else:
            effective_role = "property_worker"

    world_layout = comm.allgather(
        {
            "rank": int(rank),
            "role": str(effective_role),
            "hostname": os.uname().nodename,
            "local_rank": int(_local_rank()),
            "visible_cores": int(_local_visible_cores()),
        }
    )

    shard_root = Path(shard_dir) if shard_dir else Path(tempfile.mkdtemp(prefix="ahf_subhalo_mpi_"))
    if int(rank) == int(coordinator_rank):
        shard_root.mkdir(parents=True, exist_ok=True)
    shard_root = Path(comm.bcast(str(shard_root), root=coordinator_rank))

    if phase in {"finding_galaxies", "grouping_pipeline", "pipeline"}:
        if effective_role == "coordinator":
            stage_label = "grouping pipeline" if phase == "grouping_pipeline" else "finding galaxies"
            coordinator_label = "pipeline" if phase == "pipeline" else stage_label
            _rank0_log(f"{coordinator_label}: coordinator starting on shard_root={shard_root}")
            worker_ranks = [i for i in range(size) if int(i) != int(coordinator_rank)]
            worker_caps = [comm.recv(source=i, tag=MSG_REGISTER) for i in worker_ranks]
            worker_caps = [WorkerCapability(**cap) if isinstance(cap, dict) else cap for cap in worker_caps]
            gpu_worker_count = sum(1 for cap in worker_caps if cap.gpu_device is not None)
            cpu_worker_count = len(worker_caps) - gpu_worker_count
            _rank0_log(
                f"{stage_label}: workers registered "
                f"gpu={gpu_worker_count}, cpu={cpu_worker_count}, total={len(worker_caps)}"
            )

            _rank0_log(f"{stage_label}: preparing runtime, task manifest, and per-node payloads")
            (
                sim_runtime,
                pid_maps_sel,
                min_stars_val,
                fof_ll,
                fof_vel_ll,
                tasks,
                tasks_by_root,
                task_payloads_by_node,
                store_manifest,
            ) = _rank0_prepare_galaxy_finding(
                snapshot_file=snapshot_file,
                ahf_particles_file=ahf_particles_file,
                shard_root=shard_root,
                nproc=int(nproc),
                min_stars=min_stars,
                log_fn=_rank0_log,
                log_label=stage_label,
            )
            _rank0_log(
                f"{stage_label}: preparation complete "
                f"tasks={len(tasks)}, roots={len(tasks_by_root)}, min_stars={int(min_stars_val)}"
            )
            root_store_lookup = {
                int(k): dict(v)
                for k, v in dict(store_manifest.get("roots", {})).items()
            }

            regular_batches, small_batches = _classify_galaxy_finding_batches(
                tasks=tasks,
                gpu_worker_count=int(gpu_worker_count),
                cpu_worker_count=int(cpu_worker_count),
            )
            finding_galaxies_thread_map = _build_finding_galaxies_thread_map(
                worker_caps=worker_caps,
                world_layout=world_layout,
                coordinator_rank=coordinator_rank,
            )
            _log_thread_map("finding galaxies", worker_caps, finding_galaxies_thread_map)
            _rank0_log(
                "finding galaxies: classified batches "
                f"regular={len(regular_batches)}, small={len(small_batches)}, total={len(regular_batches) + len(small_batches)}"
            )
            finding_galaxies_regular_specs = [
                {"batch_index": idx, "batch": batch}
                for idx, batch in enumerate(regular_batches)
            ]
            finding_galaxies_small_specs = [
                {"batch_index": idx, "batch": batch}
                for idx, batch in enumerate(small_batches)
            ]

            def _materialize_finding_galaxies_regular_spec(spec, direction):
                batch_store_ids = {
                    str(root_store_lookup[int(task.top_id)]["store_id"])
                    for task in spec["batch"].tasks
                    if int(task.top_id) in root_store_lookup
                }
                return _materialize_galaxy_finding_batch_item(
                    shard_root=shard_root,
                    batch=spec["batch"],
                    prefix="regular",
                    batch_index=int(spec["batch_index"]),
                    task_payloads_by_node=task_payloads_by_node,
                    fof_ll=float(fof_ll),
                    fof_vel_ll=fof_vel_ll,
                    min_stars=int(min_stars_val),
                    backend="cpu",
                    device_id=None,
                    store_ids=tuple(sorted(batch_store_ids)),
                )

            def _materialize_finding_galaxies_small_spec(spec, direction):
                batch_store_ids = {
                    str(root_store_lookup[int(task.top_id)]["store_id"])
                    for task in spec["batch"].tasks
                    if int(task.top_id) in root_store_lookup
                }
                return _materialize_galaxy_finding_batch_item(
                    shard_root=shard_root,
                    batch=spec["batch"],
                    prefix="small",
                    batch_index=int(spec["batch_index"]),
                    task_payloads_by_node=task_payloads_by_node,
                    fof_ll=float(fof_ll),
                    fof_vel_ll=fof_vel_ll,
                    min_stars=int(min_stars_val),
                    backend="cpu",
                    device_id=None,
                    store_ids=tuple(sorted(batch_store_ids)),
                )

            def _bind_finding_galaxies_item(base_item, *, cap, direction):
                item = dict(base_item)
                backend = "gpu" if str(cap.role) == "gpu_worker" else "cpu"
                item["backend"] = backend
                item["device_id"] = cap.gpu_device if backend == "gpu" else None
                return item

            buffer_scale = max(1, _env_int("CAESAR_AHF_SUBHALO_MPI_STAGE1_BUFFER_SCALE", 4))
            finding_galaxies_cpu_capacity = sum(
                _cpu_local_workers(cap, stage_name="finding_galaxies", threads=int(finding_galaxies_thread_map.get(int(cap.rank), cap.threads)))
                for cap in worker_caps
                if cap.gpu_device is None
            )
            finding_galaxies_regular_queue = _BufferedPreparedQueue(
                specs=finding_galaxies_regular_specs,
                prepare_fn=_materialize_finding_galaxies_regular_spec,
                total=len(finding_galaxies_regular_specs),
                front_target=max(0, int(gpu_worker_count) * int(buffer_scale)),
                back_target=max(0, int(finding_galaxies_cpu_capacity) * int(buffer_scale)),
                max_workers=max(1, min(8, int(gpu_worker_count) + int(cpu_worker_count))),
            )
            finding_galaxies_small_queue = _BufferedPreparedQueue(
                specs=finding_galaxies_small_specs,
                prepare_fn=_materialize_finding_galaxies_small_spec,
                total=len(finding_galaxies_small_specs),
                front_target=max(0, int(finding_galaxies_cpu_capacity) * int(buffer_scale)),
                back_target=0,
                max_workers=max(1, min(8, int(cpu_worker_count))),
            )

            finding_galaxies_results = _dispatch_stage(
                comm,
                worker_caps=worker_caps,
                regular_queue=finding_galaxies_regular_queue,
                small_queue=finding_galaxies_small_queue,
                stage_name="finding_galaxies",
                display_name="finding galaxies",
                prepare_regular=_bind_finding_galaxies_item,
                prepare_small=_bind_finding_galaxies_item,
                regular_total=len(finding_galaxies_regular_specs),
                small_total=len(finding_galaxies_small_specs),
                worker_threads=finding_galaxies_thread_map,
                progress_unit="batches",
                progress_metrics=[
                    ProgressMetric("nodes", "count_nodes", total=len(tasks)),
                ],
            )

            root_to_shards: Dict[int, List[str]] = {int(root): [] for root in tasks_by_root}
            for result in finding_galaxies_results:
                for root_id in result["root_ids"]:
                    root_to_shards[int(root_id)].append(str(result["shard_path"]))
            _rank0_log(
                "finding galaxies: result aggregation complete "
                f"roots_with_shards={sum(1 for paths in root_to_shards.values() if paths)}"
            )

            _rank0_log(f"reconciling subhalos: writing root payloads for {len(tasks_by_root)} roots")
            root_payload_paths = _write_reconciliation_root_payloads(
                shard_root=shard_root,
                tasks_by_root=tasks_by_root,
                store_manifest=store_manifest,
                direct_state=sim_runtime if isinstance(sim_runtime, AHFSubhaloDirectState) else None,
                log_fn=_rank0_log,
                progress_label="reconciling subhalos",
            )
            root_costs = {
                int(root_id): int(sum(int(task.fof_candidates) for task in root_tasks))
                for root_id, root_tasks in tasks_by_root.items()
            }
            _write_reconciling_subhalos_manifest(
                shard_root=shard_root,
                snapshot_file=snapshot_file,
                ahf_particles_file=ahf_particles_file,
                output_file=output_file,
                snapshot_hash=getattr(sim_runtime, "hash", None),
                fof_ll=float(fof_ll),
                fof_vel_ll=fof_vel_ll,
                min_stars=int(min_stars_val),
                root_payload_paths=root_payload_paths,
                root_to_shards=root_to_shards,
                root_costs=root_costs,
            )
            _rank0_log("finding galaxies: wrote reconciliation manifest")

            if phase == "finding_galaxies":
                _stop_workers(comm, worker_caps=worker_caps)
                return

            if phase == "pipeline":
                reconciling_subhalos_regular_roots = [
                    int(root_id)
                    for root_id, _cost in sorted(root_costs.items(), key=lambda kv: int(kv[1]), reverse=True)
                ]
                reconciling_subhalos_small_roots = []
                reconciling_subhalos_thread_map = _build_uniform_stage_thread_map(
                    worker_caps=worker_caps,
                    world_layout=world_layout,
                    coordinator_rank=coordinator_rank,
                )
                reconciling_subhalos_worker_roles = {int(cap.rank): "cpu_worker" for cap in worker_caps}
            else:
                reconciling_subhalos_regular_roots, reconciling_subhalos_small_roots = _classify_reconciling_subhalos_roots(
                    root_costs=root_costs,
                    gpu_worker_count=int(gpu_worker_count),
                    cpu_worker_count=int(cpu_worker_count),
                )
                reconciling_subhalos_thread_map = {int(cap.rank): int(cap.threads) for cap in worker_caps}
                reconciling_subhalos_worker_roles = {int(cap.rank): str(cap.role) for cap in worker_caps}
            _log_thread_map("reconciling subhalos", worker_caps, reconciling_subhalos_thread_map)
            _rank0_log(
                "reconciling subhalos: classified roots "
                f"regular={len(reconciling_subhalos_regular_roots)}, small={len(reconciling_subhalos_small_roots)}, total={len(reconciling_subhalos_regular_roots) + len(reconciling_subhalos_small_roots)}"
            )

            def _prepare_reconciling_subhalos_root(root_id, *, cap, direction):
                return {
                    "root_id": int(root_id),
                    "shard_paths": root_to_shards[int(root_id)],
                    "root_payload_path": root_payload_paths[int(root_id)],
                    "backend": "cpu",
                    "device_id": None,
                    "fof_ll": float(fof_ll),
                    "fof_vel_ll": fof_vel_ll,
                    "min_stars": int(min_stars_val),
                }

            reconciling_subhalos_results = _dispatch_stage(
                comm,
                worker_caps=worker_caps,
                regular_queue=reconciling_subhalos_regular_roots,
                small_queue=reconciling_subhalos_small_roots,
                stage_name="reconciling_subhalos",
                display_name="reconciling subhalos",
                prepare_regular=_prepare_reconciling_subhalos_root,
                prepare_small=_prepare_reconciling_subhalos_root,
                regular_total=len(reconciling_subhalos_regular_roots),
                small_total=len(reconciling_subhalos_small_roots),
                worker_threads=reconciling_subhalos_thread_map,
                worker_roles=reconciling_subhalos_worker_roles,
                progress_unit="roots",
                progress_metrics=[
                    ProgressMetric("galaxies_out", "count"),
                ],
            )
            root_results = []
            for result in reconciling_subhalos_results:
                root_results.extend(list(result.get("root_results", [])))
            root_results = sorted(root_results, key=lambda rec: int(rec["root_id"]))
            store_roots = {
                int(k): dict(v)
                for k, v in dict(store_manifest.get("roots", {})).items()
            }
            root_results_enriched = [
                {
                    "root_id": int(rec["root_id"]),
                    "shard_path": str(rec["shard_path"]),
                    "root_payload_path": str(root_payload_paths[int(rec["root_id"])]),
                    "store_id": str(store_roots[int(rec["root_id"])]["store_id"]),
                    "particle_store_path": str(store_roots[int(rec["root_id"])]["particle_store_path"]),
                    "node_store_path": str(store_roots[int(rec["root_id"])]["node_store_path"]),
                    "count": int(rec.get("count", 0)),
                    "cost": int(root_costs.get(int(rec["root_id"]), 0)),
                    "halo_count": int(rec.get("halo_count", 0)),
                    "property_cost": int(rec.get("property_cost", 0)),
                }
                for rec in root_results
            ]
            final_shards = [str(rec["shard_path"]) for rec in root_results_enriched]
            _rank0_log(f"reconciling subhalos: complete; final_shards={len(final_shards)}")

            if phase == "pipeline":
                calculating_properties_thread_map = _build_uniform_stage_thread_map(
                    worker_caps=worker_caps,
                    world_layout=world_layout,
                    coordinator_rank=coordinator_rank,
                )
                _log_thread_map("calculating properties", worker_caps, calculating_properties_thread_map)
                snapshot_meta, property_results = _rank0_calculate_properties(
                    comm,
                    worker_caps=worker_caps,
                    snapshot_file=snapshot_file,
                    ahf_particles_file=ahf_particles_file,
                    output_file=output_file,
                    root_results=root_results_enriched,
                    nproc=int(nproc),
                    shard_root=shard_root,
                    snapshot_hash=getattr(sim_runtime, "hash", None),
                    sim=sim_runtime,
                    pid_maps_sel=pid_maps_sel,
                    worker_threads=calculating_properties_thread_map,
                )
                _stop_workers(comm, worker_caps=worker_caps)
                _rank0_write_catalogue(
                    snapshot_meta=snapshot_meta,
                    property_results=property_results,
                    output_file=output_file,
                )
                if bool(cleanup):
                    _cleanup_intermediate_shards(shard_root=shard_root, log_fn=_rank0_log)
                return

            _stop_workers(comm, worker_caps=worker_caps)
            _write_calculating_properties_manifest(
                shard_root=shard_root,
                snapshot_file=snapshot_file,
                ahf_particles_file=ahf_particles_file,
                output_file=output_file,
                final_shards=final_shards,
                root_results=root_results_enriched,
                snapshot_hash=getattr(sim_runtime, "hash", None),
            )
            _rank0_log("grouping pipeline: wrote property manifest")
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
            work_start = time.monotonic()
            if stage == "finding_galaxies":
                result = _worker_run_finding_galaxies(
                    item=item,
                    fof_ll=float(item["fof_ll"]),
                    fof_vel_ll=item.get("fof_vel_ll"),
                    min_stars=int(item["min_stars"]),
                    shard_dir=shard_root,
                    nproc=int(cap.threads),
                )
            elif stage == "reconciling_subhalos":
                result = _worker_run_reconciling_subhalos(
                    item=item,
                    fof_ll=float(item["fof_ll"]),
                    fof_vel_ll=item.get("fof_vel_ll"),
                    min_stars=int(item["min_stars"]),
                    shard_dir=shard_root,
                    nproc=int(cap.threads),
                )
            elif stage == "calculating_properties":
                result = _worker_run_calculating_properties(
                    item=item,
                    shard_dir=shard_root,
                    nproc=int(cap.threads),
                )
            else:
                raise RuntimeError(f"Unknown MPI stage: {stage}")

            result = dict(result)
            result["elapsed_seconds"] = float(time.monotonic() - work_start)
            comm.send(result, dest=coordinator_rank, tag=MSG_RESULT)
        return

    if phase == "reconciling_subhalos":
        if effective_role == "coordinator":
            _rank0_log(f"reconciling subhalos: coordinator starting on shard_root={shard_root}")
            worker_ranks = [i for i in range(size) if int(i) != int(coordinator_rank)]
            worker_caps = [comm.recv(source=i, tag=MSG_REGISTER) for i in worker_ranks]
            worker_caps = [WorkerCapability(**cap) if isinstance(cap, dict) else cap for cap in worker_caps]
            cpu_worker_count = len(worker_caps)
            _rank0_log(
                "reconciling subhalos: workers registered "
                f"cpu={cpu_worker_count}, total={len(worker_caps)}"
            )

            manifest = _load_reconciling_subhalos_manifest(shard_root)
            root_payload_paths = {
                int(k): str(v) for k, v in dict(manifest.get("root_payload_paths", {})).items()
            }
            root_to_shards = {
                int(k): [str(x) for x in v]
                for k, v in dict(manifest.get("root_to_shards", {})).items()
            }
            root_costs = {
                int(k): int(v) for k, v in dict(manifest.get("root_costs", {})).items()
            }
            reconciling_subhalos_regular_roots, reconciling_subhalos_small_roots = _classify_reconciling_subhalos_roots(
                root_costs=root_costs,
                gpu_worker_count=0,
                cpu_worker_count=int(cpu_worker_count),
            )
            reconciling_subhalos_thread_map = _build_uniform_stage_thread_map(
                worker_caps=worker_caps,
                world_layout=world_layout,
                coordinator_rank=coordinator_rank,
            )
            _log_thread_map("reconciling subhalos", worker_caps, reconciling_subhalos_thread_map)
            _rank0_log(
                "reconciling subhalos: classified roots "
                f"regular={len(reconciling_subhalos_regular_roots)}, small={len(reconciling_subhalos_small_roots)}, total={len(reconciling_subhalos_regular_roots) + len(reconciling_subhalos_small_roots)}"
            )

            def _prepare_reconciling_subhalos_root(root_id, *, cap, direction):
                return {
                    "root_id": int(root_id),
                    "shard_paths": root_to_shards[int(root_id)],
                    "root_payload_path": root_payload_paths[int(root_id)],
                    "backend": "cpu",
                    "device_id": None,
                    "fof_ll": float(manifest.get("fof_ll", 0.0)),
                    "fof_vel_ll": manifest.get("fof_vel_ll"),
                    "min_stars": int(manifest.get("min_stars", min_stars or 0)),
                }

            reconciling_subhalos_results = _dispatch_stage(
                comm,
                worker_caps=worker_caps,
                regular_queue=reconciling_subhalos_regular_roots,
                small_queue=reconciling_subhalos_small_roots,
                stage_name="reconciling_subhalos",
                display_name="reconciling subhalos",
                prepare_regular=_prepare_reconciling_subhalos_root,
                prepare_small=_prepare_reconciling_subhalos_root,
                regular_total=len(reconciling_subhalos_regular_roots),
                small_total=len(reconciling_subhalos_small_roots),
                worker_threads=reconciling_subhalos_thread_map,
                worker_roles={int(cap.rank): "cpu_worker" for cap in worker_caps},
                progress_unit="roots",
                progress_metrics=[
                    ProgressMetric("galaxies_out", "count"),
                ],
            )
            _stop_workers(comm, worker_caps=worker_caps)
            root_results = []
            for result in reconciling_subhalos_results:
                root_results.extend(list(result.get("root_results", [])))
            root_results = sorted(root_results, key=lambda rec: int(rec["root_id"]))
            root_payload_meta = {
                int(root_id): dict(_load_pickle(Path(str(path))))
                for root_id, path in root_payload_paths.items()
            }
            root_results_enriched = [
                {
                    "root_id": int(rec["root_id"]),
                    "shard_path": str(rec["shard_path"]),
                    "root_payload_path": str(root_payload_paths[int(rec["root_id"])]),
                    "store_id": str(root_payload_meta[int(rec["root_id"])].get("store_id", "")),
                    "particle_store_path": str(root_payload_meta[int(rec["root_id"])].get("particle_store_path", "")),
                    "node_store_path": str(root_payload_meta[int(rec["root_id"])].get("node_store_path", "")),
                    "count": int(rec.get("count", 0)),
                    "cost": int(root_costs.get(int(rec["root_id"]), 0)),
                    "halo_count": int(rec.get("halo_count", 0)),
                    "property_cost": int(rec.get("property_cost", 0)),
                }
                for rec in root_results
            ]
            final_shards = [str(rec["shard_path"]) for rec in root_results_enriched]
            _rank0_log(f"reconciling subhalos: complete; final_shards={len(final_shards)}")
            _write_calculating_properties_manifest(
                shard_root=shard_root,
                snapshot_file=str(manifest.get("snapshot_file", snapshot_file)),
                ahf_particles_file=str(manifest.get("ahf_particles_file", ahf_particles_file)),
                output_file=str(manifest.get("output_file", output_file)),
                final_shards=final_shards,
                root_results=root_results_enriched,
                snapshot_hash=manifest.get("snapshot_hash"),
            )
            _rank0_log("reconciling subhalos: wrote property manifest")
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
            if str(msg["stage"]) != "reconciling_subhalos":
                raise RuntimeError(f"Unexpected reconciling-subhalos MPI message: {msg}")
            item = msg["item"]
            work_start = time.monotonic()
            result = _worker_run_reconciling_subhalos(
                item=item,
                fof_ll=float(item["fof_ll"]),
                fof_vel_ll=item.get("fof_vel_ll"),
                min_stars=int(item["min_stars"]),
                shard_dir=shard_root,
                nproc=int(cap.threads),
            )
            result = dict(result)
            result["elapsed_seconds"] = float(time.monotonic() - work_start)
            comm.send(result, dest=coordinator_rank, tag=MSG_RESULT)
        return

    if effective_role == "coordinator":
        worker_ranks = [i for i in range(size) if int(i) != int(coordinator_rank)]
        worker_caps = [comm.recv(source=i, tag=MSG_REGISTER) for i in worker_ranks]
        worker_caps = [WorkerCapability(**cap) if isinstance(cap, dict) else cap for cap in worker_caps]
        manifest = _load_calculating_properties_manifest(shard_root)
        calculating_properties_thread_map = _build_uniform_stage_thread_map(
            worker_caps=worker_caps,
            world_layout=world_layout,
            coordinator_rank=coordinator_rank,
        )
        _log_thread_map("calculating properties", worker_caps, calculating_properties_thread_map)
        snapshot_meta, property_results = _rank0_calculate_properties(
            comm,
            worker_caps=worker_caps,
            snapshot_file=str(manifest.get("snapshot_file", snapshot_file)),
            ahf_particles_file=str(manifest.get("ahf_particles_file", ahf_particles_file)),
            output_file=str(manifest.get("output_file", output_file)),
            root_results=[dict(v) for v in manifest.get("root_results", [])],
            nproc=int(nproc),
            shard_root=shard_root,
            snapshot_hash=manifest.get("snapshot_hash"),
            worker_threads=calculating_properties_thread_map,
        )
        _stop_workers(comm, worker_caps=worker_caps)
        _rank0_write_catalogue(
            snapshot_meta=snapshot_meta,
            property_results=property_results,
            output_file=str(manifest.get("output_file", output_file)),
        )
        if bool(cleanup):
            _cleanup_intermediate_shards(shard_root=shard_root, log_fn=_rank0_log)
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
        if str(msg["stage"]) != "calculating_properties":
            raise RuntimeError(f"Unexpected calculating-properties MPI message: {msg}")
        work_start = time.monotonic()
        result = _worker_run_calculating_properties(item=msg["item"], shard_dir=shard_root, nproc=int(cap.threads))
        result = dict(result)
        result["elapsed_seconds"] = float(time.monotonic() - work_start)
        comm.send(result, dest=coordinator_rank, tag=MSG_RESULT)


def main():
    parser = argparse.ArgumentParser(description="MPI runtime for CAESAR AHF-subhalo mode")
    parser.add_argument("snapshot", type=str, help="Path to snapshot file")
    parser.add_argument("ahf", type=str, help="Path to AHF_particles file")
    parser.add_argument("out", type=str, help="Path to output CAESAR file")
    parser.add_argument(
        "--phase",
        type=str,
        default="pipeline",
        choices=("finding_galaxies", "reconciling_subhalos", "calculating_properties", "grouping_pipeline", "pipeline"),
        help="MPI phase to run: finding_galaxies, reconciling_subhalos, calculating_properties, grouping_pipeline, or persistent single-job pipeline",
    )
    parser.add_argument(
        "--role",
        type=str,
        default="auto",
        choices=("auto", "coordinator", "gpu_worker", "cpu_worker", "property_worker"),
        help="MPI rank role; finding_galaxies/grouping_pipeline use coordinator/gpu_worker/cpu_worker, calculating_properties uses coordinator/property_worker",
    )
    parser.add_argument("--nproc", type=int, default=1, help="Per-rank CAESAR nproc/thread budget")
    parser.add_argument("--min-stars", type=int, default=None, help="Minimum stars per galaxy")
    parser.add_argument("--shard-dir", type=str, default=None, help="Directory for intermediate shard files")
    parser.add_argument(
        "--cleanup",
        dest="cleanup",
        action="store_true",
        default=True,
        help="Delete intermediate shard files after a successful final export (default)",
    )
    parser.add_argument(
        "--no-cleanup",
        dest="cleanup",
        action="store_false",
        help="Keep intermediate shard files after a successful final export",
    )
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
        cleanup=bool(args.cleanup),
    )


if __name__ == "__main__":
    main()
