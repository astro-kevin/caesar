"""AHF-FAST-specific galaxy construction entry point.

This module holds the full implementation of the AHF-FAST galaxy builder
so that the fast path can evolve independently of the generic matching
utilities in :mod:`caesar.halo_matching`.  The :mod:`halo_matching`
module re-exports a thin wrapper for backwards compatibility, but the
source of truth lives here.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator


_GIB = float(2**30)


@dataclass
class HostSchedState:
    order_idx: int
    root_id: int
    nodes: Set[int]
    fof_candidates: int
    star_count: int
    is_heavy: bool = False
    work_units: int = 1
    predicted_bytes: int = 0
    queue_index: int = 0
    submit_ts: float = 0.0
    submit_inflight: int = 0
    submit_rss_bytes: int = -1
    is_tiny_proxy: bool = False


@dataclass
class _MemModel:
    knots_x: np.ndarray
    knots_y: np.ndarray
    interp: PchipInterpolator


def _ahf_fast_clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def _ahf_fast_weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    tau: float,
) -> float:
    if values.size == 0:
        return 0.0
    tau = float(_ahf_fast_clamp(tau, 0.01, 0.99))
    order = np.argsort(values)
    v = np.asarray(values[order], dtype=np.float64)
    w = np.asarray(weights[order], dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 1.0)
    cdf = np.cumsum(w)
    total = float(cdf[-1])
    if total <= 0.0:
        return float(v[-1])
    target = tau * total
    idx = int(np.searchsorted(cdf, target, side="left"))
    idx = max(0, min(idx, v.size - 1))
    return float(v[idx])


def _ahf_fast_local_weighted_quantile(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    center: float,
    neighbors: int,
    tau: float,
) -> float:
    if x.size == 0:
        return 0.0
    k = max(1, min(int(neighbors), x.size))
    dist = np.abs(x - float(center))
    idx = np.argpartition(dist, kth=(k - 1))[:k]
    local_x = x[idx]
    local_y = y[idx]
    local_w = sample_weight[idx]

    max_d = float(np.max(np.abs(local_x - float(center))))
    if max_d <= 0.0:
        kern = np.ones_like(local_w, dtype=np.float64)
    else:
        # Tri-cube kernel for locality without assuming any global scaling law.
        u = np.abs(local_x - float(center)) / max_d
        kern = np.power(np.clip(1.0 - np.power(u, 3.0), 0.0, None), 3.0)
    w = np.asarray(local_w, dtype=np.float64) * np.asarray(kern, dtype=np.float64)
    return _ahf_fast_weighted_quantile(local_y, w, tau)


def _ahf_fast_fit_mem_model(
    *,
    sample_x: Sequence[float],
    sample_y: Sequence[float],
    sample_w: Sequence[float],
    tau: float,
    knots: int,
    neighbors: int,
) -> Optional[_MemModel]:
    if not sample_x or not sample_y:
        return None
    x = np.asarray(sample_x, dtype=np.float64)
    y = np.asarray(sample_y, dtype=np.float64)
    w = np.asarray(sample_w, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (y > 0.0)
    if np.count_nonzero(valid) < 2:
        return None
    x = x[valid]
    y = y[valid]
    w = np.where(w[valid] > 0.0, w[valid], 1.0)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    w = w[order]

    n_knots = max(2, min(int(knots), x.size))
    knot_q = np.linspace(0.0, 1.0, n_knots)
    knots_x = np.quantile(x, knot_q)
    knots_x = np.unique(np.asarray(knots_x, dtype=np.float64))
    if knots_x.size < 2:
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        if xmax <= xmin:
            xmax = xmin + 1.0e-6
        knots_x = np.asarray([xmin, xmax], dtype=np.float64)

    knots_y = np.asarray(
        [
            _ahf_fast_local_weighted_quantile(
                x=x,
                y=y,
                sample_weight=w,
                center=float(kx),
                neighbors=int(neighbors),
                tau=float(tau),
            )
            for kx in knots_x
        ],
        dtype=np.float64,
    )
    knots_y = np.maximum.accumulate(knots_y)
    interp = PchipInterpolator(knots_x, knots_y, extrapolate=True)
    return _MemModel(knots_x=knots_x, knots_y=knots_y, interp=interp)


def _ahf_fast_fit_direct_monotone_model(
    *,
    sample_x: Sequence[float],
    sample_y: Sequence[float],
    sample_w: Sequence[float],
) -> Optional[_MemModel]:
    if not sample_x or not sample_y:
        return None
    x = np.asarray(sample_x, dtype=np.float64)
    y = np.asarray(sample_y, dtype=np.float64)
    w = np.asarray(sample_w, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (y > 0.0)
    if np.count_nonzero(valid) < 2:
        return None
    x = x[valid]
    y = y[valid]
    w = np.where(w[valid] > 0.0, w[valid], 1.0)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    w = w[order]

    knots_x: List[float] = []
    knots_y: List[float] = []
    start = 0
    while start < x.size:
        end = start + 1
        while end < x.size and x[end] == x[start]:
            end += 1
        weights = w[start:end]
        values = y[start:end]
        denom = float(np.sum(weights))
        if denom <= 0.0:
            y_val = float(np.mean(values))
        else:
            y_val = float(np.sum(values * weights) / denom)
        knots_x.append(float(x[start]))
        knots_y.append(y_val)
        start = end

    if len(knots_x) < 2:
        return None
    knots_x_arr = np.asarray(knots_x, dtype=np.float64)
    knots_y_arr = np.maximum.accumulate(np.asarray(knots_y, dtype=np.float64))
    interp = PchipInterpolator(knots_x_arr, knots_y_arr, extrapolate=True)
    return _MemModel(knots_x=knots_x_arr, knots_y=knots_y_arr, interp=interp)


def _ahf_fast_seed_prediction_bytes(
    *,
    fof_candidates: int,
    work_unit: int,
    base_unit_bytes: int,
    pred_min_bytes: int,
    pred_max_bytes: int,
) -> int:
    scaled = float(max(1, int(fof_candidates))) / float(max(1, int(work_unit)))
    seed = float(max(1, int(base_unit_bytes))) * float(np.sqrt(max(1.0, scaled)))
    return int(
        round(
            _ahf_fast_clamp(
                seed,
                float(max(1, int(pred_min_bytes))),
                float(max(int(pred_min_bytes), int(pred_max_bytes))),
            )
        )
    )


def _ahf_fast_predict_reservation_bytes(
    *,
    fof_candidates: int,
    model: Optional[_MemModel],
    seed_bytes: int,
    pred_min_bytes: int,
    pred_max_bytes: int,
) -> int:
    predicted = float(seed_bytes)
    if model is not None:
        x = float(np.log1p(max(0, int(fof_candidates))))
        predicted = float(model.interp(x))
    return int(
        round(
            _ahf_fast_clamp(
                predicted,
                float(max(1, int(pred_min_bytes))),
                float(max(int(pred_min_bytes), int(pred_max_bytes))),
            )
        )
    )


def _ahf_fast_seed_anchor_samples(
    *,
    fof_candidates: Sequence[int],
    work_unit: int,
    base_unit_bytes: int,
    pred_min_bytes: int,
    pred_max_bytes: int,
    anchor_count: int,
    anchor_weight: float,
) -> Tuple[List[float], List[float], List[float]]:
    if not fof_candidates or int(anchor_count) <= 0 or float(anchor_weight) <= 0.0:
        return [], [], []

    values = np.asarray([max(1, int(v)) for v in fof_candidates], dtype=np.float64)
    if values.size == 0:
        return [], [], []

    quantiles = np.linspace(0.0, 1.0, max(2, int(anchor_count)))
    anchor_candidates = np.quantile(values, quantiles)
    anchor_candidates = np.asarray(np.maximum(1.0, np.rint(anchor_candidates)), dtype=np.int64)
    anchor_candidates = np.unique(anchor_candidates)
    if anchor_candidates.size < 2:
        return [], [], []

    anchor_x: List[float] = []
    anchor_y: List[float] = []
    anchor_w: List[float] = []
    for fof_count in anchor_candidates.tolist():
        anchor_x.append(float(np.log1p(max(0, int(fof_count)))))
        anchor_y.append(
            float(
                _ahf_fast_seed_prediction_bytes(
                    fof_candidates=int(fof_count),
                    work_unit=int(work_unit),
                    base_unit_bytes=int(base_unit_bytes),
                    pred_min_bytes=int(pred_min_bytes),
                    pred_max_bytes=int(pred_max_bytes),
                )
            )
        )
        anchor_w.append(float(anchor_weight))
    return anchor_x, anchor_y, anchor_w


def _ahf_fast_probe_fof_candidates(
    fof_candidates: Sequence[int],
    probe_count: int,
) -> List[int]:
    if not fof_candidates or int(probe_count) <= 0:
        return []
    values = np.asarray([max(1, int(v)) for v in fof_candidates], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, max(2, int(probe_count)))
    probes = np.quantile(values, quantiles)
    probes = np.asarray(np.maximum(1.0, np.rint(probes)), dtype=np.int64)
    return [int(v) for v in np.unique(probes)]


def _ahf_fast_select_bootstrap_hosts(
    pending_hosts: Sequence[HostSchedState],
    count: int,
) -> List[HostSchedState]:
    if int(count) <= 0:
        return []
    hosts = [host for host in pending_hosts if not bool(host.is_tiny_proxy)]
    hosts.sort(key=lambda h: (int(-h.fof_candidates), int(h.root_id)))
    return list(hosts[: max(0, int(count))])


def _ahf_fast_model_fit_samples(
    *,
    seed_anchor_x: Sequence[float],
    seed_anchor_y: Sequence[float],
    seed_anchor_w: Sequence[float],
    sample_x: Sequence[float],
    sample_y: Sequence[float],
    sample_w: Sequence[float],
    seed_drop_samples: int,
) -> Tuple[List[float], List[float], List[float]]:
    if len(sample_x) < int(seed_drop_samples):
        fit_x = list(seed_anchor_x)
        fit_y = list(seed_anchor_y)
        fit_w = list(seed_anchor_w)
        fit_x.extend(sample_x)
        fit_y.extend(sample_y)
        fit_w.extend(sample_w)
        return fit_x, fit_y, fit_w
    return list(sample_x), list(sample_y), list(sample_w)


def _ahf_fast_next_refit_target(
    *,
    current_target: int,
    exp_max: int,
) -> int:
    nxt = max(1, int(current_target)) * 2
    max_target = int(2 ** max(0, int(exp_max)))
    return int(min(nxt, max_target))


def _ahf_fast_mode_from_memory(
    *,
    avail_bytes: int,
    safety_floor_bytes: int,
    total_bytes: int,
) -> str:
    if int(avail_bytes) < 0:
        return "nominal"
    if int(avail_bytes) <= int(max(0, int(safety_floor_bytes))):
        return "critical"
    capacity = max(0, int(avail_bytes) - int(safety_floor_bytes))
    if int(total_bytes) > 0:
        low_cap = int(max(16.0 * _GIB, float(total_bytes) * 0.02))
        high_cap = int(max(96.0 * _GIB, float(total_bytes) * 0.22))
    else:
        low_cap = int(16.0 * _GIB)
        high_cap = int(96.0 * _GIB)
    if capacity <= low_cap:
        return "low"
    if capacity >= high_cap:
        return "high"
    return "nominal"


def _ahf_fast_select_best_fit_index(
    pending_hosts: Sequence[HostSchedState],
    scan_window: int,
    remaining_bytes: int,
) -> int:
    if not pending_hosts:
        return -1
    limit = max(1, min(int(scan_window), len(pending_hosts)))
    remaining = int(remaining_bytes)
    best_idx = -1
    best_pred = -1
    for idx in range(limit):
        if bool(pending_hosts[idx].is_tiny_proxy):
            continue
        predicted = int(max(0, int(pending_hosts[idx].predicted_bytes)))
        if predicted <= remaining and predicted > best_pred:
            best_idx = idx
            best_pred = predicted
    return int(best_idx)


def _ahf_fast_select_smallest_index(
    pending_hosts: Sequence[HostSchedState],
    scan_window: int,
) -> int:
    if not pending_hosts:
        return -1
    limit = max(1, min(int(scan_window), len(pending_hosts)))
    return int(
        min(
            range(limit),
            key=lambda i: int(max(0, int(pending_hosts[i].predicted_bytes))),
        )
    )


def _ahf_fast_observed_bytes_estimate(
    *,
    submit_inflight: int,
    rss_submit_bytes: int,
    rss_done_bytes: int,
) -> Optional[int]:
    delta = int(rss_done_bytes) - int(rss_submit_bytes)
    inflight = max(1, int(submit_inflight))
    return int(
        max(
            int(round(128.0 * 1024.0**2)),
            int(round(float(delta) / float(inflight))),
        )
    )


def _ahf_fast_sample_weight(
    *,
    submit_inflight: int,
    anchor_max_inflight: int,
    sample_max_inflight: int,
) -> float:
    inflight = max(1, int(submit_inflight))
    if inflight > int(sample_max_inflight):
        return 0.0
    if int(submit_inflight) <= 1:
        return 1.0
    if int(submit_inflight) <= int(anchor_max_inflight):
        return 0.5
    return float(1.0 / float(inflight))


def _ahf_fast_select_tiny_backfill_index(
    pending_hosts: Sequence[HostSchedState],
    scan_window: int,
) -> int:
    if not pending_hosts:
        return -1
    limit = max(1, min(int(scan_window), len(pending_hosts)))
    best_idx = -1
    best_key = None
    for idx in range(limit):
        host = pending_hosts[idx]
        if not bool(host.is_tiny_proxy):
            continue
        key = (
            int(max(0, int(host.fof_candidates))),
            int(max(0, int(host.predicted_bytes))),
            int(host.order_idx),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_idx = idx
    return int(best_idx)


def build_galaxies_from_ahf_fast(
    sim,
    ahf_particles_file: str,
    *,
    min_stars: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> None:
    """Build galaxies directly from AHF nodes using a star-count gate.

    Streaming host-by-host keeps memory bounded while joblib threads reuse
    the shared DataManager arrays for per-galaxy construction.
    """

    from yt.funcs import mylog
    import os as _os
    import json as _json
    import time as _time
    import threading as _threading
    from collections import deque as _deque
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

    from caesar.group import create_new_group
    from caesar.group import get_group_properties as _get_group_properties
    from caesar.property_manager import get_property, has_ptype
    from caesar.AHF_FAST_loader import (
        load_ahf_halos_dataframe,
        load_ahf_particle_blocks,
    )
    from caesar.fof6d import fof6d_halo as _fof6d_halo
    from caesar.fof6d import kernel_table as _fof6d_kernel_table
    from caesar.fubar import get_b as _get_b
    from caesar.fubar import get_mean_interparticle_separation as _get_mean_interparticle_separation
    from caesar.halo_matching import (
        ParticleMembership,
        _PidLookup,
        _build_selected_pid_maps,
        _read_ahf_hierarchy,
        _group_nodes_by_root,
        _compute_exclusive_memberships,
        _ensure_missing_ahf_halos,
        _update_ahf_galaxy_maps,
        _prune_halos_after_galaxies,
    )

    # Resolve minimum-star threshold from a single source (kwarg -> default)
    from caesar.group import get_min_stars as _get_min_stars

    min_stars = _get_min_stars(sim, override=min_stars)

    # Optional debug controls for the FAST path
    debug_fast = _os.environ.get("CAESAR_AHF_FAST_DEBUG", "0") == "1"
    debug_host: Optional[int]
    try:
        _h = _os.environ.get("CAESAR_AHF_FAST_DEBUG_HOST")
        debug_host = int(_h) if _h not in (None, "") else None
    except Exception:
        debug_host = None

    # Optional phase-wise memory tracing (survives OOM via JSONL + fsync)
    phase_memlog_enabled = _os.environ.get("CAESAR_AHF_FAST_PHASE_MEMLOG", "0") == "1"
    try:
        phase_memlog_every = int(_os.environ.get("CAESAR_AHF_FAST_PHASE_MEMLOG_EVERY", "100"))
    except Exception:
        phase_memlog_every = 100
    phase_memlog_every = max(1, phase_memlog_every)
    phase_memlog_file = _os.environ.get("CAESAR_AHF_FAST_PHASE_MEMLOG_FILE")
    if phase_memlog_enabled and not phase_memlog_file:
        phase_memlog_file = _os.path.abspath("ahf_fast_phase_memory.jsonl")
    phase_memlog_fsync = _os.environ.get("CAESAR_AHF_FAST_PHASE_MEMLOG_FSYNC", "1") == "1"
    model_debug_file = _os.environ.get("CAESAR_AHF_FAST_MODEL_DEBUG_FILE")
    if debug_fast and not model_debug_file:
        if phase_memlog_file:
            if phase_memlog_file.endswith(".jsonl"):
                model_debug_file = phase_memlog_file[:-6] + ".model.jsonl"
            else:
                model_debug_file = phase_memlog_file + ".model.jsonl"
        else:
            model_debug_file = _os.path.abspath("ahf_fast_model_debug.jsonl")
    _phase_lock = _threading.Lock()

    _psutil = None
    _proc = None
    if phase_memlog_enabled:
        try:
            import psutil as _ps
            _psutil = _ps
            _proc = _psutil.Process()
        except Exception:
            _psutil = None
            _proc = None

    def _snapshot_memory():
        rss_bytes = -1
        vms_bytes = -1
        avail_bytes = -1
        total_bytes = -1
        if _psutil is not None and _proc is not None:
            try:
                mi = _proc.memory_info()
                rss_bytes = int(getattr(mi, "rss", -1))
                vms_bytes = int(getattr(mi, "vms", -1))
            except Exception:
                pass
            try:
                vm = _psutil.virtual_memory()
                avail_bytes = int(getattr(vm, "available", -1))
                total_bytes = int(getattr(vm, "total", -1))
            except Exception:
                pass
        return rss_bytes, vms_bytes, avail_bytes, total_bytes

    def _phase_memlog(phase: str, **fields) -> None:
        if not phase_memlog_enabled:
            return

        rss_bytes, vms_bytes, avail_bytes, total_bytes = _snapshot_memory()
        record = {
            "ts": _time.time(),
            "phase": str(phase),
            "rss_gb": (rss_bytes / 2**30) if rss_bytes >= 0 else None,
            "vms_gb": (vms_bytes / 2**30) if vms_bytes >= 0 else None,
            "avail_gb": (avail_bytes / 2**30) if avail_bytes >= 0 else None,
            "total_gb": (total_bytes / 2**30) if total_bytes >= 0 else None,
            "pid": _os.getpid(),
        }
        if fields:
            record.update(fields)

        parts = [f"phase={record['phase']}"]
        if record["rss_gb"] is not None:
            parts.append(f"rss={record['rss_gb']:.2f}GB")
        if record["avail_gb"] is not None:
            parts.append(f"avail={record['avail_gb']:.2f}GB")
        for key in ("order_idx", "host_id", "heavy", "galaxies", "hosts_done", "hosts_total"):
            if key in record:
                parts.append(f"{key}={record[key]}")
        mylog.info("AHF-FAST mem: " + " ".join(parts))

        if phase_memlog_file:
            try:
                line = _json.dumps(record, sort_keys=True)
                with _phase_lock:
                    with open(phase_memlog_file, "a") as _fh:
                        _fh.write(line + "\n")
                        _fh.flush()
                        if phase_memlog_fsync:
                            _os.fsync(_fh.fileno())
            except Exception:
                pass

    def _model_debuglog(event: str, **fields) -> None:
        if not model_debug_file:
            return
        record = {
            "ts": _time.time(),
            "event": str(event),
            "pid": _os.getpid(),
        }
        if fields:
            record.update(fields)
        try:
            line = _json.dumps(record, sort_keys=True)
            with _phase_lock:
                with open(model_debug_file, "a") as _fh:
                    _fh.write(line + "\n")
                    _fh.flush()
                    if phase_memlog_fsync:
                        _os.fsync(_fh.fileno())
        except Exception:
            pass

    pid_maps_sel = _build_selected_pid_maps(sim)
    dm_pid_lookup = pid_maps_sel.get("dm")
    dm_full_lookup: Optional[_PidLookup] = None
    ndm_full = 0
    if dm_pid_lookup is not None and has_ptype(sim, "dm"):
        dm_pids_full = get_property(sim, "pid", "dm").d
        dm_pids_full = np.asarray(dm_pids_full, dtype=np.int64)
        ndm_full = dm_pids_full.size
        if ndm_full > 0:
            dm_full_lookup = _PidLookup(dm_pids_full)
        del dm_pids_full

    # Prefer pre-loaded hierarchy/memberships (set by AHF_FAST.run) so that we
    # do not re-read the AHF_particles file. Fall back to on-demand loading if
    # they are not available (e.g. in tests).
    parent_of = getattr(sim, "_ahf_fast_parent_of", None)
    children_of = getattr(sim, "_ahf_fast_children_of", None)
    membership_arrays = getattr(sim, "_ahf_fast_memberships", None)

    if not parent_of:
        parent_of, children_of = _read_ahf_hierarchy(ahf_particles_file)
        membership_arrays = None

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

    # Ensure we have memberships for all needed nodes.
    load_dm = dm_pid_lookup is not None
    if membership_arrays is None:
        membership_arrays = load_ahf_particle_blocks(
            ahf_particles_file,
            needed_nodes=all_needed_nodes,
            load_dm=load_dm,
        )

    jobs = n_jobs
    if jobs is None:
        jobs = getattr(sim, "nproc", 1)
    try:
        jobs = int(jobs)
    except Exception:
        jobs = 1
    jobs = max(1, jobs)

    show_progress = bool(getattr(sim, "_show_progress", True))
    total_hosts = len(host_to_nodes)
    host_progress = tqdm(
        total=total_hosts,
        desc="Building galaxies (AHF-FAST)",
        disable=(not show_progress or total_hosts == 0),
        leave=False,
        mininterval=2.0,
        miniters=32,
        smoothing=0.0,
    )

    halos_df = getattr(sim, "_ahf_fast_halos_df", None)
    if halos_df is None:
        halos_df = load_ahf_halos_dataframe(ahf_particles_file)
        sim._ahf_fast_halos_df = halos_df
    if len(halos_df) == 0:
        raise AssertionError("AHF-FAST invariant violated: AHF_halos dataframe is empty")
    if "n_star" not in halos_df.columns:
        raise AssertionError("AHF-FAST invariant violated: AHF_halos dataframe is missing n_star")

    node_npart = {
        int(hid): int(npart)
        for hid, npart in zip(halos_df["hid"].to_numpy(), halos_df["npart"].to_numpy())
    }
    node_nstar = {
        int(hid): max(0, int(nstar))
        for hid, nstar in zip(halos_df["hid"].to_numpy(), halos_df["n_star"].to_numpy())
    }

    host_schedule: List[HostSchedState] = []
    host_workloads: List[int] = []
    host_items = sorted(host_to_nodes.items(), key=lambda kv: int(kv[0]))
    for order_idx, (root_id, nodes_for_host) in enumerate(host_items):
        host_star_count = int(node_nstar.get(int(root_id), -1))
        if host_star_count < 0:
            host_star_count = int(
                max((int(node_nstar.get(int(node_id), 0)) for node_id in nodes_for_host), default=0)
            )
        host_star_count = max(0, int(host_star_count))
        fof_candidates = int(
            sum(int(node_npart.get(int(node_id), 0)) for node_id in nodes_for_host)
        )
        fof_candidates = max(1, fof_candidates)
        host_schedule.append(
            HostSchedState(
                order_idx=int(order_idx),
                root_id=int(root_id),
                nodes=set(nodes_for_host),
                fof_candidates=int(fof_candidates),
                star_count=int(host_star_count),
            )
        )
        host_workloads.append(int(fof_candidates))

    def _env_int(name: str, default: int) -> int:
        try:
            return int(_os.environ.get(name, str(default)))
        except Exception:
            return int(default)

    def _env_float(name: str, default: float) -> float:
        try:
            return float(_os.environ.get(name, str(default)))
        except Exception:
            return float(default)

    def _env_optional_float(name: str) -> Optional[float]:
        raw = _os.environ.get(name)
        if raw is None:
            return None
        txt = str(raw).strip().lower()
        if txt in ("", "auto"):
            return None
        try:
            return float(txt)
        except Exception:
            return None

    scheduler_enabled = jobs > 1 and bool(host_schedule)
    heavy_threshold = 0
    heavy_count = 0

    # Host-atomic scheduler controls.
    work_unit = 1
    tiny_proxy_star_threshold = 32
    adjust_every = max(1, _env_int("CAESAR_AHF_FAST_SCHED_ADJUST_EVERY", max(2, jobs // 2)))
    adjust_interval_s = 0.75
    mem_slope_warn_gbps = max(0.1, _env_float("CAESAR_AHF_FAST_MEM_SLOPE_WARN_GBPS", 1.0))
    mem_slope_crit_gbps = max(mem_slope_warn_gbps, _env_float("CAESAR_AHF_FAST_MEM_SLOPE_CRIT_GBPS", 2.5))
    mem_safety_frac = float(_ahf_fast_clamp(_env_float("CAESAR_AHF_FAST_MEM_SAFETY_FRAC", 0.18), 0.01, 0.90))
    mem_safety_gb_min = max(1.0, _env_float("CAESAR_AHF_FAST_MEM_SAFETY_GB_MIN", 64.0))
    resv_base_unit_gb = _env_optional_float("CAESAR_AHF_FAST_RESV_BASE_UNIT_GB")
    resv_min_bytes = max(1, int(round(max(0.01, _env_float("CAESAR_AHF_FAST_RESV_MIN_GB", 0.25)) * _GIB)))
    resv_max_bytes = max(
        resv_min_bytes,
        int(
            round(
                max(
                    _env_float("CAESAR_AHF_FAST_RESV_MIN_GB", 0.25),
                    _env_float("CAESAR_AHF_FAST_RESV_MAX_GB", 32.0),
                )
                * _GIB
            )
        ),
    )
    model_tau = float(_ahf_fast_clamp(_env_float("CAESAR_AHF_FAST_MODEL_TAU", 0.90), 0.50, 0.99))
    model_knots = max(8, _env_int("CAESAR_AHF_FAST_MODEL_KNOTS", 32))
    model_neighbors = max(16, _env_int("CAESAR_AHF_FAST_MODEL_NEIGHBORS", 256))
    model_refit_exp_start = max(0, _env_int("CAESAR_AHF_FAST_MODEL_REFIT_EXP_START", 3))
    model_refit_exp_max = max(model_refit_exp_start, _env_int("CAESAR_AHF_FAST_MODEL_REFIT_EXP_MAX", 16))
    model_drift_lo = float(_ahf_fast_clamp(_env_float("CAESAR_AHF_FAST_MODEL_DRIFT_LO", 0.70), 0.10, 1.0))
    model_drift_hi = float(max(model_drift_lo + 1.0e-6, _env_float("CAESAR_AHF_FAST_MODEL_DRIFT_HI", 1.35)))
    model_drift_cooldown_s = max(0.0, _env_float("CAESAR_AHF_FAST_MODEL_DRIFT_COOLDOWN_S", 30.0))
    anchor_max_inflight = max(1, _env_int("CAESAR_AHF_FAST_RESV_ANCHOR_MAX_INFLIGHT", 2))
    model_sample_max_inflight = max(
        int(anchor_max_inflight),
        _env_int("CAESAR_AHF_FAST_MODEL_SAMPLE_MAX_INFLIGHT", 8),
    )
    model_bootstrap_large_hosts = max(
        0,
        _env_int("CAESAR_AHF_FAST_MODEL_BOOTSTRAP_LARGE_HOSTS", 4),
    )
    model_bootstrap_max_inflight = max(
        1,
        min(
            int(model_sample_max_inflight),
            _env_int("CAESAR_AHF_FAST_MODEL_BOOTSTRAP_MAX_INFLIGHT", int(anchor_max_inflight)),
        ),
    )
    dispatch_heartbeat_ms = max(1, _env_int("CAESAR_AHF_FAST_DISPATCH_HEARTBEAT_MS", 50))
    model_seed_anchor_count = max(2, min(12, int(model_knots)))
    model_seed_anchor_weight = 0.25
    model_seed_drop_samples = max(
        16,
        _env_int(
            "CAESAR_AHF_FAST_MODEL_SEED_DROP_SAMPLES",
            max(64, int(4 * model_seed_anchor_count)),
        ),
    )

    # Obtain total memory once to set scale-aware defaults when thresholds are
    # not explicitly configured.
    if _psutil is None or _proc is None:
        try:
            import psutil as _ps
            _psutil = _ps
            _proc = _psutil.Process()
        except Exception:
            pass
    _, _, _, _total_b0 = _snapshot_memory()
    total_bytes = int(_total_b0) if _total_b0 and _total_b0 > 0 else 0
    safety_floor_base_bytes = int(max(float(total_bytes) * float(mem_safety_frac), float(mem_safety_gb_min) * _GIB))

    positive_workloads = [int(w) for w in host_workloads if int(w) > 0]
    if positive_workloads:
        work_unit = max(1, int(np.percentile(np.asarray(positive_workloads, dtype=np.float64), 50.0)))

    trace_heavy_percentile = min(100.0, max(0.0, _env_float("CAESAR_AHF_FAST_TRACE_HEAVY_PERCENTILE", 97.0)))
    trace_heavy_min = max(0, _env_int("CAESAR_AHF_FAST_TRACE_HEAVY_MIN_CANDIDATES", 50000))
    if host_workloads:
        heavy_threshold = max(
            trace_heavy_min,
            int(np.percentile(np.asarray(host_workloads, dtype=np.float64), trace_heavy_percentile)),
        )
    else:
        heavy_threshold = trace_heavy_min

    units_for_auto = []
    for item in host_schedule:
        w = int(item.fof_candidates)
        item.is_heavy = bool(w >= heavy_threshold)
        item.is_tiny_proxy = bool(int(item.star_count) <= int(tiny_proxy_star_threshold))
        # Sub-linear work units preserve scale differences without making
        # high-work hosts effectively single-threaded.
        scaled = float(max(1, w)) / float(max(1, work_unit))
        item.work_units = int(min(max(1, np.ceil(np.sqrt(max(1.0, scaled)))), max(2, jobs * 2)))
        units_for_auto.append(max(1, int(item.work_units)))

    median_units = float(np.percentile(np.asarray(units_for_auto, dtype=np.float64), 50.0)) if units_for_auto else 1.0
    phasea_units_for_auto = [
        max(1, int(item.work_units)) for item in host_schedule if not bool(item.is_tiny_proxy)
    ]
    phasea_units_scale = (
        float(np.percentile(np.asarray(phasea_units_for_auto, dtype=np.float64), 75.0))
        if phasea_units_for_auto
        else float(max(1.0, median_units))
    )
    if resv_base_unit_gb is not None:
        base_unit_bytes = int(max(1, round(float(resv_base_unit_gb) * _GIB)))
    else:
        auto_total = float(total_bytes) if total_bytes > 0 else float(max(1, jobs) * 4.0 * _GIB)
        worker_scale = float(max(32, min(int(jobs), 128)))
        auto_bytes = (0.45 * auto_total) / max(1.0, worker_scale * max(1.0, phasea_units_scale))
        base_unit_bytes = int(round(auto_bytes))
    base_unit_bytes = int(_ahf_fast_clamp(base_unit_bytes, resv_min_bytes, resv_max_bytes))

    phasea_fof_candidates = [
        int(item.fof_candidates) for item in host_schedule if not bool(item.is_tiny_proxy)
    ]
    model_probe_candidates = _ahf_fast_probe_fof_candidates(
        fof_candidates=phasea_fof_candidates,
        probe_count=12,
    )
    seed_anchor_x, seed_anchor_y, seed_anchor_w = _ahf_fast_seed_anchor_samples(
        fof_candidates=phasea_fof_candidates,
        work_unit=int(work_unit),
        base_unit_bytes=int(base_unit_bytes),
        pred_min_bytes=int(resv_min_bytes),
        pred_max_bytes=int(resv_max_bytes),
        anchor_count=int(model_seed_anchor_count),
        anchor_weight=float(model_seed_anchor_weight),
    )
    initial_mem_model = _ahf_fast_fit_direct_monotone_model(
        sample_x=seed_anchor_x,
        sample_y=seed_anchor_y,
        sample_w=seed_anchor_w,
    )

    for item in host_schedule:
        seed_bytes = _ahf_fast_seed_prediction_bytes(
            fof_candidates=int(item.fof_candidates),
            work_unit=int(work_unit),
            base_unit_bytes=int(base_unit_bytes),
            pred_min_bytes=int(resv_min_bytes),
            pred_max_bytes=int(resv_max_bytes),
        )
        item.predicted_bytes = _ahf_fast_predict_reservation_bytes(
            fof_candidates=int(item.fof_candidates),
            model=initial_mem_model,
            seed_bytes=int(seed_bytes),
            pred_min_bytes=int(resv_min_bytes),
            pred_max_bytes=int(resv_max_bytes),
        )

    heavy_count = int(sum(1 for item in host_schedule if item.is_heavy))
    heavy_examples = sorted(
        (
            (
                int(item.fof_candidates),
                int(item.work_units),
                int(item.star_count),
                int(item.root_id),
            )
            for item in host_schedule
            if item.is_heavy
        ),
        reverse=True,
    )[:5]

    if scheduler_enabled:
        mylog.info(
            "AHF-FAST: scheduler enabled: hosts=%d workers=%d work_unit=%d mode=global_phaseA+tiny_saturation adjust_every=%d",
            len(host_schedule),
            jobs,
            work_unit,
            adjust_every,
        )
        mylog.info(
            "AHF-FAST: tracing threshold fof_candidates >= %d (count=%d)",
            heavy_threshold,
            heavy_count,
        )
        mylog.info(
            "AHF-FAST: largest host examples (fof_candidates, work_units, stars, host_id): %s",
            heavy_examples,
        )
        mylog.info(
            "AHF-FAST: reservation model safety=(frac=%.2f,min=%.1fGB,base=%.1fGB) "
            "slope_gbps=(warn=%.2f,crit=%.2f) base_unit=%.2fGB phasea_units_scale=%.2f clamp=(min=%.2fGB,max=%.2fGB)",
            mem_safety_frac,
            mem_safety_gb_min,
            float(safety_floor_base_bytes) / _GIB,
            mem_slope_warn_gbps,
            mem_slope_crit_gbps,
            float(base_unit_bytes) / _GIB,
            float(phasea_units_scale),
            float(resv_min_bytes) / _GIB,
            float(resv_max_bytes) / _GIB,
        )
        mylog.info(
            "AHF-FAST: model config tau=%.2f knots=%d neighbors=%d refit_exp=[%d,%d] drift=[%.3f,%.3f] cooldown=%.1fs sample_max_inflight=%d bootstrap_large_hosts=%d bootstrap_max_inflight=%d seed_anchors=%d seed_weight=%.2f seed_drop_samples=%d",
            model_tau,
            int(model_knots),
            int(model_neighbors),
            int(model_refit_exp_start),
            int(model_refit_exp_max),
            model_drift_lo,
            model_drift_hi,
            model_drift_cooldown_s,
            int(model_sample_max_inflight),
            int(model_bootstrap_large_hosts),
            int(model_bootstrap_max_inflight),
            int(len(seed_anchor_x)),
            float(model_seed_anchor_weight),
            int(model_seed_drop_samples),
        )
        _model_debuglog(
            "model_initial",
            sample_count=0,
            using_seed_anchors=True,
            seed_anchor_count=int(len(seed_anchor_x)),
            seed_drop_samples=int(model_seed_drop_samples),
            probe_fof_candidates=model_probe_candidates,
            probe_pred_gb=[
                round(
                    float(
                        _ahf_fast_predict_reservation_bytes(
                            fof_candidates=int(v),
                            model=initial_mem_model,
                            seed_bytes=_ahf_fast_seed_prediction_bytes(
                                fof_candidates=int(v),
                                work_unit=int(work_unit),
                                base_unit_bytes=int(base_unit_bytes),
                                pred_min_bytes=int(resv_min_bytes),
                                pred_max_bytes=int(resv_max_bytes),
                            ),
                            pred_min_bytes=int(resv_min_bytes),
                            pred_max_bytes=int(resv_max_bytes),
                        )
                    )
                    / _GIB,
                    4,
                )
                for v in model_probe_candidates
            ],
        )
        mylog.info("AHF-FAST: tiny saturation stars<=%d", int(tiny_proxy_star_threshold))
    else:
        mylog.info(
            "AHF-FAST: serial scheduler (workers=%d, hosts=%d)",
            jobs,
            len(host_schedule),
        )

    def _trace_host(order_idx: int, is_heavy: bool) -> bool:
        if not phase_memlog_enabled:
            return False
        if is_heavy:
            return True
        return (order_idx % phase_memlog_every) == 0

    _phase_memlog(
        "host_prepare",
        hosts_total=len(host_schedule),
        jobs=jobs,
        scheduler_mode="hybrid_host_atomic" if scheduler_enabled else "serial",
        work_unit=int(work_unit),
        heavy_hosts=heavy_count,
        heavy_threshold=heavy_threshold,
        reserve_base_gb=round(float(base_unit_bytes) / _GIB, 4),
        model_tau=model_tau,
        model_knots=int(model_knots),
        model_neighbors=int(model_neighbors),
    )

    def map_sel(pidset: Set[int], key: str) -> np.ndarray:
        lookup = pid_maps_sel.get(key)
        if lookup is None or not pidset:
            return np.empty(0, dtype=np.int32)
        mapped = lookup.map(pidset)
        if mapped.size == 0:
            return mapped
        return np.unique(mapped)

    # 6D-FOF configuration for subhalo payloads in AHF-FAST mode.
    subhalo_fof6d_enabled = _os.environ.get("CAESAR_AHF_FAST_SUBHALO_FOF6D", "1") == "1"
    fof_nHlim = _env_float("CAESAR_AHF_FAST_FOF_NHLIM", 0.13)
    fof_Tlim = _env_float("CAESAR_AHF_FAST_FOF_TLIM", 1.0e5)
    fof_use_sfr_gate = _os.environ.get("CAESAR_AHF_FAST_FOF_USE_SFR", "1") == "1"
    fof_vel_ll = 1.0
    try:
        _vel_env = _os.environ.get("CAESAR_FOF6D_VEL_LL")
        if _vel_env is not None and _vel_env != "":
            fof_vel_ll = float(_vel_env)
    except Exception:
        pass
    if _os.environ.get("CAESAR_FOF6D_DISABLE_VEL", "0") == "1":
        fof_vel_ll = None

    fof_ll = 0.0
    fof_kerneltab = None
    fof_boxsize = float(sim.simulation.boxsize.d)
    if subhalo_fof6d_enabled:
        try:
            fof_mis = float(_get_mean_interparticle_separation(sim).d)
            fof_ll = float(fof_mis * _get_b(sim, "galaxy"))
            fof_kerneltab = _fof6d_kernel_table(fof_ll)
            mylog.info(
                "AHF-FAST: subhalo 6D-FOF enabled (LL=%g, vel_LL=%s, min_stars=%d)",
                fof_ll,
                "None" if fof_vel_ll is None else f"{fof_vel_ll:g}",
                int(min_stars),
            )
        except Exception as exc:
            subhalo_fof6d_enabled = False
            mylog.warning("AHF-FAST: disabling subhalo 6D-FOF due to setup error: %s", exc)
    else:
        mylog.info("AHF-FAST: subhalo 6D-FOF disabled via CAESAR_AHF_FAST_SUBHALO_FOF6D=0")

    tiny_star_threshold = int(tiny_proxy_star_threshold)
    huge_q = min(0.99, max(0.55, _env_float("CAESAR_AHF_FAST_BIN_HUGE_Q", 0.85)))
    payload_update_every = max(1, _env_int("CAESAR_AHF_FAST_BIN_UPDATE_EVERY", 32))
    payload_warmup = max(8, _env_int("CAESAR_AHF_FAST_BIN_WARMUP", max(16, jobs * 4)))
    payload_sample_cap = max(128, _env_int("CAESAR_AHF_FAST_BIN_SAMPLE_CAP", 4096))
    payload_samples = _deque(maxlen=payload_sample_cap)
    payload_samples.append(int(max(1, min_stars)))

    payload_cutoffs = {"tiny": int(tiny_star_threshold), "huge": int(max(2, min_stars + 1))}
    payload_lock = _threading.Lock()

    def _refresh_payload_cutoffs_locked() -> None:
        if not payload_samples:
            payload_cutoffs["tiny"] = int(tiny_star_threshold)
            payload_cutoffs["huge"] = int(max(2, min_stars + 1))
            return
        arr = np.asarray(payload_samples, dtype=np.float64)
        huge_est = int(np.percentile(arr, 100.0 * huge_q))
        huge_floor = max(int(min_stars) + 1, int(np.percentile(arr, 60.0)))
        huge_est = max(huge_floor, huge_est)
        payload_cutoffs["tiny"] = int(tiny_star_threshold)
        payload_cutoffs["huge"] = int(huge_est)

    with payload_lock:
        _refresh_payload_cutoffs_locked()
        init_tiny = int(payload_cutoffs["tiny"])
        init_huge = int(payload_cutoffs["huge"])
    mylog.info(
        "AHF-FAST: dynamic payload bins enabled (tiny_stars<=%d fixed, huge_q=%.2f init_cutoffs=(tiny=%d,huge=%d))",
        int(tiny_star_threshold),
        huge_q,
        init_tiny,
        init_huge,
    )

    def _snapshot_payload_cutoffs() -> Tuple[int, int]:
        with payload_lock:
            return int(payload_cutoffs["tiny"]), int(payload_cutoffs["huge"])

    def _update_payload_samples(sample_counts: Sequence[int]) -> None:
        if not sample_counts:
            return
        with payload_lock:
            before = len(payload_samples)
            for count in sample_counts:
                payload_samples.append(max(0, int(count)))
            after = len(payload_samples)
            if after <= payload_warmup:
                _refresh_payload_cutoffs_locked()
                return
            if payload_update_every <= 1:
                _refresh_payload_cutoffs_locked()
                return
            prev_bucket = int(before // payload_update_every)
            new_bucket = int(after // payload_update_every)
            if new_bucket > prev_bucket:
                _refresh_payload_cutoffs_locked()

    def _classify_payload_size(
        star_particle_count: int,
        fof_particle_count: int,
        tiny_cut: int,
        huge_cut: int,
    ) -> str:
        stars = max(0, int(star_particle_count))
        count = max(0, int(fof_particle_count))
        if stars <= int(tiny_cut):
            return "tiny"
        if count >= int(huge_cut):
            return "huge"
        return "normal"

    def _dense_gas_selected(gidx: np.ndarray) -> np.ndarray:
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

    def _make_group(
        node_id: int,
        star_sel: np.ndarray,
        gas_sel_dense: np.ndarray,
        bh_sel: np.ndarray,
        dm_selected: np.ndarray,
        dm_exc: Optional[Set[int]] = None,
    ):
        grp = create_new_group(sim, "galaxy")
        grp.AHF_haloID = int(node_id)
        grp.slist = np.asarray(star_sel, dtype=np.int32)
        grp.glist = np.asarray(gas_sel_dense, dtype=np.int32)
        if "bh" in pid_maps_sel:
            grp.bhlist = np.asarray(bh_sel, dtype=np.int32)
        grp.dmlist = np.asarray(dm_selected, dtype=np.int32)
        grp.global_indexes = np.array([], dtype=np.int64)
        grp.__dict__["_dm_exclusive_pids"] = set(dm_exc) if dm_exc else set()
        return grp

    skipped_empty_payloads = 0

    def process_host(
        order_idx: int,
        root_id: int,
        bucket: Dict[int, ParticleMembership],
        fof_candidates: int = 0,
        is_heavy: bool = False,
    ):
        nodes_for_host = host_to_nodes.get(root_id, set())
        if not nodes_for_host:
            return order_idx, [], 0, {}

        if _trace_host(order_idx, is_heavy):
            _phase_memlog(
                "depth_wave_start",
                order_idx=order_idx,
                host_id=int(root_id),
                heavy=int(is_heavy),
                fof_candidates=int(fof_candidates),
                nodes=int(len(nodes_for_host)),
            )

        # Ensure every node has a membership object (possibly empty)
        for node in nodes_for_host:
            bucket.setdefault(node, ParticleMembership(node))

        payloads: List[Tuple[int, Set[int], Set[int], Set[int], Set[int]]] = []

        # Fast path: a host tree with one node has no exclusive-subtraction
        # work, so we can skip depth traversal/set carry propagation.
        if len(nodes_for_host) == 1:
            node = next(iter(nodes_for_host))
            pm = bucket.get(node, ParticleMembership(node))
            if len(pm.parttype4) >= min_stars:
                payloads.append((node, pm.parttype4, pm.parttype0, pm.parttype5, pm.parttype1))
        else:
            exclusives = _compute_exclusive_memberships(bucket, children_of, nodes_for_host)

            from collections import defaultdict as _dd

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
            if _trace_host(order_idx, is_heavy):
                _phase_memlog(
                    "depth_wave_done",
                    order_idx=order_idx,
                    host_id=int(root_id),
                    heavy=int(is_heavy),
                    fof_candidates=int(fof_candidates),
                    payloads=0,
                    skipped=0,
                )
            return order_idx, [], 0, {}

        host_galaxies: List[Tuple[int, object]] = []
        local_skipped = 0
        host_stats = {
            "payloads": 0,
            "tiny": 0,
            "normal": 0,
            "huge": 0,
            "tiny_skips": 0,
            "fof_runs": 0,
            "fof_groups": 0,
            "fof_unassigned": 0,
            "fof_empty": 0,
            "fof_fallback": 0,
        }

        tiny_cut, huge_cut = _snapshot_payload_cutoffs()
        new_payload_samples: List[int] = []

        for payload in payloads:
            node_id, star_set, gas_set, bh_set, dm_exc = payload
            dm_pm = bucket.get(node_id)
            dm_inclusive = dm_pm.parttype1 if dm_pm is not None else set()

            star_sel = map_sel(star_set, "star")
            gas_sel = map_sel(gas_set, "gas")
            gas_sel_dense = _dense_gas_selected(gas_sel)
            if "bh" in pid_maps_sel:
                bh_sel = map_sel(bh_set, "bh")
            else:
                bh_sel = np.empty(0, dtype=np.int32)
            if "dm" in pid_maps_sel:
                dm_selected = map_sel(dm_inclusive, "dm")
            else:
                dm_selected = np.empty(0, dtype=np.int32)

            star_particle_count = int(len(star_sel))
            fof_particle_count = int(star_particle_count + len(gas_sel_dense) + len(bh_sel))
            if star_particle_count > int(tiny_star_threshold):
                new_payload_samples.append(int(fof_particle_count))
            mode = _classify_payload_size(
                star_particle_count,
                fof_particle_count,
                int(tiny_cut),
                int(huge_cut),
            )
            host_stats["payloads"] += 1
            host_stats[mode] += 1

            did_run_fof = False
            built_from_fof = False

            if (
                subhalo_fof6d_enabled
                and fof_kerneltab is not None
                and fof_ll > 0.0
                and mode != "tiny"
                and fof_particle_count > int(min_stars)
            ):
                gas_concat = sim.data_manager.selected_to_concat("gas", np.asarray(gas_sel_dense, dtype=np.int64))
                star_concat = sim.data_manager.selected_to_concat("star", np.asarray(star_sel, dtype=np.int64))
                if bh_sel.size > 0:
                    bh_concat = sim.data_manager.selected_to_concat("bh", np.asarray(bh_sel, dtype=np.int64))
                else:
                    bh_concat = np.empty(0, dtype=np.int64)
                eligible_concat = np.concatenate((gas_concat, star_concat, bh_concat), axis=None).astype(np.int64, copy=False)

                if eligible_concat.size > int(min_stars):
                    did_run_fof = True
                    host_stats["fof_runs"] += 1
                    tags, nfof = _fof6d_halo(
                        int(eligible_concat.size),
                        int(eligible_concat.size),
                        sim.data_manager.pos[eligible_concat],
                        sim.data_manager.vel[eligible_concat],
                        int(min_stars),
                        fof_boxsize,
                        fof_ll,
                        fof_vel_ll,
                        fof_kerneltab,
                    )
                    tags = np.asarray(tags, dtype=np.int64)
                    valid_tags = np.unique(tags[tags >= 0])
                    host_stats["fof_unassigned"] += int(np.sum(tags < 0))

                    if valid_tags.size > 0:
                        ng = int(gas_sel_dense.size)
                        ns = int(star_sel.size)
                        nb = int(bh_sel.size)
                        allow_dm = (valid_tags.size == 1)
                        if (not allow_dm) and dm_selected.size > 0:
                            dm_selected = np.empty(0, dtype=np.int32)
                        built_local = 0
                        for gid in valid_tags:
                            mask = tags == int(gid)
                            gsub = gas_sel_dense[mask[:ng]] if ng > 0 else np.empty(0, dtype=np.int32)
                            ssub = star_sel[mask[ng:ng + ns]] if ns > 0 else np.empty(0, dtype=np.int32)
                            bsub = bh_sel[mask[ng + ns:ng + ns + nb]] if nb > 0 else np.empty(0, dtype=np.int32)
                            # Keep the same galaxy validity gate as CAESAR core.
                            if int(ssub.size) < int(min_stars):
                                continue
                            grp = _make_group(
                                int(node_id),
                                ssub,
                                gsub,
                                bsub,
                                dm_selected if allow_dm else np.empty(0, dtype=np.int32),
                                dm_exc if allow_dm else None,
                            )
                            host_galaxies.append((int(node_id), grp))
                            built_local += 1

                        if built_local > 0:
                            built_from_fof = True
                            host_stats["fof_groups"] += int(built_local)
                    else:
                        host_stats["fof_empty"] += 1

            if built_from_fof:
                continue

            if did_run_fof:
                host_stats["fof_fallback"] += 1
            if mode == "tiny":
                host_stats["tiny_skips"] += 1

            grp = _make_group(int(node_id), star_sel, gas_sel_dense, bh_sel, dm_selected, dm_exc)
            mapped_star = len(grp.slist) if hasattr(grp, "slist") else 0
            mapped_gas = len(grp.glist) if hasattr(grp, "glist") else 0
            mapped_bh = len(grp.bhlist) if hasattr(grp, "bhlist") else 0
            mapped_dm = len(dm_selected)
            particle_total = mapped_star + mapped_gas + mapped_bh + mapped_dm
            if particle_total == 0:
                local_skipped += 1
                if local_skipped <= 3:
                    mylog.warning(
                        "AHF-FAST: node %d had %d star / %d gas / %d bh / %d dm particles "
                        "from AHF but none mapped into CAESAR selection",
                        node_id,
                        len(star_set),
                        len(gas_set),
                        len(bh_set),
                        len(dm_inclusive),
                    )
                continue
            host_galaxies.append((int(node_id), grp))

        _update_payload_samples(new_payload_samples)

        if _trace_host(order_idx, is_heavy):
            _phase_memlog(
                "depth_wave_done",
                order_idx=order_idx,
                host_id=int(root_id),
                heavy=int(is_heavy),
                fof_candidates=int(fof_candidates),
                payloads=int(host_stats["payloads"]),
                galaxies=int(len(host_galaxies)),
                skipped=int(local_skipped),
                fof_runs=int(host_stats["fof_runs"]),
                fof_groups=int(host_stats["fof_groups"]),
                tiny_cut=int(tiny_cut),
                huge_cut=int(huge_cut),
            )

        return order_idx, host_galaxies, local_skipped, host_stats

    pending_results: Dict[int, List] = {}
    host_meta = {int(item.order_idx): item for item in host_schedule}
    next_to_emit = 0
    completed_hosts = 0
    emitted_hosts = 0
    inflight_count = 0
    reserved_bytes_total = 0
    fof_totals = {
        "payloads": 0,
        "tiny": 0,
        "normal": 0,
        "huge": 0,
        "tiny_skips": 0,
        "fof_runs": 0,
        "fof_groups": 0,
        "fof_unassigned": 0,
        "fof_empty": 0,
        "fof_fallback": 0,
    }

    def build_bucket(nodes_for_host: Set[int]) -> Dict[int, ParticleMembership]:
        """Build a membership bucket for one host from loader arrays."""
        bucket: Dict[int, ParticleMembership] = {}
        if not membership_arrays:
            return bucket

        for node_id in nodes_for_host:
            arr = membership_arrays.get(int(node_id))
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.int64)
            if arr.size == 0:
                continue
            if arr.ndim != 2 or arr.shape[1] != 2:
                arr = arr.reshape(-1, 2)
            pids = arr[:, 0]
            ptypes = arr[:, 1]
            pm = ParticleMembership(int(node_id))
            mask0 = ptypes == 0
            if np.any(mask0):
                pm.parttype0 = set(int(v) for v in pids[mask0])
            if load_dm:
                mask1 = ptypes == 1
                if np.any(mask1):
                    pm.parttype1 = set(int(v) for v in pids[mask1])
                mask2 = ptypes == 2
                if np.any(mask2):
                    pm.parttype2 = set(int(v) for v in pids[mask2])
                mask3 = ptypes == 3
                if np.any(mask3):
                    pm.parttype3 = set(int(v) for v in pids[mask3])
            mask4 = ptypes == 4
            if np.any(mask4):
                pm.parttype4 = set(int(v) for v in pids[mask4])
            mask5 = ptypes == 5
            if np.any(mask5):
                pm.parttype5 = set(int(v) for v in pids[mask5])
            bucket[int(node_id)] = pm
        return bucket

    def process_host_task(
        order_idx: int,
        root_id: int,
        nodes_for_host: Set[int],
        fof_candidates: int = 0,
        is_heavy: bool = False,
    ):
        bucket = build_bucket(nodes_for_host)
        return process_host(order_idx, root_id, bucket, fof_candidates, is_heavy)

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        pending_futures: Dict = {}
        bootstrap_phasea = _deque(
            _ahf_fast_select_bootstrap_hosts(
                [item for item in host_schedule if not bool(item.is_tiny_proxy)],
                count=int(model_bootstrap_large_hosts),
            )
        )
        bootstrap_phasea_ids = {int(item.order_idx) for item in bootstrap_phasea}
        pending_phasea: List[HostSchedState] = [
            item
            for item in host_schedule
            if (not bool(item.is_tiny_proxy)) and (int(item.order_idx) not in bootstrap_phasea_ids)
        ]
        pending_tiny = _deque(
            sorted(
                (item for item in host_schedule if bool(item.is_tiny_proxy)),
                key=lambda h: (int(h.fof_candidates), int(h.root_id)),
            )
        )

        # Scheduler telemetry/state.
        mem_samples = _deque(maxlen=max(6, int(adjust_every * 2)))
        drift_samples = _deque(maxlen=512)
        reserved_samples: List[int] = []
        scheduler_mode = "nominal"
        scheduler_slope_gbps = 0.0
        safety_floor_bytes = int(safety_floor_base_bytes)
        capacity_bytes = 0
        completed_since_adjust = 0
        last_adjust_ts = 0.0
        model_refits = 0
        drift_refits = 0
        phasea_fit_success_count = 0
        phasea_fit_fail_count = 0
        phaseb_backfill_count = 0
        deadlock_guard_count = 0
        dispatch_wakeup_count = 0
        dispatch_submit_batches = 0
        sample_x: List[float] = []
        sample_y: List[float] = []
        sample_w: List[float] = []
        mem_model: Optional[_MemModel] = initial_mem_model
        max_refit_target = int(2 ** int(model_refit_exp_max))
        next_refit_target = int(2 ** int(model_refit_exp_start))
        last_model_refit_ts = 0.0
        last_model_refit_completed = 0
        latest_model_sample: Optional[Dict[str, float]] = None
        bootstrap_inflight_count = 0
        phasea_index_dirty = True
        phasea_predicted: List[int] = []

        def _predict_host_reservation_bytes(host_state: HostSchedState) -> int:
            seed_bytes = _ahf_fast_seed_prediction_bytes(
                fof_candidates=int(host_state.fof_candidates),
                work_unit=int(work_unit),
                base_unit_bytes=int(base_unit_bytes),
                pred_min_bytes=int(resv_min_bytes),
                pred_max_bytes=int(resv_max_bytes),
            )
            predicted = _ahf_fast_predict_reservation_bytes(
                fof_candidates=int(host_state.fof_candidates),
                model=mem_model,
                seed_bytes=int(seed_bytes),
                pred_min_bytes=int(resv_min_bytes),
                pred_max_bytes=int(resv_max_bytes),
            )
            return int(_ahf_fast_clamp(float(predicted), float(resv_min_bytes), float(resv_max_bytes)))

        def _rebuild_phasea_index() -> None:
            nonlocal phasea_index_dirty, phasea_predicted
            if not pending_phasea:
                phasea_predicted = []
                phasea_index_dirty = False
                return
            for item in pending_phasea:
                item.predicted_bytes = int(_predict_host_reservation_bytes(item))
            pending_phasea.sort(
                key=lambda h: (
                    int(h.predicted_bytes),
                    int(h.fof_candidates),
                    int(h.root_id),
                )
            )
            phasea_predicted = [int(item.predicted_bytes) for item in pending_phasea]
            for idx, item in enumerate(pending_phasea):
                item.queue_index = int(idx)
            phasea_index_dirty = False

        def _mark_phasea_index_dirty() -> None:
            nonlocal phasea_index_dirty
            phasea_index_dirty = True

        def _pop_phasea_at(idx: int) -> HostSchedState:
            item = pending_phasea.pop(int(idx))
            if 0 <= int(idx) < len(phasea_predicted):
                phasea_predicted.pop(int(idx))
            return item

        def _maybe_refit_model(now_ts: float, force: bool = False) -> None:
            nonlocal mem_model, model_refits, drift_refits
            nonlocal last_model_refit_ts, last_model_refit_completed, next_refit_target
            min_samples = max(8, int(2 ** int(model_refit_exp_start)))
            if len(sample_x) < int(min_samples):
                return

            should_refit = bool(force)
            if not should_refit and int(completed_hosts) >= int(next_refit_target):
                if int(next_refit_target) < int(max_refit_target) or int(last_model_refit_completed) < int(next_refit_target):
                    should_refit = True

            if (
                not should_refit
                and drift_samples
                and float(now_ts - last_model_refit_ts) >= float(model_drift_cooldown_s)
            ):
                drift_med = float(np.median(np.asarray(drift_samples, dtype=np.float64)))
                if drift_med < float(model_drift_lo) or drift_med > float(model_drift_hi):
                    should_refit = True
                    drift_refits += 1

            if not should_refit:
                return

            fit_x, fit_y, fit_w = _ahf_fast_model_fit_samples(
                seed_anchor_x=seed_anchor_x,
                seed_anchor_y=seed_anchor_y,
                seed_anchor_w=seed_anchor_w,
                sample_x=sample_x,
                sample_y=sample_y,
                sample_w=sample_w,
                seed_drop_samples=int(model_seed_drop_samples),
            )
            if len(sample_x) < int(model_seed_drop_samples):
                fitted = _ahf_fast_fit_direct_monotone_model(
                    sample_x=fit_x,
                    sample_y=fit_y,
                    sample_w=fit_w,
                )
            else:
                fitted = _ahf_fast_fit_mem_model(
                    sample_x=fit_x,
                    sample_y=fit_y,
                    sample_w=fit_w,
                    tau=float(model_tau),
                    knots=int(model_knots),
                    neighbors=int(model_neighbors),
                )
            if fitted is None:
                return

            mem_model = fitted
            model_refits += 1
            _mark_phasea_index_dirty()
            _model_debuglog(
                "model_refit",
                sample_count=int(len(sample_x)),
                using_seed_anchors=bool(len(sample_x) < int(model_seed_drop_samples)),
                seed_anchor_count=int(len(seed_anchor_x)),
                seed_drop_samples=int(model_seed_drop_samples),
                refits=int(model_refits),
                latest_sample=latest_model_sample,
                probe_fof_candidates=model_probe_candidates,
                probe_pred_gb=[
                    round(
                        float(
                            _ahf_fast_predict_reservation_bytes(
                                fof_candidates=int(v),
                                model=fitted,
                                seed_bytes=_ahf_fast_seed_prediction_bytes(
                                    fof_candidates=int(v),
                                    work_unit=int(work_unit),
                                    base_unit_bytes=int(base_unit_bytes),
                                    pred_min_bytes=int(resv_min_bytes),
                                    pred_max_bytes=int(resv_max_bytes),
                                ),
                                pred_min_bytes=int(resv_min_bytes),
                                pred_max_bytes=int(resv_max_bytes),
                            )
                        )
                        / _GIB,
                        4,
                    )
                    for v in model_probe_candidates
                ],
            )
            last_model_refit_ts = float(now_ts)
            last_model_refit_completed = int(completed_hosts)
            while int(next_refit_target) < int(max_refit_target) and int(completed_hosts) >= int(next_refit_target):
                next_refit_target = _ahf_fast_next_refit_target(
                    current_target=int(next_refit_target),
                    exp_max=int(model_refit_exp_max),
                )

        def _adjust_scheduler(done_count: int = 0, force: bool = False) -> None:
            nonlocal completed_since_adjust, last_adjust_ts
            nonlocal scheduler_mode, scheduler_slope_gbps, safety_floor_bytes, capacity_bytes
            completed_since_adjust += int(done_count)
            now = _time.time()
            if (not force) and completed_since_adjust < adjust_every and (now - last_adjust_ts) < adjust_interval_s:
                return
            completed_since_adjust = 0
            last_adjust_ts = now

            _, _, avail_bytes, _ = _snapshot_memory()
            avail = int(avail_bytes) if avail_bytes is not None else -1
            if avail >= 0:
                mem_samples.append((now, avail))
            if len(mem_samples) >= 2:
                t0, a0 = mem_samples[0]
                t1, a1 = mem_samples[-1]
                dt = max(1.0e-6, float(t1) - float(t0))
                scheduler_slope_gbps = float((float(a1) - float(a0)) / dt / _GIB)
            else:
                scheduler_slope_gbps = 0.0

            safety_floor_bytes = int(safety_floor_base_bytes)
            if scheduler_slope_gbps <= -float(mem_slope_crit_gbps):
                safety_floor_bytes += int(round(32.0 * _GIB))
            elif scheduler_slope_gbps <= -float(mem_slope_warn_gbps):
                safety_floor_bytes += int(round(16.0 * _GIB))

            scheduler_mode = _ahf_fast_mode_from_memory(
                avail_bytes=int(avail),
                safety_floor_bytes=int(safety_floor_bytes),
                total_bytes=int(total_bytes),
            )

            if avail >= 0:
                capacity_bytes = max(0, int(avail) - int(safety_floor_bytes))
                avail_gb = float(avail) / _GIB
            else:
                fallback_total = int(total_bytes) if total_bytes > 0 else int(max(1, jobs) * max(1, resv_max_bytes))
                capacity_bytes = max(0, fallback_total - int(safety_floor_bytes))
                avail_gb = -1.0

            reserved_samples.append(int(reserved_bytes_total))
            phasea_scan_size = int(len(phasea_predicted) if not phasea_index_dirty else len(pending_phasea))
            mylog.info(
                "AHF-FAST: scheduler update mode=%s avail_gb=%.1f safety_gb=%.1f "
                "reserved_gb=%.1f capacity_gb=%.1f pending=%d phasea_pending=%d phasea_scan_size=%d inflight=%d "
                "completed=%d emitted=%d emit_lag=%d",
                str(scheduler_mode),
                float(avail_gb),
                float(safety_floor_bytes) / _GIB,
                float(reserved_bytes_total) / _GIB,
                float(capacity_bytes) / _GIB,
                int(len(bootstrap_phasea) + len(pending_phasea) + len(pending_tiny)),
                int(len(bootstrap_phasea) + len(pending_phasea)),
                int(len(bootstrap_phasea) + phasea_scan_size),
                int(inflight_count),
                int(completed_hosts),
                int(emitted_hosts),
                int(max(0, int(completed_hosts) - int(emitted_hosts))),
            )

        def _flush_completed(futures, block: bool = False):
            nonlocal next_to_emit, skipped_empty_payloads, inflight_count, reserved_bytes_total
            nonlocal completed_hosts, emitted_hosts
            nonlocal latest_model_sample
            nonlocal bootstrap_inflight_count
            if not futures:
                return 0
            timeout = None if block else 0
            done, _ = wait(list(futures.keys()), timeout=timeout, return_when=FIRST_COMPLETED)
            if not done:
                return 0
            done_count = 0
            for fut in done:
                fut_meta = futures.pop(fut, None)
                reserved = int(fut_meta.get("reserved_bytes", 0)) if isinstance(fut_meta, dict) else 0
                if isinstance(fut_meta, dict) and bool(fut_meta.get("bootstrap", False)):
                    bootstrap_inflight_count = max(0, int(bootstrap_inflight_count) - 1)
                inflight_count = max(0, int(inflight_count) - 1)
                reserved_bytes_total = max(0, int(reserved_bytes_total) - max(0, int(reserved)))

                order_idx, host_gals, local_skipped, host_stats = fut.result()
                completed_hosts += 1
                done_count += 1

                skipped_empty_payloads += int(local_skipped)
                if isinstance(host_stats, dict):
                    for key in fof_totals.keys():
                        if key in host_stats:
                            fof_totals[key] += int(host_stats[key])
                pending_results[order_idx] = host_gals

                if isinstance(fut_meta, dict):
                    submit_inflight = int(fut_meta.get("submit_inflight", 0))
                    submit_rss = int(fut_meta.get("submit_rss_bytes", -1))
                    predicted = int(fut_meta.get("predicted_bytes", 0))
                    fof_candidates_i = int(fut_meta.get("fof_candidates", 0))
                    rss_done, _, _, _ = _snapshot_memory()
                    rss_done = int(rss_done)
                    if submit_rss >= 0 and rss_done >= 0:
                        observed = _ahf_fast_observed_bytes_estimate(
                            submit_inflight=int(submit_inflight),
                            rss_submit_bytes=int(submit_rss),
                            rss_done_bytes=int(rss_done),
                        )
                        if observed is not None:
                            weight = _ahf_fast_sample_weight(
                                submit_inflight=int(submit_inflight),
                                anchor_max_inflight=int(anchor_max_inflight),
                                sample_max_inflight=int(model_sample_max_inflight),
                            )
                            if weight > 0.0:
                                sample_x.append(float(np.log1p(max(0, int(fof_candidates_i)))))
                                sample_y.append(float(observed))
                                sample_w.append(float(weight))
                                latest_model_sample = {
                                    "fof_candidates": int(fof_candidates_i),
                                    "observed_gb": round(float(observed) / _GIB, 6),
                                    "predicted_gb": round(float(predicted) / _GIB, 6),
                                    "sample_weight": float(weight),
                                    "submit_inflight": int(submit_inflight),
                                }
                            if predicted > 0:
                                ratio = float(observed) / float(predicted)
                                drift_samples.append(float(ratio))

            _maybe_refit_model(now_ts=_time.time(), force=False)

            while next_to_emit in pending_results:
                host_gals = pending_results.pop(next_to_emit)
                meta = host_meta.get(next_to_emit)
                host_id = int(meta.root_id) if meta is not None else -1
                is_heavy = bool(meta.is_heavy) if meta is not None else False
                if host_gals:
                    for node_id, grp in host_gals:
                        galaxies.append(grp)
                        galaxy_node_ids.append(int(node_id))
                emitted_hosts += 1
                if _trace_host(next_to_emit, is_heavy):
                    _phase_memlog(
                        "materialize",
                        order_idx=next_to_emit,
                        host_id=host_id,
                        heavy=int(is_heavy),
                        galaxies=int(len(host_gals)),
                        inflight=int(inflight_count),
                        reserved_gb=round(float(reserved_bytes_total) / _GIB, 4),
                        completed_hosts=int(completed_hosts),
                        emitted_hosts=int(emitted_hosts),
                        emit_lag=int(max(0, int(completed_hosts) - int(emitted_hosts))),
                        hosts_total=int(total_hosts),
                        pending=int(len(pending_results)),
                    )
                next_to_emit += 1
            if host_progress is not None and done_count > 0:
                host_progress.update(int(done_count))
            return int(done_count)

        def _submit_host(item: HostSchedState, launch_phase: str, reserved_for_item: int, bootstrap: bool = False) -> None:
            nonlocal inflight_count, reserved_bytes_total, bootstrap_inflight_count
            started_at = _time.time()
            submit_inflight = int(inflight_count) + 1
            rss_submit, _, _, _ = _snapshot_memory()
            item.submit_ts = float(started_at)
            item.submit_inflight = int(submit_inflight)
            item.submit_rss_bytes = int(rss_submit) if rss_submit is not None else -1
            future = executor.submit(
                process_host_task,
                int(item.order_idx),
                int(item.root_id),
                item.nodes,
                int(item.fof_candidates),
                bool(item.is_heavy),
            )
            pending_futures[future] = {
                "order_idx": int(item.order_idx),
                "phase": str(launch_phase),
                "started_at": float(started_at),
                "fof_candidates": int(item.fof_candidates),
                "predicted_bytes": int(item.predicted_bytes),
                "reserved_bytes": int(reserved_for_item),
                "submit_inflight": int(submit_inflight),
                "submit_rss_bytes": int(item.submit_rss_bytes),
                "bootstrap": bool(bootstrap),
            }
            inflight_count += 1
            reserved_bytes_total += int(reserved_for_item)
            if bool(bootstrap):
                bootstrap_inflight_count += 1

        def _submit_until_blocked() -> int:
            nonlocal phasea_fit_success_count, phasea_fit_fail_count
            nonlocal phaseb_backfill_count, deadlock_guard_count, dispatch_submit_batches
            submissions = 0
            while int(inflight_count) < int(jobs) and (bootstrap_phasea or pending_phasea or pending_tiny):
                if int(capacity_bytes) <= 0:
                    # Emergency gate: pause all new launches when memory is at/below safety floor.
                    break

                if bootstrap_phasea or int(bootstrap_inflight_count) > 0:
                    if int(inflight_count) >= int(model_bootstrap_max_inflight):
                        break
                    if not bootstrap_phasea:
                        break
                    bootstrap_item = bootstrap_phasea[0]
                    bootstrap_item.predicted_bytes = int(_predict_host_reservation_bytes(bootstrap_item))
                    remaining_capacity = int(capacity_bytes) - int(reserved_bytes_total)
                    if int(bootstrap_item.predicted_bytes) <= int(max(0, remaining_capacity)) or int(inflight_count) == 0:
                        bootstrap_phasea.popleft()
                        reserved_for_item = int(max(0, int(bootstrap_item.predicted_bytes)))
                        _submit_host(
                            item=bootstrap_item,
                            launch_phase="A",
                            reserved_for_item=reserved_for_item,
                            bootstrap=True,
                        )
                        phasea_fit_success_count += 1
                        submissions += 1
                        continue
                    break

                if phasea_index_dirty:
                    _rebuild_phasea_index()

                remaining_capacity = int(capacity_bytes) - int(reserved_bytes_total)
                if pending_phasea and remaining_capacity > 0 and phasea_predicted:
                    fit_idx = int(bisect_right(phasea_predicted, int(remaining_capacity))) - 1
                else:
                    fit_idx = -1

                if fit_idx >= 0:
                    item = _pop_phasea_at(int(fit_idx))
                    reserved_for_item = int(max(0, int(item.predicted_bytes)))
                    _submit_host(item=item, launch_phase="A", reserved_for_item=reserved_for_item)
                    phasea_fit_success_count += 1
                    submissions += 1
                    continue

                if pending_phasea:
                    phasea_fit_fail_count += 1

                if pending_phasea and int(inflight_count) == 0 and (not pending_futures):
                    if phasea_index_dirty:
                        _rebuild_phasea_index()
                    if pending_phasea:
                        item = _pop_phasea_at(0)
                        reserved_for_item = int(max(0, int(item.predicted_bytes)))
                        _submit_host(item=item, launch_phase="A", reserved_for_item=reserved_for_item)
                        phasea_fit_success_count += 1
                        deadlock_guard_count += 1
                        submissions += 1
                        continue

                free_slots = int(jobs) - int(inflight_count)
                if free_slots > 0 and pending_tiny:
                    launch_n = min(int(free_slots), int(len(pending_tiny)))
                    for _ in range(int(launch_n)):
                        tiny_item = pending_tiny.popleft()
                        _submit_host(item=tiny_item, launch_phase="B", reserved_for_item=0)
                        phaseb_backfill_count += 1
                        submissions += 1
                    continue

                break

            if submissions > 0:
                dispatch_submit_batches += 1
            return int(submissions)

        _adjust_scheduler(done_count=adjust_every, force=True)
        if scheduler_enabled and pending_phasea:
            _rebuild_phasea_index()

        while bootstrap_phasea or bootstrap_inflight_count or pending_phasea or pending_tiny or pending_futures:
            submitted = _submit_until_blocked()

            if pending_futures:
                if submitted <= 0:
                    dispatch_wakeup_count += 1
                done_now = _flush_completed(pending_futures, block=(submitted <= 0))
                _adjust_scheduler(done_count=done_now, force=False)
                continue

            if bootstrap_phasea or bootstrap_inflight_count or pending_phasea or pending_tiny:
                _time.sleep(float(dispatch_heartbeat_ms) / 1000.0)
                _adjust_scheduler(done_count=0, force=True)

    avg_reserved_gb = (
        float(sum(reserved_samples)) / float(max(1, len(reserved_samples))) / _GIB
        if reserved_samples
        else 0.0
    )
    mylog.info(
        "AHF-FAST: scheduler summary refits=%d drift_refits=%d samples=%d "
        "phaseA_fit_success=%d phaseA_fit_fail=%d phaseB_tiny_launches=%d "
        "dispatch_wakeups=%d dispatch_submit_batches=%d deadlock_guard=%d avg_reserved_gb=%.2f "
        "completed_hosts=%d emitted_hosts=%d emit_lag=%d phasea_pending=%d tiny_pending=%d next_refit_target=%d",
        int(model_refits),
        int(drift_refits),
        int(len(sample_x)),
        int(phasea_fit_success_count),
        int(phasea_fit_fail_count),
        int(phaseb_backfill_count),
        int(dispatch_wakeup_count),
        int(dispatch_submit_batches),
        int(deadlock_guard_count),
        float(avg_reserved_gb),
        int(completed_hosts),
        int(emitted_hosts),
        int(max(0, int(completed_hosts) - int(emitted_hosts))),
        int(len(bootstrap_phasea) + len(pending_phasea)),
        int(len(pending_tiny)),
        int(next_refit_target),
    )

    if host_progress is not None:
        host_progress.close()

    if skipped_empty_payloads > 0:
        mylog.warning(
            "AHF-FAST: skipped %d galaxy payload(s) with no mapped particles"
            % skipped_empty_payloads
        )

    with payload_lock:
        final_tiny_cut = int(payload_cutoffs["tiny"])
        final_huge_cut = int(payload_cutoffs["huge"])
    mylog.info(
        "AHF-FAST: payload summary payloads=%d tiny=%d normal=%d huge=%d "
        "tiny_direct=%d fof_runs=%d fof_groups=%d fof_unassigned=%d fof_empty=%d fof_fallback=%d "
        "final_cutoffs=(tiny_stars<=%d,huge_fof>=%d)",
        int(fof_totals["payloads"]),
        int(fof_totals["tiny"]),
        int(fof_totals["normal"]),
        int(fof_totals["huge"]),
        int(fof_totals["tiny_skips"]),
        int(fof_totals["fof_runs"]),
        int(fof_totals["fof_groups"]),
        int(fof_totals["fof_unassigned"]),
        int(fof_totals["fof_empty"]),
        int(fof_totals["fof_fallback"]),
        int(final_tiny_cut),
        int(final_huge_cut),
    )

    sim.galaxy_list = galaxies
    sim.ngalaxies = len(galaxies)
    sim.galaxies = galaxies
    _phase_memlog(
        "materialize_start",
        galaxies=int(sim.ngalaxies),
        hosts_total=int(total_hosts),
    )
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
            if hasattr(gal, "dmlist") and gal.dmlist is not None and len(gal.dmlist) > 0 and has_ptype(sim, "dm"):
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

    def _refresh_global_indexes(gal) -> None:
        gal.global_indexes = _compute_global_indexes(gal)

    for gal in sim.galaxy_list:
        _refresh_global_indexes(gal)

    _phase_memlog(
        "materialize_done",
        galaxies=int(sim.ngalaxies),
        hosts_total=int(total_hosts),
    )

    host_indices: List[int] = []
    if galaxy_node_ids and len(galaxy_node_ids) == sim.ngalaxies:
        galaxy_node_ids = [int(nid) if nid is not None else -1 for nid in galaxy_node_ids]
        sim._ahf_galaxy_ahf_ids = list(galaxy_node_ids)
        _update_ahf_galaxy_maps(sim, sim._ahf_galaxy_ahf_ids)

        ahf_to_halo_index: Dict[int, int] = {}
        for halo_index, halo in enumerate(sim.halo_list):
            ahf_hid = getattr(halo, "AHF_haloID", None)
            if ahf_hid is None:
                continue
            try:
                ahf_to_halo_index[int(ahf_hid)] = halo_index
            except Exception:
                continue

        if debug_fast:
            mylog.info(
                "AHF-FAST debug: nhalos=%d, ngalaxies=%d",
                len(sim.halo_list),
                sim.ngalaxies,
            )
            if debug_host is not None:
                present = any(
                    int(getattr(h, "AHF_haloID", -1)) == debug_host
                    for h in sim.halo_list
                )
                mylog.info(
                    "AHF-FAST debug: host %d present in halo_list? %s",
                    debug_host,
                    present,
                )

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
        preliminary_indices: List[int] = []
        for node_id in galaxy_node_ids:
            if node_id is None or node_id == -1:
                preliminary_indices.append(-1)
                continue
            resolved = _resolve_halo_index(int(node_id))
            if resolved < 0:
                missing_hosts.add(int(node_id))
            preliminary_indices.append(resolved)
        if missing_hosts:
            # Exclude stolen hosts (MPI boundary artifacts) from the error
            stolen = getattr(sim, '_ahf_stolen_hosts', set())
            genuine_missing = missing_hosts - stolen
            stolen_missing = missing_hosts & stolen

            if stolen_missing:
                mylog.warning(
                    "AHF-FAST: %d galaxy host(s) were stolen by other hosts (MPI artifacts). "
                    "Example IDs: %s",
                    len(stolen_missing),
                    list(sorted(stolen_missing))[:5]
                )
                # Remove orphan galaxies whose host was stolen (MPI artifacts)
                orphan_indices = [i for i, nid in enumerate(galaxy_node_ids)
                                  if nid in stolen_missing]
                if orphan_indices:
                    # Remove in reverse order to preserve indices
                    for i in sorted(orphan_indices, reverse=True):
                        del sim.galaxy_list[i]
                        del preliminary_indices[i]
                        del galaxy_node_ids[i]
                    sim.ngalaxies = len(sim.galaxy_list)
                    sim._ahf_galaxy_ahf_ids = list(galaxy_node_ids)
                    mylog.warning(
                        "AHF-FAST: Removed %d orphan galaxies whose hosts were stolen",
                        len(orphan_indices)
                    )

            if genuine_missing:
                sample = list(sorted(genuine_missing))[:5]
                if debug_fast:
                    mylog.info(
                        "AHF-FAST debug: missing_hosts=%d example=%s",
                        len(genuine_missing),
                        sample,
                    )
                # Invariants for the AHF-FAST path: every galaxy host AHF ID
                # must correspond to an existing CAESAR halo built from the
                # top-level hosts.  If we ever violate this, we want an
                # immediate, loud failure rather than silently synthesizing
                # halos from the particles file.
                raise AssertionError(
                    f"AHF-FAST invariant violated: {len(genuine_missing)} galaxy host halo(s) "
                    f"were not resolved; example IDs {sample}"
                )
        host_indices = preliminary_indices
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

    ctx = _Ctx(sim)
    prop_bar = None
    if show_progress and sim.ngalaxies > 0:
        prop_bar = tqdm(total=1, desc="Computing galaxy properties", leave=False)
    _phase_memlog(
        "properties_start",
        galaxies=int(sim.ngalaxies),
        nproc=int(getattr(ctx, "nproc", 1)),
    )
    _prop_t0 = _time.time()
    try:
        _get_group_properties(ctx, sim.galaxy_list)
    finally:
        if prop_bar is not None:
            prop_bar.update(1)
            prop_bar.close()
    _phase_memlog(
        "properties_done",
        galaxies=int(sim.ngalaxies),
        elapsed_s=round(_time.time() - _prop_t0, 3),
    )

    try:
        if "galaxy" not in sim.group_types:
            sim.group_types.append("galaxy")
    except Exception:
        pass

    if ndm_full > 0 and dm_full_lookup is not None:
        exclusive_gal_dm = np.full(ndm_full, -1, dtype=np.int32)
        for gal in sim.galaxy_list:
            dm_exc = getattr(gal, "_dm_exclusive_pids", None)
            if not dm_exc:
                continue
            mapped = dm_full_lookup.map(dm_exc)
            if mapped.size == 0:
                try:
                    del gal.__dict__["_dm_exclusive_pids"]
                except KeyError:
                    pass
                continue
            exclusive_gal_dm[mapped] = int(getattr(gal, "GroupID", -1))
            # drop the cached set to free memory
            try:
                del gal.__dict__["_dm_exclusive_pids"]
            except KeyError:
                pass
        setattr(sim, "_exclusive_galaxy_dmlist", exclusive_gal_dm)
    else:
        for gal in sim.galaxy_list:
            if "_dm_exclusive_pids" in gal.__dict__:
                del gal.__dict__["_dm_exclusive_pids"]
        if hasattr(sim, "_exclusive_galaxy_dmlist"):
            delattr(sim, "_exclusive_galaxy_dmlist")

    _phase_memlog(
        "done",
        galaxies=int(sim.ngalaxies),
        halos=int(getattr(sim, "nhalos", 0)),
    )
