"""AHF-FAST-specific galaxy construction entry point.

This module holds the full implementation of the AHF-FAST galaxy builder
so that the fast path can evolve independently of the generic matching
utilities in :mod:`caesar.halo_matching`.  The :mod:`halo_matching`
module re-exports a thin wrapper for backwards compatibility, but the
source of truth lives here.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


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
    from caesar.AHF_FAST_loader import load_ahf_particle_blocks
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

    # Optional debug controls for the FAST path
    debug_fast = _os.environ.get("CAESAR_AHF_FAST_DEBUG", "0") == "1"
    debug_host: Optional[int]
    try:
        _h = _os.environ.get("CAESAR_AHF_FAST_DEBUG_HOST")
        debug_host = int(_h) if _h not in (None, "") else None
    except Exception:
        debug_host = None

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
    )

    def _estimate_host_fof_candidates(nodes_for_host: Set[int]) -> Tuple[int, int]:
        """Estimate host workload from AHF memberships.

        Returns
        -------
        fof_candidates : int
            Number of particles that can participate in galaxy FOF
            (gas + stars + BH across all nodes in the host tree).
        stars : int
            Number of star particles across all nodes in the host tree.
        """
        if not membership_arrays:
            return 0, 0

        fof_candidates = 0
        stars = 0
        for node_id in nodes_for_host:
            arr = membership_arrays.get(int(node_id))
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.int64)
            if arr.size == 0:
                continue
            if arr.ndim != 2 or arr.shape[1] != 2:
                arr = arr.reshape(-1, 2)
            ptypes = arr[:, 1]
            stars += int(np.sum(ptypes == 4))
            fof_candidates += int(np.sum((ptypes == 0) | (ptypes == 4) | (ptypes == 5)))

        return fof_candidates, stars

    host_schedule: List[Dict] = []
    host_workloads: List[int] = []
    for order_idx, (root_id, nodes_for_host) in enumerate(host_to_nodes.items()):
        fof_candidates, star_count = _estimate_host_fof_candidates(nodes_for_host)
        host_schedule.append(
            {
                "order_idx": order_idx,
                "root_id": root_id,
                "nodes": nodes_for_host,
                "fof_candidates": fof_candidates,
                "star_count": star_count,
                "is_heavy": False,
                "work_units": 1,
            }
        )
        host_workloads.append(fof_candidates)

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

    adaptive_enabled = (_os.environ.get("CAESAR_AHF_FAST_ADAPTIVE", "1") == "1")
    scheduler_enabled = jobs > 1 and bool(host_schedule)
    heavy_threshold = 0
    heavy_count = 0

    # Adaptive scheduler controls.
    work_unit = 1
    scan_window = max(8, _env_int("CAESAR_AHF_FAST_SCAN_WINDOW", max(32, jobs * 4)))
    adjust_every = max(1, _env_int("CAESAR_AHF_FAST_ADJUST_EVERY", max(2, jobs // 2)))
    adjust_interval_s = max(0.2, _env_float("CAESAR_AHF_FAST_ADJUST_INTERVAL_S", 1.0))
    runtime_ewma_alpha = min(0.95, max(0.05, _env_float("CAESAR_AHF_FAST_RUNTIME_EWMA_ALPHA", 0.25)))
    mem_drop_warn_gbps = max(0.1, _env_float("CAESAR_AHF_FAST_MEM_DROP_WARN_GBPS", 1.0))
    mem_drop_crit_gbps = max(mem_drop_warn_gbps, _env_float("CAESAR_AHF_FAST_MEM_DROP_CRIT_GBPS", 2.5))
    mem_trend_window = max(3, _env_int("CAESAR_AHF_FAST_MEM_TREND_WINDOW", 8))

    worker_scale = {
        "critical": min(1.0, max(0.05, _env_float("CAESAR_AHF_FAST_WORKERS_CRIT_SCALE", 0.35))),
        "low": min(1.0, max(0.10, _env_float("CAESAR_AHF_FAST_WORKERS_LOW_SCALE", 0.60))),
        "nominal": min(1.0, max(0.20, _env_float("CAESAR_AHF_FAST_WORKERS_NOMINAL_SCALE", 1.0))),
        "high": min(1.0, max(0.20, _env_float("CAESAR_AHF_FAST_WORKERS_HIGH_SCALE", 1.0))),
    }
    budget_scale = {
        "critical": max(1.0, _env_float("CAESAR_AHF_FAST_WORK_BUDGET_SCALE_CRIT", 1.0)),
        "low": max(1.0, _env_float("CAESAR_AHF_FAST_WORK_BUDGET_SCALE_LOW", 1.25)),
        "nominal": max(1.0, _env_float("CAESAR_AHF_FAST_WORK_BUDGET_SCALE_NOMINAL", 1.8)),
        "high": max(1.0, _env_float("CAESAR_AHF_FAST_WORK_BUDGET_SCALE_HIGH", 2.3)),
    }

    # Obtain total memory once to set scale-aware defaults when thresholds are
    # not explicitly configured.
    if _psutil is None or _proc is None:
        try:
            import psutil as _ps
            _psutil = _ps
            _proc = _psutil.Process()
        except Exception:
            pass
    _, _, _avail_b0, _total_b0 = _snapshot_memory()
    total_gb = (_total_b0 / 2**30) if _total_b0 and _total_b0 > 0 else 0.0
    crit_default = max(32.0, total_gb * 0.15) if total_gb > 0 else 120.0
    low_default = max(64.0, total_gb * 0.25) if total_gb > 0 else 240.0
    high_default = max(96.0, total_gb * 0.35) if total_gb > 0 else 340.0
    mem_crit_gb = _env_float("CAESAR_AHF_FAST_MEM_HEADROOM_CRIT_GB", crit_default)
    mem_low_gb = _env_float("CAESAR_AHF_FAST_MEM_HEADROOM_LOW_GB", low_default)
    mem_high_gb = _env_float("CAESAR_AHF_FAST_MEM_HEADROOM_HIGH_GB", high_default)
    if mem_low_gb < mem_crit_gb:
        mem_low_gb = mem_crit_gb
    if mem_high_gb < mem_low_gb:
        mem_high_gb = mem_low_gb

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

    for item in host_schedule:
        w = int(item["fof_candidates"])
        item["is_heavy"] = bool(w >= heavy_threshold)
        # Sub-linear work units preserve scale differences without making
        # high-work hosts effectively single-threaded.
        scaled = float(max(1, w)) / float(max(1, work_unit))
        item["work_units"] = int(min(max(1, np.ceil(np.sqrt(max(1.0, scaled)))), max(2, jobs * 2)))

    heavy_count = int(sum(1 for item in host_schedule if item["is_heavy"]))
    heavy_examples = sorted(
        (
            (
                int(item["fof_candidates"]),
                int(item["work_units"]),
                int(item["star_count"]),
                int(item["root_id"]),
            )
            for item in host_schedule
            if item["is_heavy"]
        ),
        reverse=True,
    )[:5]

    if scheduler_enabled:
        mylog.info(
            "AHF-FAST: adaptive scheduler enabled: hosts=%d workers=%d work_unit=%d scan_window=%d adaptive=%d",
            len(host_schedule),
            jobs,
            work_unit,
            scan_window,
            int(adaptive_enabled),
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
        if adaptive_enabled:
            mylog.info(
                "AHF-FAST: controller adjust_every=%d adjust_interval_s=%.2f ewma_alpha=%.2f "
                "headroom_gb=(crit=%.1f,low=%.1f,high=%.1f) mem_drop_gbps=(warn=%.2f,crit=%.2f) "
                "worker_scale=(crit=%.2f,low=%.2f,nominal=%.2f,high=%.2f) "
                "work_budget_scale=(crit=%.2f,low=%.2f,nominal=%.2f,high=%.2f)",
                adjust_every,
                adjust_interval_s,
                runtime_ewma_alpha,
                mem_crit_gb,
                mem_low_gb,
                mem_high_gb,
                mem_drop_warn_gbps,
                mem_drop_crit_gbps,
                worker_scale["critical"],
                worker_scale["low"],
                worker_scale["nominal"],
                worker_scale["high"],
                budget_scale["critical"],
                budget_scale["low"],
                budget_scale["nominal"],
                budget_scale["high"],
            )
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
        scheduler_mode="adaptive_global" if scheduler_enabled else "serial",
        scan_window=int(scan_window),
        work_unit=int(work_unit),
        heavy_hosts=heavy_count,
        heavy_threshold=heavy_threshold,
        adaptive=int(adaptive_enabled),
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

    tiny_star_threshold = 32
    huge_q = min(0.99, max(0.55, _env_float("CAESAR_AHF_FAST_BIN_HUGE_Q", 0.85)))
    payload_update_every = max(1, _env_int("CAESAR_AHF_FAST_BIN_UPDATE_EVERY", 32))
    payload_warmup = max(8, _env_int("CAESAR_AHF_FAST_BIN_WARMUP", max(16, jobs * 4)))
    payload_sample_cap = max(128, _env_int("CAESAR_AHF_FAST_BIN_SAMPLE_CAP", 4096))
    payload_samples = _deque(maxlen=payload_sample_cap)
    if positive_workloads:
        for w in positive_workloads:
            payload_samples.append(max(0, int(w)))
    if not payload_samples:
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

    def _classify_payload_size(star_particle_count: int, fof_particle_count: int) -> Tuple[str, int, int]:
        stars = max(0, int(star_particle_count))
        count = max(0, int(fof_particle_count))
        with payload_lock:
            if stars > int(tiny_star_threshold):
                payload_samples.append(count)
                n = len(payload_samples)
                if n <= payload_warmup or (n % payload_update_every) == 0:
                    _refresh_payload_cutoffs_locked()
            else:
                payload_cutoffs["tiny"] = int(tiny_star_threshold)
            tiny_cut = int(payload_cutoffs["tiny"])
            huge_cut = int(payload_cutoffs["huge"])
        if stars <= tiny_cut:
            return "tiny", tiny_cut, huge_cut
        if count >= huge_cut:
            return "huge", tiny_cut, huge_cut
        return "normal", tiny_cut, huge_cut

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
            mode, tiny_cut, huge_cut = _classify_payload_size(star_particle_count, fof_particle_count)
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
    host_meta = {int(item["order_idx"]): item for item in host_schedule}
    next_to_emit = 0
    inflight_count = 0
    inflight_work = 0
    runtime_stats = {"done": 0, "ewma_s": 0.0, "last_s": 0.0}
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

    def flush_completed(futures, block: bool = False):
        nonlocal next_to_emit, skipped_empty_payloads, inflight_count, inflight_work, fof_totals
        if not futures:
            return 0
        timeout = None if block else 0
        done, _ = wait(list(futures.keys()), timeout=timeout, return_when=FIRST_COMPLETED)
        if not done:
            return 0
        for fut in done:
            fut_meta = futures.pop(fut, None)
            started_at = None
            submitted_units = 0
            if isinstance(fut_meta, dict):
                started_at = fut_meta.get("started_at")
                try:
                    submitted_units = int(fut_meta.get("work_units", 0))
                except Exception:
                    submitted_units = 0
            inflight_count = max(0, int(inflight_count) - 1)
            if submitted_units > 0:
                inflight_work = max(0, int(inflight_work) - submitted_units)
            if started_at is not None:
                try:
                    elapsed_s = max(0.0, _time.time() - float(started_at))
                    prev = float(runtime_stats.get("ewma_s", 0.0))
                    if prev <= 0.0:
                        runtime_stats["ewma_s"] = elapsed_s
                    else:
                        runtime_stats["ewma_s"] = (
                            float(runtime_ewma_alpha) * elapsed_s
                            + (1.0 - float(runtime_ewma_alpha)) * prev
                        )
                    runtime_stats["last_s"] = elapsed_s
                    runtime_stats["done"] = int(runtime_stats.get("done", 0)) + 1
                except Exception:
                    pass
            order_idx, host_gals, local_skipped, host_stats = fut.result()
            skipped_empty_payloads += int(local_skipped)
            if isinstance(host_stats, dict):
                for key in fof_totals.keys():
                    if key in host_stats:
                        fof_totals[key] += int(host_stats[key])
            pending_results[order_idx] = host_gals
        while next_to_emit in pending_results:
            host_gals = pending_results.pop(next_to_emit)
            meta = host_meta.get(next_to_emit, {})
            host_id = int(meta.get("root_id", -1))
            is_heavy = bool(meta.get("is_heavy", False))
            if host_gals:
                for node_id, grp in host_gals:
                    galaxies.append(grp)
                    galaxy_node_ids.append(int(node_id))
            if _trace_host(next_to_emit, is_heavy):
                _phase_memlog(
                    "materialize",
                    order_idx=next_to_emit,
                    host_id=host_id,
                    heavy=int(is_heavy),
                    galaxies=int(len(host_gals)),
                    inflight=int(inflight_count),
                    inflight_work=int(inflight_work),
                    hosts_done=int(next_to_emit + 1),
                    hosts_total=int(total_hosts),
                    pending=int(len(pending_results)),
                )
            if host_progress is not None:
                host_progress.update(1)
            next_to_emit += 1
        return len(done)

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        pending_futures: Dict = {}

        pending_hosts = _deque(host_schedule)
        pending_work = int(sum(int(item.get("work_units", 1)) for item in pending_hosts))
        completed_since_adjust = 0
        last_adjust_ts = 0.0
        target_workers = max(1, min(jobs, jobs if scheduler_enabled else 1))
        work_budget = max(target_workers, int(round(target_workers * float(budget_scale.get("nominal", 1.8)))))
        mem_samples = _deque(maxlen=max(3, int(mem_trend_window)))

        def _mem_mode_from_state(avail_gb: float, slope_gbps: float) -> str:
            mode = "nominal"
            if avail_gb >= 0:
                if avail_gb <= mem_crit_gb:
                    mode = "critical"
                elif avail_gb < mem_low_gb:
                    mode = "low"
                elif avail_gb >= mem_high_gb:
                    mode = "high"
            if slope_gbps <= -float(mem_drop_warn_gbps):
                mode = {
                    "high": "nominal",
                    "nominal": "low",
                    "low": "critical",
                    "critical": "critical",
                }[mode]
            if slope_gbps <= -float(mem_drop_crit_gbps):
                mode = {
                    "high": "low",
                    "nominal": "critical",
                    "low": "critical",
                    "critical": "critical",
                }[mode]
            return mode

        def _adjust_scheduler(done_count: int = 0, force: bool = False) -> None:
            nonlocal completed_since_adjust, last_adjust_ts, target_workers, work_budget
            if not scheduler_enabled:
                target_workers = 1
                work_budget = max(1, int(inflight_work))
                return
            if not adaptive_enabled:
                target_workers = jobs
                work_budget = max(target_workers, int(round(target_workers * float(budget_scale.get("nominal", 1.8)))))
                return

            completed_since_adjust += int(done_count)
            now = _time.time()
            if (not force) and completed_since_adjust < adjust_every and (now - last_adjust_ts) < adjust_interval_s:
                return
            completed_since_adjust = 0
            last_adjust_ts = now

            old_workers = int(target_workers)
            old_budget = int(work_budget)

            _, _, avail_bytes, _ = _snapshot_memory()
            avail_gb = (avail_bytes / 2**30) if avail_bytes is not None and avail_bytes >= 0 else -1.0
            slope_gbps = 0.0
            if avail_gb >= 0:
                mem_samples.append((now, float(avail_gb)))
                if len(mem_samples) >= 2:
                    t0, a0 = mem_samples[0]
                    t1, a1 = mem_samples[-1]
                    dt = max(1.0e-6, float(t1) - float(t0))
                    slope_gbps = float(a1 - a0) / dt

            mode = _mem_mode_from_state(float(avail_gb), float(slope_gbps))
            worker_frac = float(worker_scale.get(mode, 1.0))
            target_workers = max(1, min(jobs, int(round(float(jobs) * worker_frac))))

            avg_pending_units = (float(pending_work) / float(len(pending_hosts))) if pending_hosts else 1.0
            mix_boost = 1.0 + min(0.60, max(0.0, (avg_pending_units - 1.0) / 10.0))
            budget_per_worker = float(budget_scale.get(mode, budget_scale.get("nominal", 1.8)))
            work_budget = int(max(target_workers, round(float(target_workers) * budget_per_worker * mix_boost)))
            if work_budget < int(inflight_work):
                work_budget = int(inflight_work)

            if target_workers != old_workers or work_budget != old_budget:
                mylog.info(
                    "AHF-FAST: scheduler update mode=%s avail=%.1fGB slope=%.3fGB/s "
                    "pending=%d inflight=%d target_workers=%d work_budget=%d inflight_work=%d runtime_ewma_s=%.3f",
                    mode,
                    avail_gb,
                    slope_gbps,
                    int(len(pending_hosts)),
                    int(inflight_count),
                    int(target_workers),
                    int(work_budget),
                    int(inflight_work),
                    float(runtime_stats.get("ewma_s", 0.0)),
                )

        def _pop_best_host(max_units: int, allow_oversize: bool = False):
            if not pending_hosts:
                return None
            window = min(int(len(pending_hosts)), int(scan_window))
            best_idx = -1
            best_units = -1
            for idx in range(window):
                item = pending_hosts[idx]
                units = int(item.get("work_units", 1))
                if units <= int(max_units) and units > best_units:
                    best_idx = idx
                    best_units = units
            if best_idx < 0:
                if not allow_oversize:
                    return None
                best_idx = 0
            pending_hosts.rotate(-best_idx)
            item = pending_hosts.popleft()
            pending_hosts.rotate(best_idx)
            return item

        _adjust_scheduler(done_count=adjust_every, force=True)

        while pending_hosts or pending_futures:
            _adjust_scheduler(done_count=0, force=False)
            submitted = False

            while pending_hosts and inflight_count < int(target_workers):
                remaining_budget = int(work_budget) - int(inflight_work)
                allow_oversize = inflight_count == 0
                item = _pop_best_host(max_units=max(0, remaining_budget), allow_oversize=allow_oversize)
                if item is None:
                    break

                units_needed = int(item.get("work_units", 1))
                pending_work = max(0, int(pending_work) - units_needed)
                bucket = build_bucket(item["nodes"])
                future = executor.submit(
                    process_host,
                    int(item["order_idx"]),
                    int(item["root_id"]),
                    bucket,
                    int(item["fof_candidates"]),
                    bool(item.get("is_heavy", False)),
                )
                pending_futures[future] = {
                    "order_idx": int(item["order_idx"]),
                    "started_at": _time.time(),
                    "work_units": int(units_needed),
                }
                inflight_count += 1
                inflight_work += int(units_needed)
                submitted = True

            if pending_futures:
                done_now = flush_completed(pending_futures, block=(not submitted))
                _adjust_scheduler(done_count=done_now, force=False)

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
