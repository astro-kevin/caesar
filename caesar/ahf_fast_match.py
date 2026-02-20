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
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

    from caesar.group import create_new_group
    from caesar.group import get_group_properties as _get_group_properties
    from caesar.property_manager import get_property, has_ptype
    from caesar.AHF_FAST_loader import load_ahf_particle_blocks
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
                "lane": "light",
            }
        )
        host_workloads.append(fof_candidates)

    tri_bin_requested = (_os.environ.get("CAESAR_AHF_FAST_THREE_BIN", "1") == "1")
    tri_bin_enabled = jobs > 1 and tri_bin_requested
    heavy_inflight_limit = 0
    light_worker_limit = jobs
    medium_worker_limit = 0
    heavy_threshold = 0
    medium_threshold = 0
    heavy_count = 0
    medium_count = 0
    adaptive_enabled = False

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

    cap_init = {"light": jobs, "medium": 0, "heavy": 0}
    cap_min = {"light": 1, "medium": 0, "heavy": 0}
    cap_max = {"light": jobs, "medium": jobs, "heavy": jobs}
    adjust_every = 25
    mem_crit_gb = 120.0
    mem_low_gb = 240.0
    mem_high_gb = 340.0

    if tri_bin_enabled and host_schedule:
        medium_percentile = min(100.0, max(0.0, _env_float("CAESAR_AHF_FAST_MEDIUM_PERCENTILE", 65.0)))
        heavy_percentile = min(100.0, max(0.0, _env_float("CAESAR_AHF_FAST_HEAVY_PERCENTILE", 90.0)))

        medium_min = max(0, _env_int("CAESAR_AHF_FAST_MEDIUM_MIN_CANDIDATES", 8000))
        heavy_min = max(0, _env_int("CAESAR_AHF_FAST_HEAVY_MIN_CANDIDATES", 50000))

        workload_arr = np.asarray(host_workloads, dtype=np.float64)
        medium_cut = int(np.percentile(workload_arr, medium_percentile))
        heavy_cut = int(np.percentile(workload_arr, heavy_percentile))
        medium_threshold = max(medium_min, medium_cut)
        heavy_threshold = max(heavy_min, heavy_cut, medium_threshold)

        for item in host_schedule:
            w = int(item["fof_candidates"])
            if w >= heavy_threshold:
                lane = "heavy"
            elif w >= medium_threshold:
                lane = "medium"
            else:
                lane = "light"
            item["lane"] = lane
            item["is_heavy"] = lane == "heavy"

        heavy_count = int(sum(1 for item in host_schedule if item["lane"] == "heavy"))
        medium_count = int(sum(1 for item in host_schedule if item["lane"] == "medium"))
        light_count = int(len(host_schedule) - heavy_count - medium_count)

        heavy_default = 1 if heavy_count > 0 else 0
        heavy_inflight_limit = max(0, min(_env_int("CAESAR_AHF_FAST_HEAVY_INFLIGHT", heavy_default), jobs - 1))
        medium_default = 0
        if medium_count > 0:
            medium_default = max(1, min(max(2, jobs // 10), max(1, jobs - heavy_inflight_limit - 1)))
        medium_worker_limit = max(0, min(_env_int("CAESAR_AHF_FAST_MEDIUM_INFLIGHT", medium_default), jobs))

        light_worker_limit = max(1, jobs - heavy_inflight_limit - medium_worker_limit)
        spill = max(0, (heavy_inflight_limit + medium_worker_limit + light_worker_limit) - jobs)
        if spill > 0:
            take_m = min(spill, max(0, medium_worker_limit))
            medium_worker_limit -= take_m
            spill -= take_m
            if spill > 0:
                heavy_inflight_limit = max(0, heavy_inflight_limit - spill)
        light_worker_limit = max(1, jobs - heavy_inflight_limit - medium_worker_limit)

        cap_init = {
            "light": int(light_worker_limit),
            "medium": int(medium_worker_limit),
            "heavy": int(heavy_inflight_limit),
        }

        heavy_min_cap_default = 0
        heavy_max_cap_default = max(heavy_inflight_limit, min(4, max(0, jobs - 1)))
        medium_min_cap_default = 0
        medium_max_cap_default = max(medium_worker_limit, min(max(4, jobs // 3), max(0, jobs - 1)))

        cap_min["heavy"] = max(0, min(_env_int("CAESAR_AHF_FAST_HEAVY_INFLIGHT_MIN", heavy_min_cap_default), jobs))
        cap_max["heavy"] = max(cap_min["heavy"], min(_env_int("CAESAR_AHF_FAST_HEAVY_INFLIGHT_MAX", heavy_max_cap_default), jobs))
        cap_min["medium"] = max(0, min(_env_int("CAESAR_AHF_FAST_MEDIUM_INFLIGHT_MIN", medium_min_cap_default), jobs))
        cap_max["medium"] = max(cap_min["medium"], min(_env_int("CAESAR_AHF_FAST_MEDIUM_INFLIGHT_MAX", medium_max_cap_default), jobs))
        cap_min["light"] = 1
        cap_max["light"] = jobs

        if heavy_count == 0:
            cap_init["heavy"] = 0
            cap_min["heavy"] = 0
            cap_max["heavy"] = 0
        if medium_count == 0:
            cap_init["medium"] = 0
            cap_min["medium"] = 0
            cap_max["medium"] = 0
        cap_init["light"] = max(1, jobs - cap_init["heavy"] - cap_init["medium"])

        adaptive_enabled = (_os.environ.get("CAESAR_AHF_FAST_ADAPTIVE", "1") == "1")
        adjust_every = max(1, _env_int("CAESAR_AHF_FAST_ADJUST_EVERY", 25))

        # Obtain total memory once to set scale-aware defaults when thresholds
        # are not explicitly configured.
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

        heavy_examples = sorted(
            (
                (int(item["fof_candidates"]), int(item["star_count"]), int(item["root_id"]))
                for item in host_schedule
                if item["lane"] == "heavy"
            ),
            reverse=True,
        )[:5]
        medium_examples = sorted(
            (
                (int(item["fof_candidates"]), int(item["star_count"]), int(item["root_id"]))
                for item in host_schedule
                if item["lane"] == "medium"
            ),
            reverse=True,
        )[:5]

        mylog.info(
            "AHF-FAST: tri-bin scheduler enabled: hosts=%d light=%d medium=%d heavy=%d "
            "(thresholds medium=%d [p=%.1f,min=%d], heavy=%d [p=%.1f,min=%d]), "
            "caps(light=%d, medium=%d, heavy=%d), adaptive=%d",
            len(host_schedule),
            light_count,
            medium_count,
            heavy_count,
            medium_threshold,
            medium_percentile,
            medium_min,
            heavy_threshold,
            heavy_percentile,
            heavy_min,
            cap_init["light"],
            cap_init["medium"],
            cap_init["heavy"],
            int(adaptive_enabled),
        )
        mylog.info(
            "AHF-FAST: medium host examples (fof_candidates, stars, host_id): %s",
            medium_examples,
        )
        mylog.info(
            "AHF-FAST: heavy host examples (fof_candidates, stars, host_id): %s",
            heavy_examples,
        )
        if adaptive_enabled:
            mylog.info(
                "AHF-FAST: adaptive caps adjust_every=%d headroom_gb=(crit=%.1f,low=%.1f,high=%.1f) "
                "bounds medium=[%d,%d] heavy=[%d,%d]",
                adjust_every,
                mem_crit_gb,
                mem_low_gb,
                mem_high_gb,
                cap_min["medium"],
                cap_max["medium"],
                cap_min["heavy"],
                cap_max["heavy"],
            )

    if not tri_bin_enabled:
        mylog.info(
            "AHF-FAST: single-lane scheduler (workers=%d, hosts=%d)",
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
        two_lane=0,
        tri_bin=int(tri_bin_enabled),
        heavy_hosts=heavy_count,
        medium_hosts=medium_count,
        heavy_threshold=heavy_threshold,
        medium_threshold=medium_threshold,
        light_workers=light_worker_limit,
        medium_workers=medium_worker_limit,
        heavy_workers=heavy_inflight_limit,
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

    def _apply_dense_gas_gate(gal) -> None:
        """Optionally restrict galaxy gas to dense/cool/SF gas before computing properties.

        Mirrors the gate used by 6D-FOF:
        nH > 0.13 and (T < 1e5 or SFR > 0).
        """
        try:
            gidx = getattr(gal, "glist", None)
            if gidx is None or len(gidx) == 0:
                return
            dm = getattr(sim, "data_manager", None)
            if dm is None:
                return
            gnh = getattr(dm, "gnh", None)
            gT = getattr(dm, "gT", None)
            gsfr = getattr(dm, "gsfr", None)
            if gnh is None or gT is None or gsfr is None:
                return
            gidx_arr = np.asarray(gidx, dtype=np.int64)
            nh = gnh[gidx_arr]
            temp = gT[gidx_arr]
            sfr = gsfr[gidx_arr]
            nHlim = 0.13
            Tlim = 1.0e5
            mask = (nh > nHlim) & ((temp < Tlim) | (sfr > 0))
            gal.glist = gidx_arr[mask]
        except Exception:
            # If anything goes wrong, leave glist unchanged.
            return

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
            return order_idx, [], 0

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
            return order_idx, [], 0

        host_galaxies: List[Tuple[int, object]] = []
        local_skipped = 0

        for payload in payloads:
            node_id, star_set, gas_set, bh_set, dm_exc = payload
            dm_pm = bucket.get(node_id)
            dm_inclusive = dm_pm.parttype1 if dm_pm is not None else set()
            grp = create_new_group(sim, "galaxy")
            grp.AHF_haloID = int(node_id)
            grp.slist = map_sel(star_set, "star")
            grp.glist = map_sel(gas_set, "gas")
            _apply_dense_gas_gate(grp)
            if "bh" in pid_maps_sel:
                grp.bhlist = map_sel(bh_set, "bh")
            if "dm" in pid_maps_sel:
                dm_selected = map_sel(dm_inclusive, "dm")
            else:
                dm_selected = np.array([], dtype=np.int32)
            grp.dmlist = dm_selected
            grp.global_indexes = np.array([], dtype=np.int64)
            if dm_exc:
                grp.__dict__["_dm_exclusive_pids"] = set(dm_exc)
            else:
                grp.__dict__["_dm_exclusive_pids"] = set()

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
                payloads=int(len(payloads)),
                galaxies=int(len(host_galaxies)),
                skipped=int(local_skipped),
            )

        return order_idx, host_galaxies, local_skipped

    pending_results: Dict[int, List] = {}
    host_meta = {int(item["order_idx"]): item for item in host_schedule}
    next_to_emit = 0
    inflight_by_lane = {"light": 0, "medium": 0, "heavy": 0}

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
        nonlocal next_to_emit, skipped_empty_payloads
        if not futures:
            return 0
        timeout = None if block else 0
        done, _ = wait(list(futures.keys()), timeout=timeout, return_when=FIRST_COMPLETED)
        if not done:
            return 0
        for fut in done:
            lane, _ = futures.pop(fut, (None, None))
            if lane not in inflight_by_lane:
                lane = "light"
            inflight_by_lane[lane] = max(0, inflight_by_lane[lane] - 1)
            order_idx, host_gals, local_skipped = fut.result()
            skipped_empty_payloads += int(local_skipped)
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
                    inflight_light=int(inflight_by_lane["light"]),
                    inflight_medium=int(inflight_by_lane["medium"]),
                    inflight_heavy=int(inflight_by_lane["heavy"]),
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
        if tri_bin_enabled:
            from collections import deque as _deque

            lane_queues = {
                "heavy": _deque([item for item in host_schedule if item.get("lane") == "heavy"]),
                "medium": _deque([item for item in host_schedule if item.get("lane") == "medium"]),
                "light": _deque([item for item in host_schedule if item.get("lane") == "light"]),
            }
            caps = {
                "light": int(cap_init.get("light", jobs)),
                "medium": int(cap_init.get("medium", 0)),
                "heavy": int(cap_init.get("heavy", 0)),
            }
            completed_since_adjust = 0

            def _rebalance_caps() -> None:
                active_lanes = [
                    lane for lane in ("light", "medium", "heavy")
                    if lane_queues[lane] or inflight_by_lane[lane] > 0
                ]
                if not active_lanes:
                    caps["light"] = 0
                    caps["medium"] = 0
                    caps["heavy"] = 0
                    return

                min_caps = {"light": 0, "medium": 0, "heavy": 0}
                for lane in ("light", "medium", "heavy"):
                    if lane not in active_lanes:
                        min_caps[lane] = 0
                        continue
                    lane_min = int(cap_min.get(lane, 0))
                    if lane == "light":
                        lane_min = max(1, lane_min)
                    min_caps[lane] = min(lane_min, int(cap_max.get(lane, jobs)))

                total_min = sum(min_caps.values())
                if total_min > jobs:
                    for lane in ("heavy", "medium", "light"):
                        floor = 1 if (lane == "light" and lane in active_lanes) else 0
                        reducible = max(0, min_caps[lane] - floor)
                        if reducible <= 0:
                            continue
                        take = min(reducible, total_min - jobs)
                        min_caps[lane] -= take
                        total_min -= take
                        if total_min <= jobs:
                            break

                desired = {"light": 0, "medium": 0, "heavy": 0}
                for lane in ("light", "medium", "heavy"):
                    if lane in active_lanes:
                        cap_hi = int(cap_max.get(lane, jobs))
                        desired[lane] = max(min_caps[lane], min(cap_hi, int(caps.get(lane, 0))))
                    if desired[lane] < inflight_by_lane[lane]:
                        desired[lane] = inflight_by_lane[lane]

                total = sum(desired.values())
                if total > jobs:
                    for lane in ("heavy", "medium", "light"):
                        floor = max(min_caps[lane], inflight_by_lane[lane])
                        reducible = max(0, desired[lane] - floor)
                        if reducible <= 0:
                            continue
                        take = min(reducible, total - jobs)
                        desired[lane] -= take
                        total -= take
                        if total <= jobs:
                            break

                if total < jobs:
                    spare = jobs - total
                    for lane in ("light", "medium", "heavy"):
                        if spare <= 0:
                            break
                        if not lane_queues[lane]:
                            continue
                        room = max(0, int(cap_max.get(lane, jobs)) - desired[lane])
                        if room <= 0:
                            continue
                        give = min(room, spare)
                        desired[lane] += give
                        spare -= give

                caps.update(desired)

            def _maybe_adjust_caps(done_count: int = 0, force: bool = False) -> None:
                nonlocal completed_since_adjust
                if not adaptive_enabled:
                    _rebalance_caps()
                    return

                completed_since_adjust += int(done_count)
                if (not force) and completed_since_adjust < adjust_every:
                    return
                completed_since_adjust = 0

                old_caps = dict(caps)
                _, _, avail_bytes, _ = _snapshot_memory()
                avail_gb = (avail_bytes / 2**30) if avail_bytes is not None and avail_bytes >= 0 else -1.0

                if avail_gb >= 0:
                    if avail_gb <= mem_crit_gb:
                        caps["heavy"] = 0
                        if caps["medium"] > max(cap_min.get("medium", 0), 1 if lane_queues["medium"] else 0):
                            caps["medium"] -= 1
                    elif avail_gb < mem_low_gb:
                        if caps["heavy"] > int(cap_min.get("heavy", 0)):
                            caps["heavy"] -= 1
                        elif caps["medium"] > int(cap_min.get("medium", 0)):
                            caps["medium"] -= 1
                    elif avail_gb > mem_high_gb:
                        if lane_queues["heavy"] and caps["heavy"] < int(cap_max.get("heavy", jobs)):
                            caps["heavy"] += 1
                        elif lane_queues["medium"] and caps["medium"] < int(cap_max.get("medium", jobs)):
                            caps["medium"] += 1

                _rebalance_caps()
                if caps != old_caps:
                    mylog.info(
                        "AHF-FAST: adaptive caps update avail=%.1fGB "
                        "pending(light=%d,medium=%d,heavy=%d) "
                        "inflight(light=%d,medium=%d,heavy=%d) "
                        "caps(light=%d,medium=%d,heavy=%d)",
                        avail_gb,
                        len(lane_queues["light"]),
                        len(lane_queues["medium"]),
                        len(lane_queues["heavy"]),
                        inflight_by_lane["light"],
                        inflight_by_lane["medium"],
                        inflight_by_lane["heavy"],
                        caps["light"],
                        caps["medium"],
                        caps["heavy"],
                    )

            _rebalance_caps()
            _maybe_adjust_caps(done_count=adjust_every, force=True)

            while (
                lane_queues["heavy"]
                or lane_queues["medium"]
                or lane_queues["light"]
                or pending_futures
            ):
                submitted = False

                while True:
                    made_progress = False
                    inflight_total = (
                        inflight_by_lane["light"]
                        + inflight_by_lane["medium"]
                        + inflight_by_lane["heavy"]
                    )
                    if inflight_total >= jobs:
                        break

                    for lane in ("heavy", "medium", "light"):
                        if not lane_queues[lane]:
                            continue
                        if inflight_by_lane[lane] >= int(caps.get(lane, 0)):
                            continue
                        item = lane_queues[lane].popleft()
                        bucket = build_bucket(item["nodes"])
                        future = executor.submit(
                            process_host,
                            int(item["order_idx"]),
                            int(item["root_id"]),
                            bucket,
                            int(item["fof_candidates"]),
                            bool(item.get("lane") == "heavy"),
                        )
                        pending_futures[future] = (lane, int(item["order_idx"]))
                        inflight_by_lane[lane] += 1
                        submitted = True
                        made_progress = True

                        inflight_total += 1
                        if inflight_total >= jobs:
                            break
                    if not made_progress:
                        break

                if pending_futures:
                    done_now = flush_completed(pending_futures, block=(not submitted))
                    _maybe_adjust_caps(done_count=done_now, force=False)
                elif lane_queues["heavy"] or lane_queues["medium"] or lane_queues["light"]:
                    _maybe_adjust_caps(done_count=adjust_every, force=True)

            while pending_futures:
                done_now = flush_completed(pending_futures, block=True)
                _maybe_adjust_caps(done_count=done_now, force=False)
        else:
            for item in host_schedule:
                bucket = build_bucket(item["nodes"])
                future = executor.submit(
                    process_host,
                    int(item["order_idx"]),
                    int(item["root_id"]),
                    bucket,
                    int(item["fof_candidates"]),
                    False,
                )
                pending_futures[future] = ("light", int(item["order_idx"]))
                inflight_by_lane["light"] += 1
                flush_completed(pending_futures, block=False)

            while pending_futures:
                flush_completed(pending_futures, block=True)

    if host_progress is not None:
        host_progress.close()

    if skipped_empty_payloads > 0:
        mylog.warning(
            "AHF-FAST: skipped %d galaxy payload(s) with no mapped particles"
            % skipped_empty_payloads
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
