from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "caesar"


def _load_module(name: str, filename: str):
    module_path = ROOT / filename
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


subhalo_mod = _load_module("ahf_subhalo_mod", "AHF_subhalo.py")
graph_mod = _load_module("fof6d_graph_mod", "fof6d_graph.py")
saver_mod = _load_module("saver_subhalo_mod", "saver.py")
loader_mod = _load_module("loader_subhalo_mod", "loader.py")
modes_mod = _load_module("modes_subhalo_mod", "modes.py")
mpi_mod = _load_module("ahf_subhalo_mpi_mod", "AHF_subhalo_mpi.py")


class DummyGalaxy:
    def __init__(self, halo_id: int, top_id: int, ancestors):
        self.AHF_haloID = halo_id
        self.AHF_top_haloID = top_id
        self.AHF_ancestor_haloIDs = np.asarray(ancestors, dtype=np.int64)
        self.glist = np.array([], dtype=np.int64)
        self.slist = np.array([], dtype=np.int64)
        self.cloud_index_list = np.array([], dtype=np.int64)


class DummyLookup:
    def map(self, pidset):
        return np.asarray(list(pidset), dtype=np.int32)


class DummyDataManager:
    def __init__(self, pos, vel):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.vel = np.asarray(vel, dtype=np.float64)

    def selected_to_concat(self, _ptype, sel_idx):
        return np.asarray(sel_idx, dtype=np.int64)


class DummySim:
    def __init__(self, pos, vel):
        self.data_manager = DummyDataManager(pos, vel)


def test_resolve_mode_ahf_subhalo():
    mode = modes_mod.resolve_mode({"haloid": "AHF-subhalo", "haloid_file": "x"})
    assert mode == modes_mod.Mode.AHF_SUBHALO


def test_ancestor_chain_and_task_manifest():
    parent_of = {10: 0, 11: 10, 12: 11, 20: 0, 21: 20}
    host_to_nodes = {10: {10, 11, 12}, 20: {20, 21}}
    node_npart = {10: 100, 11: 50, 12: 20, 20: 80, 21: 10}
    node_nstar = {10: 40, 11: 20, 12: 16, 20: 15, 21: 30}

    tasks, tasks_by_root = subhalo_mod._build_task_manifest(
        parent_of=parent_of,
        host_to_nodes=host_to_nodes,
        node_npart=node_npart,
        node_nstar=node_nstar,
        min_stars=16,
    )

    ids = {t.node_id for t in tasks}
    assert ids == {10, 11, 12, 21}
    chain_12 = next(t.ancestors for t in tasks if t.node_id == 12)
    assert chain_12 == (11, 10)
    depth_12 = next(t.depth for t in tasks if t.node_id == 12)
    assert depth_12 == 2
    assert set(t.node_id for t in tasks_by_root[10]) == {10, 11, 12}
    assert set(t.node_id for t in tasks_by_root[20]) == {21}


def test_task_manifest_applies_distinct_dm_thresholds_to_halos_and_subhalos():
    parent_of = {10: 0, 11: 10, 12: 10, 20: 0, 21: 20}
    host_to_nodes = {10: {10, 11, 12}, 20: {20, 21}}
    node_npart = {10: 100, 11: 40, 12: 40, 20: 100, 21: 40}
    node_nstar = {10: 20, 11: 20, 12: 20, 20: 20, 21: 20}
    node_ndm = {10: 63, 11: 23, 12: 24, 20: 64, 21: 24}

    tasks, tasks_by_root = subhalo_mod._build_task_manifest(
        parent_of=parent_of,
        host_to_nodes=host_to_nodes,
        node_npart=node_npart,
        node_nstar=node_nstar,
        node_ndm=node_ndm,
        min_stars=16,
    )

    ids = {t.node_id for t in tasks}
    assert ids == {12, 20, 21}
    assert 10 not in ids
    assert 11 not in ids
    assert set(t.node_id for t in tasks_by_root[10]) == {12}
    assert set(t.node_id for t in tasks_by_root[20]) == {20, 21}


def test_graph_6d_max_groups_close_particles():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    vel = np.zeros_like(pos)

    tags, ng = graph_mod.fof6d_on_pool_graph(
        pos,
        vel,
        fof_ll=0.1,
        vel_ll=1.0,
        mingrp=2,
        backend="numpy",
        cc_backend="cpu",
    )

    assert ng == 1
    assert np.array_equal(tags, np.array([0, 0, -1], dtype=np.int64))


def test_tiny_batch_builder_respects_ancestor_exclusion():
    t1 = subhalo_mod.AHFSubhaloTask(10, 0, 10, 0, tuple(), 20, 100)
    t2 = subhalo_mod.AHFSubhaloTask(11, 10, 10, 1, (10,), 18, 80)
    t3 = subhalo_mod.AHFSubhaloTask(20, 0, 20, 0, tuple(), 17, 70)

    batches = subhalo_mod._build_tiny_batches(
        [t1, t2, t3],
        max_nodes_per_batch=4,
        max_fof_candidates_per_batch=1000,
    )

    members = [set(task.node_id for task in batch.tasks) for batch in batches]
    assert {10, 11} not in members
    assert any(m == {10, 20} or m == {11, 20} for m in members)


def test_preferred_cc_backend_prefers_cugraph_for_gpu_auto(monkeypatch):
    import caesar.fof6d_graph as graph_pkg

    monkeypatch.setattr(graph_pkg, "_try_import_cugraph", lambda: object())
    assert subhalo_mod._preferred_cc_backend("auto", backend="cupy") == "cugraph"
    assert subhalo_mod._preferred_cc_backend("auto", backend="gpu") == "cugraph"

    monkeypatch.setattr(graph_pkg, "_try_import_cugraph", lambda: None)
    assert subhalo_mod._preferred_cc_backend("auto", backend="cupy") == "gpu"
    assert subhalo_mod._preferred_cc_backend("auto", backend="numpy") == "cpu"
    assert subhalo_mod._preferred_cc_backend("cugraph", backend="cupy") == "cugraph"


def test_batched_fof_returns_per_task_candidates_and_asserts_cross_task_components():
    sim = DummySim(
        pos=[
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.05, 0.0, 0.0],
        ],
        vel=np.zeros((4, 3), dtype=np.float64),
    )
    pid_maps = {"star": DummyLookup()}
    membership_arrays = {
        10: np.asarray([[0, 4], [1, 4]], dtype=np.int64),
        20: np.asarray([[2, 4], [3, 4]], dtype=np.int64),
    }
    task1 = subhalo_mod.AHFSubhaloTask(10, 0, 10, 0, tuple(), 2, 2)
    task2 = subhalo_mod.AHFSubhaloTask(20, 0, 20, 0, tuple(), 2, 2)
    batch = subhalo_mod.AHFSubhaloBatch(tasks=(task1, task2), is_tiny_batch=True, estimated_cost=4)

    out = subhalo_mod._fof_for_batch(
        sim,
        batch=batch,
        membership_arrays=membership_arrays,
        pid_maps_sel=pid_maps,
        min_stars=2,
        fof_ll=0.1,
        fof_vel_ll=1.0,
        fof_nHlim=0.13,
        fof_Tlim=1.0e5,
        fof_use_sfr_gate=True,
        backend="numpy",
        cc_backend="cpu",
        max_pairs_per_batch=1000,
    )
    assert len(out[10]) == 1
    assert len(out[20]) == 1
    assert np.array_equal(out[10][0].slist, np.array([0, 1], dtype=np.int32))
    assert np.array_equal(out[20][0].slist, np.array([2, 3], dtype=np.int32))

    sim_bad = DummySim(
        pos=[
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.08, 0.0, 0.0],
            [0.11, 0.0, 0.0],
        ],
        vel=np.zeros((4, 3), dtype=np.float64),
    )
    membership_bad = {
        10: np.asarray([[0, 4], [1, 4]], dtype=np.int64),
        20: np.asarray([[2, 4], [3, 4]], dtype=np.int64),
    }
    try:
        subhalo_mod._fof_for_batch(
            sim_bad,
            batch=batch,
            membership_arrays=membership_bad,
            pid_maps_sel=pid_maps,
            min_stars=2,
            fof_ll=0.2,
            fof_vel_ll=1.0,
            fof_nHlim=0.13,
            fof_Tlim=1.0e5,
            fof_use_sfr_gate=True,
            backend="numpy",
            cc_backend="cpu",
            max_pairs_per_batch=1000,
        )
    except AssertionError as exc:
        assert "multiple batched halos" in str(exc)
    else:
        raise AssertionError("expected cross-task graph invariant failure")


def test_ancestor_list_serialization_and_loader_accessor(tmp_path):
    galaxies = [
        DummyGalaxy(10, 100, [100]),
        DummyGalaxy(20, 200, [150, 200]),
    ]
    outfile = tmp_path / "ahf_ancestors.hdf5"

    with h5py.File(outfile, "w") as f:
        hd = f.create_group("galaxy_data")
        hdl = hd.create_group("lists")
        hdd = hd.create_group("dicts")
        saver_mod.serialize_list(galaxies, "AHF_ancestor_haloIDs", hdl)
        saver_mod.serialize_attributes(galaxies, hd, hdd)

    with h5py.File(outfile, "r") as f:
        assert np.array_equal(f["galaxy_data/lists/AHF_ancestor_haloIDs"][:], np.array([100, 150, 200]))
        assert np.array_equal(f["galaxy_data/AHF_haloID"][:], np.array([10, 20]))
        assert np.array_equal(f["galaxy_data/AHF_top_haloID"][:], np.array([100, 200]))
        assert np.array_equal(f["galaxy_data/AHF_ancestor_haloIDs_start"][:], np.array([0, 1]))
        assert np.array_equal(f["galaxy_data/AHF_ancestor_haloIDs_end"][:], np.array([1, 3]))

    class DummyObj:
        pass

    obj = DummyObj()
    obj._galaxy_AHF_ancestor_haloIDs = np.array([100, 150, 200], dtype=np.int64)
    gal = loader_mod.Galaxy.__new__(loader_mod.Galaxy)
    gal.obj = obj
    gal.AHF_ancestor_haloIDs_start = 1
    gal.AHF_ancestor_haloIDs_end = 3
    assert np.array_equal(gal.AHF_ancestor_haloIDs, np.array([150, 200], dtype=np.int64))


def test_candidate_serialization_roundtrip():
    sim = DummySim(pos=np.zeros((4, 3)), vel=np.zeros((4, 3)))
    task = subhalo_mod.AHFSubhaloTask(10, 1, 100, 2, (1, 100), 4, 10)
    gal = subhalo_mod._build_candidate_group(
        sim,
        task=task,
        star_sel=np.array([1, 3], dtype=np.int32),
        gas_sel=np.array([0], dtype=np.int32),
        bh_sel=np.array([], dtype=np.int32),
        dm_sel=np.array([2], dtype=np.int32),
    )
    payload = subhalo_mod._serialize_candidate_group(gal)
    gal2 = subhalo_mod._deserialize_candidate_group(sim, payload)
    assert int(gal2.AHF_haloID) == 10
    assert int(gal2.AHF_parent_haloID) == 1
    assert int(gal2.AHF_top_haloID) == 100
    assert int(gal2.AHF_depth) == 2
    assert np.array_equal(gal2.AHF_ancestor_haloIDs, np.array([1, 100], dtype=np.int64))
    assert np.array_equal(gal2.slist, np.array([1, 3], dtype=np.int32))


def test_task_input_payload_and_stateless_task_fof():
    sim = DummySim(
        pos=[
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
        ],
        vel=np.zeros((2, 3), dtype=np.float64),
    )
    pid_maps = {"star": DummyLookup()}
    membership_arrays = {10: np.asarray([[0, 4], [1, 4]], dtype=np.int64)}
    task = subhalo_mod.AHFSubhaloTask(10, 0, 10, 0, tuple(), 2, 2)

    task_payload = subhalo_mod._build_task_input_payload(
        sim,
        task=task,
        membership_arrays=membership_arrays,
        pid_maps_sel=pid_maps,
        fof_nHlim=0.13,
        fof_Tlim=1.0e5,
        fof_use_sfr_gate=True,
    )
    assert task_payload is not None
    out = subhalo_mod._fof_on_task_payload(
        task_payload=task_payload,
        min_stars=2,
        fof_ll=0.1,
        fof_vel_ll=1.0,
        backend="numpy",
        cc_backend="cpu",
        max_pairs_per_batch=1000,
    )
    assert len(out) == 1
    assert int(out[0]["AHF_haloID"]) == 10
    assert np.array_equal(out[0]["slist"], np.array([0, 1], dtype=np.int32))


def test_stateless_batch_fof_and_reconcile_root_payload():
    task_parent = subhalo_mod.AHFSubhaloTask(10, 0, 10, 0, tuple(), 4, 4)
    task_child = subhalo_mod.AHFSubhaloTask(11, 10, 10, 1, (10,), 2, 2)

    batch_payload = {
        "task_payloads": [
            {
                "task": subhalo_mod._serialize_task(task_parent),
                "gas_sel": np.array([], dtype=np.int32),
                "star_sel": np.array([0, 1], dtype=np.int32),
                "bh_sel": np.array([], dtype=np.int32),
                "dm_sel": np.array([], dtype=np.int32),
                "ng": 0,
                "ns": 2,
                "nb": 0,
                "eligible_pos": np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float64),
                "eligible_vel": np.zeros((2, 3), dtype=np.float64),
            },
            {
                "task": subhalo_mod._serialize_task(task_child),
                "gas_sel": np.array([], dtype=np.int32),
                "star_sel": np.array([10, 11], dtype=np.int32),
                "bh_sel": np.array([], dtype=np.int32),
                "dm_sel": np.array([], dtype=np.int32),
                "ng": 0,
                "ns": 2,
                "nb": 0,
                "eligible_pos": np.array([[10.0, 0.0, 0.0], [10.05, 0.0, 0.0]], dtype=np.float64),
                "eligible_vel": np.zeros((2, 3), dtype=np.float64),
            },
        ]
    }
    batch_out = subhalo_mod._fof_on_batch_payload(
        batch_payload=batch_payload,
        min_stars=2,
        fof_ll=0.1,
        fof_vel_ll=1.0,
        backend="numpy",
        cc_backend="cpu",
        max_pairs_per_batch=1000,
    )
    assert len(batch_out[10]) == 1
    assert len(batch_out[11]) == 1

    root_out = subhalo_mod._reconcile_root_payload(
        tasks=[task_parent, task_child],
        initial_candidates_by_node={10: batch_out[10], 11: batch_out[11]},
        task_payloads_by_node={
            10: {
                "task": subhalo_mod._serialize_task(task_parent),
                "gas_sel": np.array([], dtype=np.int32),
                "star_sel": np.array([10, 11], dtype=np.int32),
                "bh_sel": np.array([], dtype=np.int32),
                "dm_sel": np.array([], dtype=np.int32),
                "ng": 0,
                "ns": 2,
                "nb": 0,
                "eligible_pos": np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float64),
                "eligible_vel": np.zeros((2, 3), dtype=np.float64),
            },
            11: batch_payload["task_payloads"][1],
        },
        min_stars=2,
        fof_ll=0.1,
        fof_vel_ll=1.0,
        backend="numpy",
        cc_backend="cpu",
        max_pairs_per_batch=1000,
    )
    assert len(root_out) == 1
    assert int(root_out[0]["AHF_haloID"]) == 11
    assert np.array_equal(root_out[0]["slist"], np.array([10, 11], dtype=np.int32))


def test_stage1_batch_classification_prefers_gpu_for_large_regular_tasks():
    t_small = subhalo_mod.AHFSubhaloTask(10, 0, 10, 0, tuple(), 64, 1000)
    t_large = subhalo_mod.AHFSubhaloTask(20, 0, 20, 0, tuple(), 64, 100000)
    gpu_items, cpu_items = mpi_mod._classify_stage1_batches(
        tasks=[t_small, t_large],
        gpu_worker_count=1,
        cpu_worker_count=1,
    )
    assert any(int(item.tasks[0].node_id) == 20 for item in gpu_items)
    assert any(int(item.tasks[0].node_id) == 10 for item in cpu_items)


def test_gpu_device_assignment_uses_gpu_worker_order_within_host(monkeypatch):
    monkeypatch.setattr(mpi_mod, "_available_gpu_device_ids", lambda: [0, 1, 2, 3])
    layout = [
        {"rank": 0, "role": "coordinator", "hostname": "nodeA", "local_rank": 0},
        {"rank": 1, "role": "gpu_worker", "hostname": "nodeA", "local_rank": 1},
        {"rank": 2, "role": "gpu_worker", "hostname": "nodeA", "local_rank": 2},
        {"rank": 3, "role": "cpu_worker", "hostname": "nodeA", "local_rank": 3},
        {"rank": 4, "role": "gpu_worker", "hostname": "nodeA", "local_rank": 4},
    ]
    assert mpi_mod._gpu_device_for_rank(rank=1, role="gpu_worker", world_layout=layout) == 0
    assert mpi_mod._gpu_device_for_rank(rank=2, role="gpu_worker", world_layout=layout) == 1
    assert mpi_mod._gpu_device_for_rank(rank=4, role="gpu_worker", world_layout=layout) == 2
    assert mpi_mod._gpu_device_for_rank(rank=3, role="cpu_worker", world_layout=layout) is None


def test_stage1_shard_payload_serializes_node_candidates():
    sim = DummySim(pos=np.zeros((4, 3)), vel=np.zeros((4, 3)))
    task = subhalo_mod.AHFSubhaloTask(10, 0, 10, 0, tuple(), 4, 10)
    gal = subhalo_mod._build_candidate_group(
        sim,
        task=task,
        star_sel=np.array([1, 2], dtype=np.int32),
        gas_sel=np.array([], dtype=np.int32),
        bh_sel=np.array([], dtype=np.int32),
        dm_sel=np.array([], dtype=np.int32),
    )
    payload = mpi_mod._stage1_shard_payload({10: [gal]})
    assert list(payload.keys()) == [10]
    assert int(payload[10][0]["AHF_haloID"]) == 10
    assert np.array_equal(payload[10][0]["slist"], np.array([1, 2], dtype=np.int32))


def test_iter_stage1_batch_payloads_writes_lazily(tmp_path):
    task = subhalo_mod.AHFSubhaloTask(10, 0, 10, 0, tuple(), 4, 10)
    batch = subhalo_mod.AHFSubhaloBatch(tasks=(task,), is_tiny_batch=False, target_backend="cpu", estimated_cost=10)
    payload_iter = mpi_mod._iter_stage1_batch_payloads(
        shard_root=tmp_path,
        stage_batches=[batch],
        prefix="cpu",
        task_payloads_by_node={10: {"task": subhalo_mod._serialize_task(task), "dummy": 1}},
        fof_ll=0.02,
        fof_vel_ll=1.0,
        min_stars=16,
    )
    payload_path = tmp_path / "stage1_input_cpu_000000.pkl"
    assert not payload_path.exists()

    item = next(payload_iter)

    assert payload_path.exists()
    assert item["payload_path"] == str(payload_path)
    assert item["min_stars"] == 16
    assert item["fof_ll"] == 0.02


def test_dispatch_stage_consumes_iterators_incrementally(monkeypatch):
    produced = []
    send_counts = []
    logs = []

    class DummyStatus:
        def __init__(self):
            self._source = -1

        def Get_source(self):
            return self._source

    class FakeComm:
        def __init__(self):
            self.inflight = []

        def send(self, payload, dest, tag):
            if tag == mpi_mod.MSG_WORK:
                send_counts.append(len(produced))
                self.inflight.append((int(dest), payload))

        def iprobe(self, source=None, tag=None):
            return bool(self.inflight)

        def recv(self, source=None, tag=None, status=None):
            rank, payload = self.inflight.pop(0)
            status._source = int(rank)
            return {"stage": payload["stage"], "item": payload["item"], "rank": int(rank)}

    def cpu_items():
        for idx in range(3):
            produced.append(idx)
            yield {"payload_path": f"cpu_{idx}.pkl"}

    monkeypatch.setattr(mpi_mod, "MPI", SimpleNamespace(Status=DummyStatus, ANY_SOURCE=-1))
    monkeypatch.setattr(mpi_mod, "_rank0_log", logs.append)
    monkeypatch.setattr(mpi_mod.time, "sleep", lambda _seconds: None)

    results = mpi_mod._dispatch_stage(
        FakeComm(),
        worker_caps=[mpi_mod.WorkerCapability(rank=1, role="cpu_worker", hostname="nodeA", local_rank=0, gpu_device=None, threads=1)],
        gpu_queue=[],
        cpu_queue=cpu_items(),
        stage_name="stage1",
        gpu_total=0,
        cpu_total=3,
    )

    assert len(results) == 3
    assert produced == [0, 1, 2]
    assert send_counts == [1, 2, 3]
    assert any("stage1: starting" in msg for msg in logs)
    assert any("stage1: complete" in msg for msg in logs)


def test_dispatch_stage_bundles_cpu_items_by_rank_threads(monkeypatch):
    sent_payloads = []

    class DummyStatus:
        def __init__(self):
            self._source = -1

        def Get_source(self):
            return self._source

    class FakeComm:
        def __init__(self):
            self.inflight = []

        def send(self, payload, dest, tag):
            if tag == mpi_mod.MSG_WORK:
                sent_payloads.append(payload)
                self.inflight.append((int(dest), payload))

        def iprobe(self, source=None, tag=None):
            return bool(self.inflight)

        def recv(self, source=None, tag=None, status=None):
            rank, payload = self.inflight.pop(0)
            status._source = int(rank)
            item = payload["item"]
            return {"stage": payload["stage"], "completed_count": int(item.get("completed_count", 1))}

    monkeypatch.setattr(mpi_mod, "MPI", SimpleNamespace(Status=DummyStatus, ANY_SOURCE=-1))
    monkeypatch.setattr(mpi_mod, "_rank0_log", lambda _msg: None)
    monkeypatch.setattr(mpi_mod.time, "sleep", lambda _seconds: None)

    cpu_items = [{"payload_path": f"cpu_{idx}.pkl"} for idx in range(3)]
    results = mpi_mod._dispatch_stage(
        FakeComm(),
        worker_caps=[mpi_mod.WorkerCapability(rank=1, role="cpu_worker", hostname="nodeA", local_rank=0, gpu_device=None, threads=2)],
        gpu_queue=[],
        cpu_queue=cpu_items,
        stage_name="stage1",
        gpu_total=0,
        cpu_total=3,
    )

    assert len(results) == 2
    assert len(sent_payloads) == 2
    assert len(sent_payloads[0]["item"]["items"]) == 2
    assert sent_payloads[0]["item"]["completed_count"] == 2
    assert sent_payloads[1]["item"]["payload_path"] == "cpu_2.pkl"


def test_worker_run_stage1_bundles_multiple_items(tmp_path):
    task_a = subhalo_mod.AHFSubhaloTask(10, 0, 10, 0, tuple(), 2, 2)
    task_b = subhalo_mod.AHFSubhaloTask(20, 0, 20, 0, tuple(), 2, 2)
    batch_a = subhalo_mod.AHFSubhaloBatch(tasks=(task_a,), is_tiny_batch=False, target_backend="cpu", estimated_cost=2)
    batch_b = subhalo_mod.AHFSubhaloBatch(tasks=(task_b,), is_tiny_batch=False, target_backend="cpu", estimated_cost=2)

    payload_a = {
        "batch": mpi_mod._serialize_batch(batch_a),
        "task_payloads": [
            {
                "task": subhalo_mod._serialize_task(task_a),
                "gas_sel": np.array([], dtype=np.int32),
                "star_sel": np.array([0, 1], dtype=np.int32),
                "bh_sel": np.array([], dtype=np.int32),
                "dm_sel": np.array([], dtype=np.int32),
                "ng": 0,
                "ns": 2,
                "nb": 0,
                "eligible_pos": np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float64),
                "eligible_vel": np.zeros((2, 3), dtype=np.float64),
            }
        ],
    }
    payload_b = {
        "batch": mpi_mod._serialize_batch(batch_b),
        "task_payloads": [
            {
                "task": subhalo_mod._serialize_task(task_b),
                "gas_sel": np.array([], dtype=np.int32),
                "star_sel": np.array([10, 11], dtype=np.int32),
                "bh_sel": np.array([], dtype=np.int32),
                "dm_sel": np.array([], dtype=np.int32),
                "ng": 0,
                "ns": 2,
                "nb": 0,
                "eligible_pos": np.array([[10.0, 0.0, 0.0], [10.05, 0.0, 0.0]], dtype=np.float64),
                "eligible_vel": np.zeros((2, 3), dtype=np.float64),
            }
        ],
    }

    path_a = tmp_path / "stage1_input_cpu_000000.pkl"
    path_b = tmp_path / "stage1_input_cpu_000001.pkl"
    mpi_mod._dump_pickle(path_a, payload_a)
    mpi_mod._dump_pickle(path_b, payload_b)

    result = mpi_mod._worker_run_stage1(
        item={
            "items": [
                {"payload_path": str(path_a), "backend": "cpu"},
                {"payload_path": str(path_b), "backend": "cpu"},
            ]
        },
        fof_ll=0.1,
        fof_vel_ll=1.0,
        min_stars=2,
        shard_dir=tmp_path,
        nproc=2,
    )

    assert result["completed_count"] == 2
    assert set(result["node_ids"]) == {10, 20}
    payload = mpi_mod._load_pickle(Path(result["shard_path"]))
    assert set(payload.keys()) == {10, 20}
    assert np.array_equal(payload[10][0]["slist"], np.array([0, 1], dtype=np.int32))
    assert np.array_equal(payload[20][0]["slist"], np.array([10, 11], dtype=np.int32))


def test_group_state_serialization_and_apply_roundtrip():
    sim = DummySim(pos=np.zeros((4, 3)), vel=np.zeros((4, 3)))
    task = subhalo_mod.AHFSubhaloTask(10, 0, 10, 0, tuple(), 2, 2)
    gal = subhalo_mod._build_candidate_group(
        sim,
        task=task,
        star_sel=np.array([0, 1], dtype=np.int32),
        gas_sel=np.array([], dtype=np.int32),
        bh_sel=np.array([], dtype=np.int32),
        dm_sel=np.array([], dtype=np.int32),
    )
    gal._merge_id = 7
    gal.masses["stellar"] = 123.0
    gal.pos = np.array([1.0, 2.0, 3.0])

    state = subhalo_mod._serialize_group_state(gal, id_key="_merge_id")
    gal2 = subhalo_mod._build_candidate_group(
        sim,
        task=task,
        star_sel=np.array([], dtype=np.int32),
        gas_sel=np.array([], dtype=np.int32),
        bh_sel=np.array([], dtype=np.int32),
        dm_sel=np.array([], dtype=np.int32),
    )
    subhalo_mod._apply_group_state(gal2, state)

    assert int(state["id"]) == 7
    assert gal2._merge_id == 7
    assert gal2.masses["stellar"] == 123.0
    assert np.array_equal(gal2.pos, np.array([1.0, 2.0, 3.0]))


def test_stage3_payload_helpers():
    sim = DummySim(pos=np.zeros((4, 3)), vel=np.zeros((4, 3)))
    task = subhalo_mod.AHFSubhaloTask(10, 0, 10, 0, tuple(), 2, 2)
    gal = subhalo_mod._build_candidate_group(
        sim,
        task=task,
        star_sel=np.array([0, 1], dtype=np.int32),
        gas_sel=np.array([], dtype=np.int32),
        bh_sel=np.array([], dtype=np.int32),
        dm_sel=np.array([], dtype=np.int32),
    )
    gal._merge_id = 3

    class DummyHalo:
        def __init__(self, ahf_halo_id):
            self.AHF_haloID = ahf_halo_id

    halo_payload = mpi_mod._stage3_halo_work_payload([DummyHalo(101), DummyHalo(202)])
    gal_payload = mpi_mod._stage3_galaxy_work_payload([gal])

    assert halo_payload == [101, 202]
    assert len(gal_payload) == 1
    assert int(gal_payload[0]["_merge_id"]) == 3


def test_stage3_manifest_roundtrip(tmp_path):
    path = mpi_mod._write_stage3_manifest(
        shard_root=tmp_path,
        snapshot_file="snap.hdf5",
        ahf_particles_file="ahf_particles",
        output_file="out.hdf5",
        final_shards=["a.pkl", "b.pkl"],
    )
    assert path.is_file()
    payload = mpi_mod._load_stage3_manifest(tmp_path)
    assert payload["snapshot_file"] == "snap.hdf5"
    assert payload["ahf_particles_file"] == "ahf_particles"
    assert payload["output_file"] == "out.hdf5"
    assert payload["final_shards"] == ["a.pkl", "b.pkl"]


def test_stage3_property_payload_roundtrip():
    from caesar.group import create_new_group

    sim = SimpleNamespace()
    sim.units = {
        "mass": "Msun",
        "length": "kpccm",
        "velocity": "km/s",
        "time": "yr",
        "temperature": "K",
    }
    sim._kwargs = {}
    sim.load_pot = True
    sim.simulation = SimpleNamespace(
        XH=0.76,
        redshift=0.0,
        omega_baryon=0.05,
        omega_matter=0.3,
        boxsize=SimpleNamespace(d=100.0),
    )
    sim.data_manager = SimpleNamespace(
        ptypes=["gas", "star", "dm"],
        blackholes=False,
        pos=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        vel=np.zeros((3, 3), dtype=np.float32),
        mass=np.asarray([10.0, 5.0, 20.0], dtype=np.float32),
        pot=np.zeros(3, dtype=np.float32),
        ptype=np.asarray([0, 4, 1], dtype=np.int32),
        glist=np.asarray([0], dtype=np.int64),
        slist=np.asarray([1], dtype=np.int64),
        dmlist=np.asarray([2], dtype=np.int64),
        gnh=np.asarray([1.0], dtype=np.float32),
        gsfr=np.asarray([0.0], dtype=np.float32),
        gZ=np.asarray([0.01], dtype=np.float32),
        gT=np.asarray([1.0e4], dtype=np.float32),
        gfH2=np.asarray([0.2], dtype=np.float32),
        gfHI=np.asarray([0.7], dtype=np.float32),
        dustmass=np.asarray([0.0], dtype=np.float32),
        sZ=np.asarray([0.02], dtype=np.float32),
        age=np.asarray([1.0], dtype=np.float32),
    )

    halo = create_new_group(sim, "halo")
    halo.AHF_haloID = 100
    halo.global_indexes = np.asarray([0, 1, 2], dtype=np.int64)
    halo.glist = np.asarray([0], dtype=np.int64)
    halo.slist = np.asarray([0], dtype=np.int64)
    halo.dmlist = np.asarray([0], dtype=np.int64)
    halo.bhlist = np.asarray([], dtype=np.int64)
    halo.galaxy_index_list = np.asarray([0], dtype=np.int32)

    gal = create_new_group(sim, "galaxy")
    gal.AHF_haloID = 101
    gal.AHF_parent_haloID = 100
    gal.AHF_top_haloID = 100
    gal.AHF_depth = 1
    gal.AHF_ancestor_haloIDs = np.asarray([100], dtype=np.int64)
    gal.global_indexes = np.asarray([0, 1], dtype=np.int64)
    gal.glist = np.asarray([0], dtype=np.int64)
    gal.slist = np.asarray([0], dtype=np.int64)
    gal.dmlist = np.asarray([], dtype=np.int64)
    gal.bhlist = np.asarray([], dtype=np.int64)
    gal.parent_halo_index = 0
    gal._ahf_host_halo_index = 0
    gal._merge_id = 7

    sim.halo_list = [halo]
    sim.galaxy_list = [gal]

    payload = subhalo_mod._build_stage3_property_payload(sim, [halo])
    local_sim = subhalo_mod._build_stage3_property_runtime(payload, nproc=2)

    assert local_sim.nproc == 2
    assert len(local_sim.halo_list) == 1
    assert len(local_sim.galaxy_list) == 1
    assert np.array_equal(local_sim.data_manager.glist, np.asarray([0], dtype=np.int64))
    assert np.array_equal(local_sim.data_manager.slist, np.asarray([1], dtype=np.int64))
    assert local_sim.galaxy_list[0].parent_halo_index == 0
    assert local_sim._ds_type.has_property("gas", "fh2")
