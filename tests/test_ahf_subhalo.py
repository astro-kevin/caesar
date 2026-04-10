from pathlib import Path
import importlib.util
import sys

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
