from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "caesar"
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
sys.modules.pop("caesar", None)


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
group_loader_mod = _load_module("group_funcs_loader_mod", "group_funcs_loader.py")


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


def test_run_uses_direct_runtime_entrypoint(monkeypatch, tmp_path):
    calls = {}

    class FakeDS:
        directory = str(tmp_path)
        basename = "snap.hdf5"

    class FakeObj:
        def __init__(self):
            self._args = ()
            self._kwargs = {"haloid_file": "dummy.AHF_particles", "nproc": 2}
            self._ds = FakeDS()
            self.nproc = 1

    fake_obj = FakeObj()

    import caesar.group as group_pkg

    monkeypatch.setattr(group_pkg, "get_min_stars", lambda _obj: 16)

    def fake_direct(snapshot_file, ahf_particles_file, *, kwargs, nproc, min_stars):
        calls["snapshot_file"] = snapshot_file
        calls["ahf_particles_file"] = ahf_particles_file
        calls["kwargs"] = dict(kwargs)
        calls["nproc"] = int(nproc)
        calls["min_stars"] = int(min_stars)
        return SimpleNamespace(marker="direct-runtime")

    def fake_adopt(obj, runtime):
        calls["adopt_obj"] = obj
        calls["adopt_runtime"] = runtime

    monkeypatch.setattr(subhalo_mod, "_run_ahf_subhalo_direct", fake_direct)
    monkeypatch.setattr(subhalo_mod, "_adopt_caesar_runtime", fake_adopt)

    subhalo_mod.run(fake_obj)

    assert calls["snapshot_file"] == str(tmp_path / "snap.hdf5")
    assert calls["ahf_particles_file"] == "dummy.AHF_particles"
    assert calls["kwargs"]["haloid_file"] == "dummy.AHF_particles"
    assert calls["nproc"] == 2
    assert calls["min_stars"] == 16
    assert calls["adopt_obj"] is fake_obj
    assert getattr(calls["adopt_runtime"], "marker", "") == "direct-runtime"


def test_rank0_prepare_galaxy_finding_sets_velocity_linking_length(monkeypatch, tmp_path):
    class DummyNodes:
        def __init__(self):
            self.halo_id = np.asarray([10], dtype=np.int64)
            self.dm_count = np.asarray([64], dtype=np.int64)

        def __len__(self):
            return 1

    direct_state = SimpleNamespace(
        snapshot=SimpleNamespace(ptypes=("gas", "dm", "star"), boxsize=100.0, particle_counts={"dm": 1000}),
        nodes=DummyNodes(),
    )
    task = subhalo_mod.AHFSubhaloTask(10, 0, 10, 0, tuple(), 16, 32)

    monkeypatch.setattr(mpi_mod, "build_direct_state", lambda *_args, **_kwargs: direct_state)
    monkeypatch.setattr(mpi_mod, "_prepare_manifest", lambda **_kwargs: ([task], {10: [task]}, {}, {}))
    monkeypatch.setattr(
        mpi_mod,
        "_build_direct_task_payload",
        lambda *_args, **_kwargs: {"star_sel": np.arange(16, dtype=np.int32)},
    )
    monkeypatch.setattr(
        mpi_mod,
        "_build_store_artifacts",
        lambda **_kwargs: {"roots": {10: {"store_id": "10", "kind": "root", "particle_store_path": "particle_store_root10.h5", "node_store_path": "node_store_root10.pkl"}}},
    )
    monkeypatch.setattr(mpi_mod, "get_min_stars", lambda *_args, **_kwargs: 16)
    monkeypatch.delenv("CAESAR_FOF6D_VEL_LL", raising=False)
    monkeypatch.delenv("CAESAR_FOF6D_DISABLE_VEL", raising=False)

    state, pid_maps_sel, min_stars, fof_ll, fof_vel_ll, tasks, tasks_by_root, payloads, store_manifest = mpi_mod._rank0_prepare_galaxy_finding(
        snapshot_file="snap.hdf5",
        ahf_particles_file="ahf.AHF_particles",
        shard_root=tmp_path,
        nproc=1,
        min_stars=None,
        log_fn=None,
        log_label="finding galaxies",
    )

    assert state is direct_state
    assert pid_maps_sel is None
    assert min_stars == 16
    assert fof_ll > 0.0
    assert fof_vel_ll == 1.0
    assert tasks == [task]
    assert set(tasks_by_root.keys()) == {10}
    assert 10 in payloads
    assert payloads[10]["task"]["node_id"] == 10
    assert store_manifest["roots"][10]["store_id"] == "10"


def test_reconciled_galaxy_shard_payload_uses_table_format():
    records = [
        {
            "AHF_haloID": 10,
            "AHF_parent_haloID": 0,
            "AHF_top_haloID": 10,
            "AHF_depth": 0,
            "AHF_ancestor_haloIDs": np.asarray([], dtype=np.int64),
            "glist": np.asarray([1], dtype=np.int32),
            "slist": np.asarray([2, 3], dtype=np.int32),
            "dmlist": np.asarray([], dtype=np.int32),
            "bhlist": np.asarray([], dtype=np.int32),
            "dlist": np.asarray([], dtype=np.int32),
        }
    ]
    payload = mpi_mod._reconciled_galaxy_shard_payload(records)
    assert isinstance(payload, dict)
    assert "ahf_halo_id" in payload
    restored = mpi_mod.candidate_records_from_table_payload(payload)
    assert len(restored) == 1
    assert restored[0]["AHF_haloID"] == 10
    assert np.array_equal(restored[0]["slist"], np.asarray([2, 3], dtype=np.int32))


def test_calculating_properties_payload_preserves_global_particle_lists():
    class DummyGroup:
        pass

    class DummyDataManager:
        def __init__(self):
            self.pos = np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
            self.vel = np.zeros((2, 3), dtype=np.float32)
            self.mass = np.asarray([3.0, 4.0], dtype=np.float32)
            self.ptype = np.asarray([0, 4], dtype=np.int32)
            self.pot = np.asarray([-1.0, -2.0], dtype=np.float32)
            self.glist = np.asarray([0], dtype=np.int64)
            self.slist = np.asarray([1], dtype=np.int64)
            self.blackholes = False
            self.ptypes = ["gas", "star"]

    halo = DummyGroup()
    halo.AHF_haloID = 10
    halo.AHF_parent_haloID = 0
    halo.AHF_top_haloID = 10
    halo.AHF_depth = 0
    halo.AHF_ancestor_haloIDs = np.asarray([], dtype=np.int64)
    halo.global_indexes = np.asarray([0, 1], dtype=np.int64)
    halo.glist = np.asarray([0], dtype=np.int64)
    halo.slist = np.asarray([0], dtype=np.int64)
    halo.dmlist = np.asarray([], dtype=np.int64)
    halo.bhlist = np.asarray([], dtype=np.int64)
    halo.dlist = np.asarray([], dtype=np.int64)
    halo.galaxy_index_list = np.asarray([0], dtype=np.int32)
    halo._merge_id = 0

    gal = DummyGroup()
    gal.AHF_haloID = 10
    gal.AHF_parent_haloID = 0
    gal.AHF_top_haloID = 10
    gal.AHF_depth = 0
    gal.AHF_ancestor_haloIDs = np.asarray([], dtype=np.int64)
    gal.global_indexes = np.asarray([1], dtype=np.int64)
    gal.glist = np.asarray([], dtype=np.int64)
    gal.slist = np.asarray([0], dtype=np.int64)
    gal.dmlist = np.asarray([], dtype=np.int64)
    gal.bhlist = np.asarray([], dtype=np.int64)
    gal.dlist = np.asarray([], dtype=np.int64)
    gal.parent_halo_index = 0
    gal._ahf_host_halo_index = 0
    gal._merge_id = 0

    sim = SimpleNamespace(
        halo_list=[halo],
        galaxy_list=[gal],
        data_manager=DummyDataManager(),
        units={"mass": "Msun", "length": "kpccm", "velocity": "km/s", "time": "yr", "temperature": "K"},
        simulation=SimpleNamespace(
            XH=0.76,
            redshift=0.0,
            omega_baryon=0.0,
            omega_matter=0.3,
            omega_lambda=0.7,
            Om_z=0.3,
            hubble_constant=0.7,
            boxsize=SimpleNamespace(d=100.0),
            critical_density=SimpleNamespace(d=1.0),
            G=SimpleNamespace(d=1.0),
            H_z=SimpleNamespace(d=1.0),
            Densities=SimpleNamespace(d=np.asarray([1.0, 2.0, 3.0], dtype=np.float64)),
            ngas=1,
            nstar=1,
            nbh=0,
            ndust=0,
            ndm=0,
            ndm2=0,
            ndm3=0,
            ntot=2,
            baryons_present=True,
            unbind_halos=False,
            effective_resolution=1,
            mean_interparticle_separation=SimpleNamespace(d=1.0),
        ),
        _kwargs={},
        load_pot=True,
    )

    payload = subhalo_mod._build_calculating_properties_payload(sim, [halo])
    halo_rec = payload["halos"][0]
    gal_rec = payload["galaxies"][0]
    assert np.array_equal(halo_rec["global_glist"], np.asarray([0], dtype=np.int64))
    assert np.array_equal(halo_rec["global_slist"], np.asarray([0], dtype=np.int64))
    assert np.array_equal(gal_rec["global_slist"], np.asarray([0], dtype=np.int64))


def test_calculating_properties_list_blocks_prefer_preserved_global_lists():
    states = [{"id": 7, "glist": np.asarray([0, 1], dtype=np.int64), "_glist": np.asarray([10, 11], dtype=np.int64)}]
    blocks = mpi_mod._build_property_list_blocks(states, ("glist",))
    assert np.array_equal(blocks["glist"]["data"], np.asarray([10, 11], dtype=np.int64))


def test_rank0_log_includes_total_and_stage_elapsed(monkeypatch, capsys):
    ticks = iter([100.0, 105.0, 112.0])
    monkeypatch.setattr(mpi_mod.time, "monotonic", lambda: next(ticks))
    mpi_mod._reset_rank0_log_context(now=100.0)

    mpi_mod._rank0_log("finding galaxies: start")
    mpi_mod._rank0_log("finding galaxies: progress")
    mpi_mod._rank0_log("reconciling subhalos: start")

    lines = [line for line in capsys.readouterr().out.strip().splitlines() if line]
    assert "[total=0.0s][finding galaxies=0.0s] finding galaxies: start" in lines[0]
    assert "[total=5.0s][finding galaxies=5.0s] finding galaxies: progress" in lines[1]
    assert "[total=12.0s][reconciling subhalos=0.0s] reconciling subhalos: start" in lines[2]


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


def test_galaxy_finding_batch_classification_builds_front_back_regular_queue_and_small_batches():
    t_tiny = subhalo_mod.AHFSubhaloTask(5, 0, 5, 0, tuple(), 8, 100)
    t_small = subhalo_mod.AHFSubhaloTask(10, 0, 10, 0, tuple(), 64, 1000)
    t_large = subhalo_mod.AHFSubhaloTask(20, 0, 20, 0, tuple(), 64, 100000)
    regular_items, small_items = mpi_mod._classify_galaxy_finding_batches(
        tasks=[t_tiny, t_small, t_large],
        gpu_worker_count=1,
        cpu_worker_count=1,
    )
    assert [int(item.tasks[0].node_id) for item in regular_items] == [20, 10]
    assert len(small_items) == 1
    assert int(small_items[0].tasks[0].node_id) == 5


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


def test_galaxy_finding_shard_payload_serializes_node_candidates():
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
    payload = mpi_mod._galaxy_finding_shard_payload({10: [gal]})
    assert list(payload.keys()) == [10]
    assert int(payload[10][0]["AHF_haloID"]) == 10
    assert np.array_equal(payload[10][0]["slist"], np.array([1, 2], dtype=np.int32))


def test_materialize_galaxy_finding_batch_item_writes_payload_on_assignment(tmp_path):
    task = subhalo_mod.AHFSubhaloTask(10, 0, 10, 0, tuple(), 4, 10)
    batch = subhalo_mod.AHFSubhaloBatch(tasks=(task,), is_tiny_batch=False, target_backend="cpu", estimated_cost=10)
    payload_path = tmp_path / "finding_galaxies_payload_regular_000000.pkl"
    assert not payload_path.exists()

    item = mpi_mod._materialize_galaxy_finding_batch_item(
        shard_root=tmp_path,
        batch=batch,
        prefix="regular",
        batch_index=0,
        task_payloads_by_node={10: {"task": subhalo_mod._serialize_task(task), "dummy": 1}},
        fof_ll=0.02,
        fof_vel_ll=1.0,
        min_stars=16,
        backend="cpu",
        device_id=None,
        store_ids=("10",),
    )

    assert payload_path.exists()
    assert item["payload_path"] == str(payload_path)
    assert item["min_stars"] == 16
    assert item["fof_ll"] == 0.02


def test_dispatch_stage_assigns_regular_front_and_back_with_small_queue(monkeypatch):
    sent_payloads = []
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
                sent_payloads.append(payload)
                self.inflight.append((int(dest), payload))

        def iprobe(self, source=None, tag=None):
            return bool(self.inflight)

        def recv(self, source=None, tag=None, status=None):
            rank, payload = self.inflight.pop(0)
            status._source = int(rank)
            return {"stage": payload["stage"], "item": payload["item"], "rank": int(rank)}

    monkeypatch.setattr(mpi_mod, "MPI", SimpleNamespace(Status=DummyStatus, ANY_SOURCE=-1))
    monkeypatch.setattr(mpi_mod, "_rank0_log", logs.append)
    monkeypatch.setattr(mpi_mod.time, "sleep", lambda _seconds: None)

    def prepare_regular(raw, *, cap, direction):
        backend = "gpu" if str(cap.role) == "gpu_worker" else "cpu"
        return {"payload_path": f"{backend}_{raw}.pkl", "backend": backend, "direction": direction}

    def prepare_small(raw, *, cap, direction):
        return {"payload_path": f"small_{raw}.pkl", "backend": "cpu", "direction": direction}

    results = mpi_mod._dispatch_stage(
        FakeComm(),
        worker_caps=[
            mpi_mod.WorkerCapability(rank=1, role="gpu_worker", hostname="nodeA", local_rank=0, gpu_device=0, threads=1),
            mpi_mod.WorkerCapability(rank=2, role="cpu_worker", hostname="nodeA", local_rank=1, gpu_device=None, threads=2),
        ],
        regular_queue=["r0", "r1", "r2"],
        small_queue=["s0"],
        stage_name="finding_galaxies",
        display_name="finding galaxies",
        prepare_regular=prepare_regular,
        prepare_small=prepare_small,
        regular_total=3,
        small_total=1,
    )

    assert len(results) == 3
    assert len(sent_payloads) == 3
    first = sent_payloads[0]["item"]
    second = sent_payloads[1]["item"]
    third = sent_payloads[2]["item"]
    assert first["payload_path"] == "gpu_r0.pkl"
    assert second["items"][0]["payload_path"] == "small_s0.pkl"
    assert second["items"][1]["payload_path"] == "cpu_r2.pkl"
    assert third["payload_path"] == "gpu_r1.pkl"
    assert any("finding galaxies: starting" in msg for msg in logs)
    assert any("finding galaxies: complete" in msg for msg in logs)


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
        regular_queue=[],
        small_queue=cpu_items,
        stage_name="finding_galaxies",
        display_name="finding galaxies",
        regular_total=0,
        small_total=3,
    )

    assert len(results) == 2
    assert len(sent_payloads) == 2
    assert len(sent_payloads[0]["item"]["items"]) == 2
    assert sent_payloads[0]["item"]["completed_count"] == 2
    assert sent_payloads[1]["item"]["payload_path"] == "cpu_2.pkl"


def test_dispatch_stage_honors_thread_and_role_overrides(monkeypatch):
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

    cpu_items = [{"payload_path": f"cpu_{idx}.pkl"} for idx in range(4)]
    mpi_mod._dispatch_stage(
        FakeComm(),
        worker_caps=[mpi_mod.WorkerCapability(rank=1, role="gpu_worker", hostname="nodeA", local_rank=0, gpu_device=0, threads=2)],
        regular_queue=[],
        small_queue=cpu_items,
        stage_name="reconciling_subhalos",
        display_name="reconciling subhalos",
        regular_total=0,
        small_total=4,
        worker_threads={1: 3},
        worker_roles={1: "cpu_worker"},
    )

    assert len(sent_payloads) == 2
    assert len(sent_payloads[0]["item"]["items"]) == 3
    assert sent_payloads[0]["item"]["nproc"] == 3
    assert sent_payloads[1]["item"]["payload_path"] == "cpu_3.pkl"


def test_uniform_stage_thread_map_uses_all_worker_cores():
    worker_caps = [
        mpi_mod.WorkerCapability(rank=1, role="gpu_worker", hostname="nodeA", local_rank=1, gpu_device=0, threads=2),
        mpi_mod.WorkerCapability(rank=2, role="gpu_worker", hostname="nodeA", local_rank=2, gpu_device=1, threads=2),
        mpi_mod.WorkerCapability(rank=3, role="gpu_worker", hostname="nodeA", local_rank=3, gpu_device=2, threads=2),
        mpi_mod.WorkerCapability(rank=4, role="gpu_worker", hostname="nodeA", local_rank=4, gpu_device=3, threads=2),
        mpi_mod.WorkerCapability(rank=5, role="cpu_worker", hostname="nodeA", local_rank=5, gpu_device=None, threads=4),
        mpi_mod.WorkerCapability(rank=6, role="cpu_worker", hostname="nodeA", local_rank=6, gpu_device=None, threads=4),
        mpi_mod.WorkerCapability(rank=7, role="cpu_worker", hostname="nodeA", local_rank=7, gpu_device=None, threads=4),
        mpi_mod.WorkerCapability(rank=8, role="cpu_worker", hostname="nodeA", local_rank=8, gpu_device=None, threads=4),
        mpi_mod.WorkerCapability(rank=9, role="cpu_worker", hostname="nodeA", local_rank=9, gpu_device=None, threads=4),
    ]
    world_layout = [{"rank": 0, "hostname": "nodeA", "visible_cores": 32}]
    world_layout.extend(
        {"rank": int(cap.rank), "hostname": "nodeA", "visible_cores": 32} for cap in worker_caps
    )

    thread_map = mpi_mod._build_uniform_stage_thread_map(
        worker_caps=worker_caps,
        world_layout=world_layout,
        coordinator_rank=0,
    )

    vals = [int(thread_map[int(cap.rank)]) for cap in worker_caps]
    assert sum(vals) == 31
    assert min(vals) == 3
    assert max(vals) == 4


def test_calculating_properties_batches_keep_top_level_hosts_atomic():
    root_results = [
        {"root_id": 10, "shard_path": "root10.pkl", "root_payload_path": "root10_payload.pkl", "store_id": "10", "particle_store_path": "particle_store_root10.h5", "node_store_path": "node_store_root10.pkl", "count": 2, "halo_count": 2, "property_cost": 132},
        {"root_id": 20, "shard_path": "root20.pkl", "root_payload_path": "root20_payload.pkl", "store_id": "20", "particle_store_path": "particle_store_root20.h5", "node_store_path": "node_store_root20.pkl", "count": 1, "halo_count": 1, "property_cost": 19},
    ]

    batches = mpi_mod._build_calculating_properties_batches(root_results, worker_count=1)

    assert len(batches) == 2
    batch_root_ids = [{int(rec["root_id"]) for rec in batch["roots"]} for batch in batches]
    assert {10} in batch_root_ids
    assert {20} in batch_root_ids


def test_calculating_properties_batches_balance_whole_hosts_by_internal_cost(monkeypatch):
    root_results = [
        {"root_id": 10, "shard_path": "root10.pkl", "root_payload_path": "root10_payload.pkl", "store_id": "10", "particle_store_path": "particle_store_root10.h5", "node_store_path": "node_store_root10.pkl", "count": 2, "halo_count": 2, "property_cost": 229},
        {"root_id": 20, "shard_path": "root20.pkl", "root_payload_path": "root20_payload.pkl", "store_id": "bucket000001", "particle_store_path": "particle_store_bucket000001.h5", "node_store_path": "node_store_bucket000001.pkl", "count": 1, "halo_count": 1, "property_cost": 61},
        {"root_id": 30, "shard_path": "root30.pkl", "root_payload_path": "root30_payload.pkl", "store_id": "bucket000002", "particle_store_path": "particle_store_bucket000002.h5", "node_store_path": "node_store_bucket000002.pkl", "count": 0, "halo_count": 1, "property_cost": 58},
        {"root_id": 40, "shard_path": "root40.pkl", "root_payload_path": "root40_payload.pkl", "store_id": "bucket000001", "particle_store_path": "particle_store_bucket000001.h5", "node_store_path": "node_store_bucket000001.pkl", "count": 1, "halo_count": 1, "property_cost": 58},
    ]

    batches = mpi_mod._build_calculating_properties_batches(root_results, worker_count=2)

    assert len(batches) == 2
    batch_root_ids = [{int(rec["root_id"]) for rec in batch["roots"]} for batch in batches]
    assert {10} in batch_root_ids
    assert any(root_ids == {20, 40} for root_ids in batch_root_ids)


def test_rank0_calculating_properties_dispatches_root_batches_using_particle_store(monkeypatch, tmp_path):
    root_results = [
        {
            "root_id": 10,
            "shard_path": str(tmp_path / "root10.pkl"),
            "root_payload_path": str(tmp_path / "root10_payload.pkl"),
            "store_id": "bucket000001",
            "particle_store_path": str(tmp_path / "particle_store_bucket000001.h5"),
            "node_store_path": str(tmp_path / "node_store_bucket000001.pkl"),
            "count": 2,
            "halo_count": 2,
            "property_cost": 100,
        },
        {
            "root_id": 20,
            "shard_path": str(tmp_path / "root20.pkl"),
            "root_payload_path": str(tmp_path / "root20_payload.pkl"),
            "store_id": "bucket000001",
            "particle_store_path": str(tmp_path / "particle_store_bucket000001.h5"),
            "node_store_path": str(tmp_path / "node_store_bucket000001.pkl"),
            "count": 1,
            "halo_count": 1,
            "property_cost": 50,
        },
    ]
    snapshot_meta_store_path = tmp_path / "snapshot_meta_store.pkl"
    snapshot_meta_store_path.write_bytes(b"ok")
    monkeypatch.setattr(
        mpi_mod,
        "_build_calculating_properties_batches",
        lambda root_results, worker_count: [
            {
                "store_id": "bucket000001",
                "particle_store_path": str(tmp_path / "particle_store_bucket000001.h5"),
                "node_store_path": str(tmp_path / "node_store_bucket000001.pkl"),
                "roots": [dict(rec) for rec in root_results],
            }
        ],
    )
    monkeypatch.setattr(mpi_mod, "_snapshot_meta_store_path", lambda shard_root: snapshot_meta_store_path)
    monkeypatch.setattr(mpi_mod, "load_snapshot_meta_store", lambda path: "snapshot-meta")
    monkeypatch.setattr(
        mpi_mod,
        "build_direct_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("snapshot reload should not happen")),
    )
    dispatched = {}

    def _fake_dispatch(*args, **kwargs):
        dispatched["small_queue"] = [dict(item) for item in kwargs["small_queue"]]
        return []

    monkeypatch.setattr(mpi_mod, "_dispatch_stage", _fake_dispatch)

    worker_caps = [
        mpi_mod.WorkerCapability(rank=1, role="property_worker", hostname="nodeA", local_rank=0, gpu_device=None, threads=4),
    ]

    snapshot_meta, results = mpi_mod._rank0_calculate_properties(
        None,
        worker_caps=worker_caps,
        snapshot_file="snap.hdf5",
        ahf_particles_file="ahf_particles",
        output_file="out.hdf5",
        root_results=root_results,
        nproc=3,
        shard_root=tmp_path,
        sim=None,
        worker_threads=None,
    )

    assert snapshot_meta == "snapshot-meta"
    assert results == []
    assert dispatched["small_queue"] == [
        {
            "store_id": "bucket000001",
            "roots": [
                {
                    **dict(root_results[0]),
                    "halo_merge_offset": 0,
                    "galaxy_merge_offset": 0,
                },
                {
                    **dict(root_results[1]),
                    "halo_merge_offset": 2,
                    "galaxy_merge_offset": 2,
                },
            ],
            "particle_store_path": str(tmp_path / "particle_store_bucket000001.h5"),
            "node_store_path": str(tmp_path / "node_store_bucket000001.pkl"),
        }
    ]


def test_worker_run_calculating_properties_reads_root_shards_and_particle_store(monkeypatch, tmp_path):
    item = {
        "roots": [
            {
                "root_id": 10,
                "root_payload_path": str(tmp_path / "root10_payload.pkl"),
                "shard_path": str(tmp_path / "root10.pkl"),
                "halo_merge_offset": 0,
                "galaxy_merge_offset": 0,
            }
        ],
        "store_id": "10",
        "particle_store_path": str(tmp_path / "particle_store_root10.h5"),
        "node_store_path": str(tmp_path / "node_store_root10.pkl"),
    }

    payloads = {
        "root10_payload.pkl": {"halo_node_ids": [10]},
        "root10.pkl": [{"AHF_haloID": 101, "AHF_top_haloID": 10, "slist": np.asarray([2], dtype=np.int64)}],
    }
    monkeypatch.setattr(mpi_mod, "_load_pickle", lambda path: payloads[Path(path).name])
    fake_particle_store = SimpleNamespace(meta="snapshot-meta")
    monkeypatch.setattr(mpi_mod, "load_particle_store", lambda path: fake_particle_store)
    monkeypatch.setattr(
        mpi_mod,
        "load_node_store",
        lambda path: {"nodes": object(), "root_ids": (10,), "particle_rows_by_ptype": {}},
    )
    monkeypatch.setattr(mpi_mod, "_build_store_halo_record", lambda state, *, node_id: {"AHF_haloID": int(node_id), "glist": np.asarray([1], dtype=np.int64)})

    build_calls = {}

    def _fake_build_payload(particle_store_path, *, halo_records, galaxy_records, kwargs=None, load_pot=True):
        build_calls["particle_store_path"] = particle_store_path
        build_calls["halo_records"] = [dict(v) for v in halo_records]
        build_calls["galaxy_records"] = [dict(v) for v in galaxy_records]
        return {"halos": [], "galaxies": []}

    monkeypatch.setattr(mpi_mod, "_build_calculating_properties_payload_from_particle_store", _fake_build_payload)
    monkeypatch.setattr(mpi_mod, "_build_calculating_properties_runtime", lambda payload, nproc: SimpleNamespace(halo_list=[], galaxy_list=[]))
    monkeypatch.setattr(mpi_mod, "_compute_group_properties_subset", lambda *args, **kwargs: None)
    dumped = {}
    monkeypatch.setattr(mpi_mod, "_dump_pickle", lambda path, payload: dumped.setdefault(str(path), dict(payload)))

    result = mpi_mod._worker_run_calculating_properties(
        item=item,
        shard_dir=tmp_path,
        nproc=2,
    )

    assert build_calls["particle_store_path"] is fake_particle_store
    assert [int(rec["AHF_haloID"]) for rec in build_calls["halo_records"]] == [10]
    assert [int(rec["AHF_haloID"]) for rec in build_calls["galaxy_records"]] == [101]
    assert [int(rec["_merge_id"]) for rec in build_calls["halo_records"]] == [0]
    assert [int(rec["_merge_id"]) for rec in build_calls["galaxy_records"]] == [0]
    assert result["count_halos"] == 0
    assert result["count_galaxies"] == 0
    assert Path(result["shard_path"]).name.startswith("calculating_properties_shard_rank")


def test_worker_run_reconciling_subhalos_rebuilds_task_payloads_with_serialized_task(monkeypatch, tmp_path):
    task = subhalo_mod.AHFSubhaloTask(10, 0, 10, 0, tuple(), 2, 2)
    item = {
        "root_id": 10,
        "root_payload_path": str(tmp_path / "root10_payload.pkl"),
        "shard_paths": [str(tmp_path / "finding_galaxies_shard_root10.pkl")],
    }

    payloads = {
        "root10_payload.pkl": {
            "root_id": 10,
            "tasks": [subhalo_mod._serialize_task(task)],
            "halo_node_ids": [10],
            "halo_count": 1,
            "particle_store_path": str(tmp_path / "particle_store_root10.h5"),
            "node_store_path": str(tmp_path / "node_store_root10.pkl"),
        },
        "finding_galaxies_shard_root10.pkl": {
            10: [{"AHF_haloID": 10, "AHF_top_haloID": 10, "slist": np.asarray([1, 2], dtype=np.int64)}],
        },
    }
    monkeypatch.setattr(mpi_mod, "_load_pickle", lambda path: payloads[Path(path).name])
    monkeypatch.setattr(mpi_mod, "load_particle_store", lambda path: SimpleNamespace(meta="snapshot-meta"))
    monkeypatch.setattr(mpi_mod, "load_node_store", lambda path: {"nodes": object()})
    monkeypatch.setattr(
        mpi_mod,
        "_build_store_halo_record",
        lambda state, *, node_id: {"AHF_haloID": int(node_id), "slist": np.empty(0, dtype=np.int64)},
    )

    def _fake_build_direct_task_payload(state, *, task, fof_nHlim, fof_Tlim, fof_use_sfr_gate):
        return {
            "gas_sel": np.empty(0, dtype=np.int32),
            "star_sel": np.asarray([1, 2], dtype=np.int32),
            "bh_sel": np.empty(0, dtype=np.int32),
            "dm_sel": np.empty(0, dtype=np.int32),
            "ng": 0,
            "ns": 2,
            "nb": 0,
            "eligible_pos": np.asarray([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float64),
            "eligible_vel": np.zeros((2, 3), dtype=np.float64),
        }

    monkeypatch.setattr(mpi_mod, "_build_direct_task_payload", _fake_build_direct_task_payload)
    monkeypatch.setattr(
        mpi_mod,
        "_group_record_member_cost",
        lambda rec: int(np.asarray(rec.get("slist", np.empty(0, dtype=np.int64))).size),
    )

    captured = {}

    def _fake_reconcile_root_payload(
        *,
        tasks,
        initial_candidates_by_node,
        task_payloads_by_node,
        min_stars,
        fof_ll,
        fof_vel_ll,
        backend,
        cc_backend,
        max_pairs_per_batch,
        device_id=None,
    ):
        rebuilt = dict(task_payloads_by_node[int(task.node_id)])
        captured["task_payload"] = rebuilt
        assert rebuilt["task"] == subhalo_mod._serialize_task(task)
        assert np.array_equal(np.asarray(rebuilt["star_sel"]), np.asarray([1, 2], dtype=np.int32))
        return [{"AHF_haloID": 10, "AHF_top_haloID": 10, "slist": np.asarray([1, 2], dtype=np.int64)}]

    monkeypatch.setattr(mpi_mod, "_reconcile_root_payload", _fake_reconcile_root_payload)
    monkeypatch.setattr(mpi_mod, "_reconciled_galaxy_shard_payload", lambda galaxies: {"galaxies": list(galaxies)})
    dumped = {}
    monkeypatch.setattr(mpi_mod, "_dump_pickle", lambda path, payload: dumped.setdefault(Path(path).name, payload))

    result = mpi_mod._worker_run_reconciling_subhalos(
        item=item,
        fof_ll=0.1,
        fof_vel_ll=1.0,
        min_stars=2,
        shard_dir=tmp_path,
        nproc=1,
    )

    assert captured["task_payload"]["task"] == subhalo_mod._serialize_task(task)
    assert result["count"] == 1
    assert result["root_results"][0]["root_id"] == 10
    assert "reconciling_subhalos_shard_root00000010.pkl" in dumped


def test_finding_galaxies_thread_map_preserves_gpu_support_threads():
    worker_caps = [
        mpi_mod.WorkerCapability(rank=1, role="gpu_worker", hostname="nodeA", local_rank=1, gpu_device=0, threads=2),
        mpi_mod.WorkerCapability(rank=2, role="gpu_worker", hostname="nodeA", local_rank=2, gpu_device=1, threads=2),
        mpi_mod.WorkerCapability(rank=3, role="gpu_worker", hostname="nodeA", local_rank=3, gpu_device=2, threads=2),
        mpi_mod.WorkerCapability(rank=4, role="gpu_worker", hostname="nodeA", local_rank=4, gpu_device=3, threads=2),
        mpi_mod.WorkerCapability(rank=5, role="cpu_worker", hostname="nodeA", local_rank=5, gpu_device=None, threads=4),
        mpi_mod.WorkerCapability(rank=6, role="cpu_worker", hostname="nodeA", local_rank=6, gpu_device=None, threads=4),
        mpi_mod.WorkerCapability(rank=7, role="cpu_worker", hostname="nodeA", local_rank=7, gpu_device=None, threads=4),
        mpi_mod.WorkerCapability(rank=8, role="cpu_worker", hostname="nodeA", local_rank=8, gpu_device=None, threads=4),
        mpi_mod.WorkerCapability(rank=9, role="cpu_worker", hostname="nodeA", local_rank=9, gpu_device=None, threads=4),
    ]
    world_layout = [{"rank": 0, "hostname": "nodeA", "visible_cores": 32}]
    world_layout.extend(
        {"rank": int(cap.rank), "hostname": "nodeA", "visible_cores": 32} for cap in worker_caps
    )

    thread_map = mpi_mod._build_finding_galaxies_thread_map(
        worker_caps=worker_caps,
        world_layout=world_layout,
        coordinator_rank=0,
    )

    gpu_vals = [int(thread_map[int(cap.rank)]) for cap in worker_caps if cap.gpu_device is not None]
    cpu_vals = [int(thread_map[int(cap.rank)]) for cap in worker_caps if cap.gpu_device is None]
    assert gpu_vals == [2, 2, 2, 2]
    assert sum(cpu_vals) == 23
    assert sum(gpu_vals) + sum(cpu_vals) == 31


def test_worker_run_finding_galaxies_bundles_multiple_items(tmp_path):
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

    path_a = tmp_path / "finding_galaxies_payload_cpu_000000.pkl"
    path_b = tmp_path / "finding_galaxies_payload_cpu_000001.pkl"
    mpi_mod._dump_pickle(path_a, payload_a)
    mpi_mod._dump_pickle(path_b, payload_b)

    result = mpi_mod._worker_run_finding_galaxies(
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


def test_calculating_properties_payload_helpers():
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

    halo_payload = mpi_mod._calculating_properties_halo_work_payload([DummyHalo(101), DummyHalo(202)])
    gal_payload = mpi_mod._calculating_properties_galaxy_work_payload([gal])

    assert halo_payload == [101, 202]
    assert len(gal_payload) == 1
    assert int(gal_payload[0]["_merge_id"]) == 3


def test_calculating_properties_manifest_roundtrip(tmp_path):
    path = mpi_mod._write_calculating_properties_manifest(
        shard_root=tmp_path,
        snapshot_file="snap.hdf5",
        ahf_particles_file="ahf_particles",
        output_file="out.hdf5",
        final_shards=["a.pkl", "b.pkl"],
        snapshot_hash="abc123",
    )
    assert path.is_file()
    payload = mpi_mod._load_calculating_properties_manifest(tmp_path)
    assert payload["snapshot_file"] == "snap.hdf5"
    assert payload["ahf_particles_file"] == "ahf_particles"
    assert payload["output_file"] == "out.hdf5"
    assert payload["final_shards"] == ["a.pkl", "b.pkl"]
    assert payload["snapshot_hash"] == "abc123"


def test_build_caesar_runtime_from_loaded_ds_sets_hash_and_load_haloid(monkeypatch):
    class DummySim:
        def __init__(self):
            self.assigned = False

        def _assign_simulation_attributes(self):
            self.assigned = True

    class DummyDatasetType:
        def __init__(self, ds):
            self.ds = ds

    monkeypatch.setitem(sys.modules, "caesar", SimpleNamespace(CAESAR=lambda: DummySim()))
    monkeypatch.setitem(
        sys.modules,
        "caesar.property_manager",
        SimpleNamespace(DatasetType=DummyDatasetType),
    )

    sim = mpi_mod._build_caesar_runtime_from_loaded_ds("dummy-ds", nproc=3, snapshot_hash="hash123")

    assert sim._ds == "dummy-ds"
    assert sim.hash == "hash123"
    assert sim.nproc == 3
    assert sim.load_haloid is False
    assert sim.assigned is True
    assert isinstance(sim._ds_type, DummyDatasetType)


def test_calculating_properties_payload_roundtrip():
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

    payload = subhalo_mod._build_calculating_properties_payload(sim, [halo])
    local_sim = subhalo_mod._build_calculating_properties_runtime(payload, nproc=2)

    assert local_sim.nproc == 2
    assert len(local_sim.halo_list) == 1
    assert len(local_sim.galaxy_list) == 1
    assert np.array_equal(local_sim.data_manager.glist, np.asarray([0], dtype=np.int64))
    assert np.array_equal(local_sim.data_manager.slist, np.asarray([1], dtype=np.int64))
    assert local_sim.galaxy_list[0].parent_halo_index == 0
    assert local_sim._ds_type.has_property("gas", "fh2")
    subhalo_mod._compute_group_properties_subset(local_sim, group_type="halo", groups=list(local_sim.halo_list))
    subhalo_mod._compute_group_properties_subset(local_sim, group_type="galaxy", groups=list(local_sim.galaxy_list))
    assert local_sim.halo_list[0].masses["total"].value > 0.0
    assert local_sim.galaxy_list[0].masses["stellar"].value > 0.0


def test_galaxy_hydrogen_masses_sanitize_invalid_hi_h2_fractions():
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
    sim.yt_dataset = subhalo_mod._ShardYTUnitHelper(redshift=0.0)
    sim._ds = sim.yt_dataset
    sim.simulation = SimpleNamespace(
        XH=0.76,
        redshift=0.0,
        omega_baryon=0.05,
        omega_matter=0.3,
        omega_lambda=0.7,
        Om_z=0.3,
        boxsize=sim.yt_dataset.quan(100.0, "kpccm"),
        H_z=sim.yt_dataset.quan(1.0, "1/s"),
        G=sim.yt_dataset.quan(4.51691362044e-39, "kpc**3/(Msun * s**2)"),
        critical_density=sim.yt_dataset.quan(1.0, "Msun/kpc**3"),
        Densities=sim.yt_dataset.arr(np.asarray([200.0, 500.0, 2500.0], dtype=np.float64), "Msun/kpc**3"),
    )
    sim.data_manager = SimpleNamespace(
        ptypes=["gas", "star", "dm"],
        blackholes=False,
        pos=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        vel=np.zeros((4, 3), dtype=np.float32),
        mass=np.asarray([10.0, 10.0, 5.0, 20.0], dtype=np.float32),
        pot=np.zeros(4, dtype=np.float32),
        ptype=np.asarray([0, 0, 4, 1], dtype=np.int32),
        glist=np.asarray([0, 1], dtype=np.int64),
        slist=np.asarray([2], dtype=np.int64),
        dmlist=np.asarray([3], dtype=np.int64),
        gnh=np.asarray([1.0, 0.01], dtype=np.float32),
        gsfr=np.asarray([0.0, 0.0], dtype=np.float32),
        gZ=np.asarray([0.01, 0.01], dtype=np.float32),
        gT=np.asarray([1.0e4, 1.0e4], dtype=np.float32),
        gfH2=np.asarray([0.4, 0.5], dtype=np.float32),
        gfHI=np.asarray([0.9, 0.7], dtype=np.float32),
        dustmass=np.asarray([0.0, 0.0], dtype=np.float32),
        sZ=np.asarray([0.02], dtype=np.float32),
        age=np.asarray([1.0], dtype=np.float32),
    )
    sim._ds_type = subhalo_mod._ShardDatasetType(
        ptypes=sim.data_manager.ptypes,
        data_manager_attrs=set(sim.data_manager.__dict__.keys()),
    )
    sim.group_types = ["halo", "galaxy"]

    halo = create_new_group(sim, "halo")
    halo.AHF_haloID = 100
    halo.global_indexes = np.asarray([0, 1, 2, 3], dtype=np.int64)
    halo.glist = np.asarray([0, 1], dtype=np.int64)
    halo.slist = np.asarray([0], dtype=np.int64)
    halo.dmlist = np.asarray([0], dtype=np.int64)
    halo.bhlist = np.asarray([], dtype=np.int64)
    halo.dlist = np.asarray([], dtype=np.int64)
    halo.galaxy_index_list = np.asarray([0], dtype=np.int32)

    gal = create_new_group(sim, "galaxy")
    gal.AHF_haloID = 101
    gal.AHF_parent_haloID = 100
    gal.AHF_top_haloID = 100
    gal.AHF_depth = 1
    gal.AHF_ancestor_haloIDs = np.asarray([100], dtype=np.int64)
    gal.global_indexes = np.asarray([0, 1, 2], dtype=np.int64)
    gal.glist = np.asarray([0, 1], dtype=np.int64)
    gal.slist = np.asarray([0], dtype=np.int64)
    gal.dmlist = np.asarray([], dtype=np.int64)
    gal.bhlist = np.asarray([], dtype=np.int64)
    gal.dlist = np.asarray([], dtype=np.int64)
    gal.parent_halo_index = 0
    gal._ahf_host_halo_index = 0
    gal.halo = halo

    sim.halo_list = [halo]
    sim.galaxy_list = [gal]

    subhalo_mod._compute_group_properties_subset(sim, group_type="halo", groups=list(sim.halo_list))
    subhalo_mod._compute_group_properties_subset(sim, group_type="galaxy", groups=list(sim.galaxy_list))

    total_h = float(gal.masses["H"].value)
    total_hi = float(gal.masses["HI"].value)
    total_h2 = float(gal.masses["H2"].value)
    assert np.isclose(total_h, 15.2)
    assert np.isclose(total_hi, 9.88)
    assert np.isclose(total_h2, 3.04)
    assert total_hi + total_h2 <= total_h + 1.0e-8

    gas_30 = float(gal.masses["gas_30kpc"].value)
    hi_30 = float(gal.masses["HI_30kpc"].value)
    h2_30 = float(gal.masses["H2_30kpc"].value)
    assert hi_30 + h2_30 <= gas_30 * 0.76 + 1.0e-8


def test_host_local_hydrogen_assignment_uses_mass_weighted_distance():
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
    sim.yt_dataset = subhalo_mod._ShardYTUnitHelper(redshift=0.0)
    sim._ds = sim.yt_dataset
    sim.simulation = SimpleNamespace(
        XH=0.76,
        redshift=0.0,
        omega_baryon=0.05,
        omega_matter=0.3,
        omega_lambda=0.7,
        Om_z=0.3,
        boxsize=sim.yt_dataset.quan(100.0, "kpccm"),
        H_z=sim.yt_dataset.quan(1.0, "1/s"),
        G=sim.yt_dataset.quan(4.51691362044e-39, "kpc**3/(Msun * s**2)"),
        critical_density=sim.yt_dataset.quan(1.0, "Msun/kpc**3"),
        Densities=sim.yt_dataset.arr(np.asarray([200.0, 500.0, 2500.0], dtype=np.float64), "Msun/kpc**3"),
    )
    sim.data_manager = SimpleNamespace(
        ptypes=["gas", "star"],
        blackholes=False,
        pos=np.asarray([[4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32),
        vel=np.zeros((3, 3), dtype=np.float32),
        mass=np.asarray([10.0, 5.0, 20.0], dtype=np.float32),
        pot=np.zeros(3, dtype=np.float32),
        ptype=np.asarray([0, 4, 4], dtype=np.int32),
        glist=np.asarray([0], dtype=np.int64),
        slist=np.asarray([1, 2], dtype=np.int64),
        gnh=np.asarray([1.0], dtype=np.float32),
        gsfr=np.asarray([0.0], dtype=np.float32),
        gZ=np.asarray([0.01], dtype=np.float32),
        gT=np.asarray([1.0e4], dtype=np.float32),
        gfH2=np.asarray([0.1], dtype=np.float32),
        gfHI=np.asarray([0.5], dtype=np.float32),
        dustmass=np.asarray([0.0], dtype=np.float32),
        sZ=np.asarray([0.02, 0.02], dtype=np.float32),
        age=np.asarray([1.0, 1.0], dtype=np.float32),
    )
    sim._ds_type = subhalo_mod._ShardDatasetType(
        ptypes=sim.data_manager.ptypes,
        data_manager_attrs=set(sim.data_manager.__dict__.keys()),
    )
    sim.group_types = ["halo", "galaxy"]

    host = create_new_group(sim, "halo")
    host.AHF_haloID = 100
    host.AHF_top_haloID = 100
    host.global_indexes = np.asarray([0, 1, 2], dtype=np.int64)
    host.glist = np.asarray([0], dtype=np.int64)
    host.slist = np.asarray([0, 1], dtype=np.int64)
    host.dmlist = np.asarray([], dtype=np.int64)
    host.bhlist = np.asarray([], dtype=np.int64)
    host.dlist = np.asarray([], dtype=np.int64)
    host.galaxy_index_list = np.asarray([0, 1], dtype=np.int32)

    gal_a = create_new_group(sim, "galaxy")
    gal_a.AHF_haloID = 101
    gal_a.AHF_parent_haloID = 100
    gal_a.AHF_top_haloID = 100
    gal_a.AHF_depth = 1
    gal_a.AHF_ancestor_haloIDs = np.asarray([100], dtype=np.int64)
    gal_a.global_indexes = np.asarray([1], dtype=np.int64)
    gal_a.glist = np.asarray([], dtype=np.int64)
    gal_a.slist = np.asarray([0], dtype=np.int64)
    gal_a.dmlist = np.asarray([], dtype=np.int64)
    gal_a.bhlist = np.asarray([], dtype=np.int64)
    gal_a.dlist = np.asarray([], dtype=np.int64)
    gal_a.parent_halo_index = 0
    gal_a._ahf_host_halo_index = 0
    gal_a.halo = host

    gal_b = create_new_group(sim, "galaxy")
    gal_b.AHF_haloID = 102
    gal_b.AHF_parent_haloID = 100
    gal_b.AHF_top_haloID = 100
    gal_b.AHF_depth = 1
    gal_b.AHF_ancestor_haloIDs = np.asarray([100], dtype=np.int64)
    gal_b.global_indexes = np.asarray([2], dtype=np.int64)
    gal_b.glist = np.asarray([], dtype=np.int64)
    gal_b.slist = np.asarray([1], dtype=np.int64)
    gal_b.dmlist = np.asarray([], dtype=np.int64)
    gal_b.bhlist = np.asarray([], dtype=np.int64)
    gal_b.dlist = np.asarray([], dtype=np.int64)
    gal_b.parent_halo_index = 0
    gal_b._ahf_host_halo_index = 0
    gal_b.halo = host

    sim.halo_list = [host]
    sim.galaxy_list = [gal_a, gal_b]

    subhalo_mod._compute_group_properties_subset(sim, group_type="halo", groups=list(sim.halo_list))
    subhalo_mod._compute_group_properties_subset(sim, group_type="galaxy", groups=list(sim.galaxy_list))

    assert np.isclose(float(gal_a.masses["HI"].value), 0.0)
    assert np.isclose(float(gal_a.masses["H2"].value), 0.0)
    assert np.isclose(float(gal_b.masses["HI"].value), 3.8)
    assert np.isclose(float(gal_b.masses["H2"].value), 0.76)
    assert np.isclose(float(gal_a.masses["HI_30kpc"].value), 3.8)
    assert np.isclose(float(gal_b.masses["HI_30kpc"].value), 3.8)


def test_populate_caesar_ahf_lineage_indexes_maps_parent_and_top_host():
    root = SimpleNamespace(
        AHF_haloID=10,
        AHF_parent_haloID=0,
        AHF_top_haloID=10,
        GroupID=0,
    )
    child = SimpleNamespace(
        AHF_haloID=11,
        AHF_parent_haloID=10,
        AHF_top_haloID=10,
        GroupID=1,
    )
    gal = SimpleNamespace(
        AHF_haloID=12,
        AHF_parent_haloID=11,
        AHF_top_haloID=10,
    )
    sim = SimpleNamespace(halo_list=[root, child], galaxy_list=[gal])

    subhalo_mod._populate_caesar_ahf_lineage_indexes(sim)

    assert root.caesar_parent_halo_index == -1
    assert root.caesar_top_halo_index == 0
    assert child.caesar_parent_halo_index == 0
    assert child.caesar_top_halo_index == 0
    assert gal.caesar_parent_halo_index == 1
    assert gal.caesar_top_halo_index == 0


def test_group_funcs_loader_falls_back_to_local_extension(monkeypatch):
    monkeypatch.setitem(sys.modules, "caesar.group_funcs", SimpleNamespace())
    funcs = group_loader_mod.load_group_funcs(
        "get_group_overall_properties",
        "get_group_bh_properties",
    )
    assert len(funcs) == 2
    assert all(callable(func) for func in funcs)
