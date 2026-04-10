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


class DummyGalaxy:
    def __init__(self, halo_id: int, top_id: int, ancestors):
        self.AHF_haloID = halo_id
        self.AHF_top_haloID = top_id
        self.AHF_ancestor_haloIDs = np.asarray(ancestors, dtype=np.int64)
        self.glist = np.array([], dtype=np.int64)
        self.slist = np.array([], dtype=np.int64)
        self.cloud_index_list = np.array([], dtype=np.int64)


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
