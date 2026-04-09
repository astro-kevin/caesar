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


assignment_mod = _load_module("assignment_mod", "assignment.py")
halo_matching_mod = _load_module("halo_matching_mod", "halo_matching.py")
saver_mod = _load_module("saver_mod", "saver.py")


class DummyHalo:
    def __init__(self, ahf_halo_id: int, ndm: int = 16):
        self.AHF_haloID = ahf_halo_id
        self.ndm = ndm
        self.galaxy_index_list = []
        self.GroupID = -1


class DummyGalaxy:
    def __init__(self, host_index: int, ahf_halo_id: int, ahf_top_halo_id: int):
        self._ahf_host_halo_index = host_index
        self.AHF_haloID = ahf_halo_id
        self.AHF_top_haloID = ahf_top_halo_id
        self.parent_halo_index = -1
        self.glist = np.array([], dtype=np.int64)
        self.slist = np.array([], dtype=np.int64)


class DummyObj:
    def __init__(self, halos, galaxies):
        self._has_galaxies = True
        self.halos = halos
        self.halo_list = halos
        self.galaxies = galaxies
        self.galaxy_list = galaxies
        self.ngalaxies = len(galaxies)
        # Deliberately stale positional host list to emulate post-sort misalignment.
        self._ahf_galaxy_hosts = [0, 1, 2]


def test_assign_galaxies_to_halos_prefers_per_galaxy_host_index():
    halos = [DummyHalo(100), DummyHalo(200), DummyHalo(300)]
    galaxies = [
        DummyGalaxy(2, 30, 300),
        DummyGalaxy(0, 10, 100),
        DummyGalaxy(1, 20, 200),
    ]
    obj = DummyObj(halos, galaxies)

    assignment_mod.assign_galaxies_to_halos(obj)

    assert [g.parent_halo_index for g in obj.galaxies] == [2, 0, 1]
    assert [h.galaxy_index_list for h in obj.halos] == [[1], [2], [0]]
    assert obj._ahf_galaxy_hosts == [2, 0, 1]
    assert obj._ahf_galaxy_ahf_ids == [30, 10, 20]
    assert obj._ahf_galaxy_top_ahf_ids == [300, 100, 200]


def test_prune_halos_remaps_per_galaxy_host_index():
    halos = [
        DummyHalo(100, ndm=16),
        DummyHalo(200, ndm=0),
        DummyHalo(300, ndm=16),
    ]
    galaxies = [
        DummyGalaxy(2, 30, 300),
        DummyGalaxy(0, 10, 100),
    ]
    halos[0].galaxy_index_list = [1]
    halos[2].galaxy_index_list = [0]
    for galaxy in galaxies:
        galaxy.parent_halo_index = galaxy._ahf_host_halo_index

    sim = DummyObj(halos, galaxies)
    sim._ahf_galaxy_hosts = [2, 0]

    halo_matching_mod._prune_halos_after_galaxies(sim)

    assert len(sim.halos) == 2
    assert [g.parent_halo_index for g in sim.galaxies] == [1, 0]
    assert [g._ahf_host_halo_index for g in sim.galaxies] == [1, 0]
    assert sim._ahf_galaxy_hosts == [1, 0]


def test_serialize_attributes_writes_ahf_top_haloid(tmp_path):
    galaxies = [
        DummyGalaxy(0, 10, 100),
        DummyGalaxy(1, 20, 200),
    ]
    outfile = tmp_path / "ahf_top_haloid.hdf5"

    with h5py.File(outfile, "w") as f:
        hd = f.create_group("galaxy_data")
        hd_dicts = hd.create_group("dicts")
        saver_mod.serialize_attributes(galaxies, hd, hd_dicts)

    with h5py.File(outfile, "r") as f:
        assert np.array_equal(f["galaxy_data/AHF_haloID"][:], np.array([10, 20]))
        assert np.array_equal(f["galaxy_data/AHF_top_haloID"][:], np.array([100, 200]))
