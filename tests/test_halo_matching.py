import os
from pathlib import Path
import importlib.util

module_path = Path(__file__).resolve().parents[1] / "caesar" / "halo_matching.py"
spec = importlib.util.spec_from_file_location("halo_matching", module_path)
halo_matching = importlib.util.module_from_spec(spec)
import sys as _sys
_sys.modules["halo_matching"] = halo_matching
spec.loader.exec_module(halo_matching)

from halo_matching import (
    read_file_to_structure,
    galaxies_to_namedata,
    find_best_matches,
    get_caesar_file,
    get_AHF_file,
    read_single_halo_from_file,
    match_subhalos_to_galaxies,
)


class DummyGalaxy:
    def __init__(self, gid, slist):
        self.GroupID = gid
        self.slist = slist
        self.masses = {}


def create_sample_data(tmp_path: Path):
    ahf_lines = (
        "3 1\n"
        "101\t4\n102\t4\n103\t4\n"
        "3 2\n"
        "201\t4\n202\t4\n203\t4\n"
    )
    ahf_file = tmp_path / "Simba_M200_snap_000.z1.000.AHF_particles"
    ahf_file.write_text(ahf_lines)

    star_ids = [101, 102, 103, 201, 202, 203]
    galaxies = [
        DummyGalaxy(1, [0, 1, 2]),
        DummyGalaxy(2, [3, 4, 5]),
    ]
    caesar_file = tmp_path / "Simba_M200_snap_000.h5"
    caesar_file.touch()
    return ahf_file, caesar_file, star_ids, galaxies


def test_matching(tmp_path: Path):
    ahf_file, caesar_path, star_ids, galaxies = create_sample_data(tmp_path)

    ahf = read_file_to_structure(str(ahf_file))
    caesar = galaxies_to_namedata(galaxies, star_ids)
    matches = find_best_matches(caesar, ahf)

    assert matches[0][1] == 1
    assert matches[1][1] == 2

    assert get_caesar_file(tmp_path, 0) == str(caesar_path)
    assert get_AHF_file(tmp_path, 0) == str(ahf_file)

    halo = read_single_halo_from_file(str(ahf_file), 1)
    assert halo is not None
    assert halo.id == 1
    assert 101 in halo.parttype4


class DummySim:
    def __init__(self, galaxies):
        self.galaxies = galaxies
        self.ngalaxies = len(galaxies)


def test_subhalo_merge(tmp_path: Path):
    ahf_lines = (
        "9 1\n"
        "101\t4\n102\t4\n103\t4\n201\t4\n202\t4\n203\t4\n501\t1\n502\t1\n503\t1\n"
    )
    ahf_file = tmp_path / "Simba_M200_snap_000.z1.000.AHF_particles"
    ahf_file.write_text(ahf_lines)

    star_ids = [101, 102, 103, 201, 202, 203]
    galaxies = [DummyGalaxy(1, [0, 1, 2]), DummyGalaxy(2, [3, 4, 5])]
    sim = DummySim(galaxies)

    match_subhalos_to_galaxies(
        sim,
        ahf_file=str(ahf_file),
        star_particle_ids=star_ids,
    )

    assert sim.ngalaxies == 1
    assert len(sim.galaxies[0].slist) == 6
    assert sim.galaxies[0].masses["dm"] == 3.0

def test_snapshot_auto_match(monkeypatch, tmp_path: Path):
    """Ensure subhalo matching runs automatically when haloid='AHF'."""

    called = {}

    def fake_match(sim, *, ahf_file, snapshot_file=None, star_particle_ids=None):
        called['ahf_file'] = ahf_file
        called['snapshot_file'] = snapshot_file

    class FakeDS:
        cosmological_simulation = 1
        current_redshift = 0.0
        fullpath = str(tmp_path)

    class FakeCAESAR:
        def __init__(self, ds):
            self.ds = ds
        def member_search(self, *_args, **_kwargs):
            pass
        def save(self, *_args, **_kwargs):
            pass

    import types, sys, importlib.util

    # fake modules for driver import
    fake_caesar = types.ModuleType("caesar")
    fake_caesar.CAESAR = FakeCAESAR
    fake_caesar.progen = types.ModuleType("caesar.progen")
    def progen_finder(*args, **kwargs):
        pass
    fake_caesar.progen.progen_finder = progen_finder
    fake_hm = types.ModuleType("caesar.halo_matching")
    fake_hm.match_subhalos_to_galaxies = fake_match
    sys.modules["caesar"] = fake_caesar
    sys.modules["caesar.progen"] = fake_caesar.progen
    sys.modules["caesar.halo_matching"] = fake_hm

    fake_yt = types.ModuleType("yt")
    fake_funcs = types.ModuleType("yt.funcs")
    class DummyLog:
        def warning(self, *args, **kwargs):
            pass
    fake_funcs.mylog = DummyLog()
    fake_yt.load = lambda p: FakeDS()
    fake_yt.funcs = fake_funcs
    sys.modules["yt"] = fake_yt
    sys.modules["yt.funcs"] = fake_funcs

    driver_path = Path(__file__).resolve().parents[1] / "caesar" / "driver.py"
    spec = importlib.util.spec_from_file_location("driver_mod", driver_path)
    driver_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver_mod)

    monkeypatch.setattr(driver_mod.os.path, "isfile", lambda p: True)

    snap = driver_mod.Snapshot(str(tmp_path), "snap_", 0, "hdf5")
    ahf_file = str(tmp_path / "file.AHF_particles")
    snap.member_search(False, False, haloid="AHF", haloid_file=ahf_file)

    assert called["ahf_file"] == ahf_file

