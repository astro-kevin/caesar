from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd


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


tables_mod = _load_module("ahf_subhalo_tables_mod", "ahf_subhalo_tables.py")
hdf5_mod = _load_module("ahf_subhalo_hdf5_mod", "ahf_subhalo_hdf5.py")
subhalo_mod = _load_module("ahf_subhalo_runtime_mod", "AHF_subhalo.py")
loader_mod = _load_module("ahf_subhalo_loader_mod", "loader.py")
export_mod = _load_module("ahf_subhalo_export_mod", "ahf_subhalo_export.py")
pipeline_utils_mod = _load_module("pipeline_utils_subhalo_mod", "pipeline_utils.py")
mpi_mod = _load_module("ahf_subhalo_mpi_test_mod", "AHF_subhalo_mpi.py")


def _write_test_snapshot(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["BoxSize"] = 100000.0
        header.attrs["HubbleParam"] = 0.5
        header.attrs["Omega0"] = 0.3
        header.attrs["OmegaLambda"] = 0.7
        header.attrs["Redshift"] = 1.0
        header.attrs["Time"] = 0.5
        header.attrs["MassTable"] = np.zeros(6, dtype=np.float64)

        gas = handle.create_group("PartType0")
        gas["ParticleIDs"] = np.asarray([1001, 1002], dtype=np.int64)
        gas["Coordinates"] = np.asarray([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]], dtype=np.float32)
        gas["Velocities"] = np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
        gas["Masses"] = np.asarray([1.0, 2.0], dtype=np.float32)
        gas["Density"] = np.asarray([1.0e-3, 2.0e-3], dtype=np.float32)
        gas["InternalEnergy"] = np.asarray([1.0, 2.0], dtype=np.float32)
        gas["ElectronAbundance"] = np.asarray([0.5, 1.0], dtype=np.float32)
        gas["StarFormationRate"] = np.asarray([0.0, 1.0], dtype=np.float32)
        gas["GrackleHI"] = np.asarray([0.7, 0.2], dtype=np.float32)
        gas["FractionH2"] = np.asarray([0.1, 0.5], dtype=np.float32)
        gas["Metallicity"] = np.asarray([[0.02], [0.03]], dtype=np.float32)

        dm = handle.create_group("PartType1")
        dm["ParticleIDs"] = np.asarray([2001, 2002], dtype=np.int64)
        dm["Coordinates"] = np.asarray([[11.0, 12.0, 13.0], [15.0, 16.0, 17.0]], dtype=np.float32)
        dm["Velocities"] = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        dm["Masses"] = np.asarray([3.0, 4.0], dtype=np.float32)
        dm["Potential"] = np.asarray([-1.0, -2.0], dtype=np.float32)

        star = handle.create_group("PartType4")
        star["ParticleIDs"] = np.asarray([3001, 3002], dtype=np.int64)
        star["Coordinates"] = np.asarray([[21.0, 22.0, 23.0], [25.0, 26.0, 27.0]], dtype=np.float32)
        star["Velocities"] = np.asarray([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)
        star["Masses"] = np.asarray([5.0, 6.0], dtype=np.float32)
        star["StellarFormationTime"] = np.asarray([0.25, 0.4], dtype=np.float32)
        star["Metallicity"] = np.asarray([[0.01], [0.015]], dtype=np.float32)

        bh = handle.create_group("PartType5")
        bh["ParticleIDs"] = np.asarray([4001], dtype=np.int64)
        bh["Coordinates"] = np.asarray([[31.0, 32.0, 33.0]], dtype=np.float32)
        bh["Velocities"] = np.asarray([[13.0, 14.0, 15.0]], dtype=np.float32)
        bh["Masses"] = np.asarray([7.0], dtype=np.float32)
        bh["BH_Mass"] = np.asarray([8.0], dtype=np.float32)
        bh["BH_Mdot"] = np.asarray([0.25], dtype=np.float32)


def _assert_hdf5_equal(path_a: Path, path_b: Path) -> None:
    def _cmp(a, b, prefix: str = ""):
        assert set(a.keys()) == set(b.keys()), prefix or "/"
        for key in a.keys():
            aa = a[key]
            bb = b[key]
            here = f"{prefix}/{key}"
            assert type(aa) is type(bb), here
            if isinstance(aa, h5py.Group):
                assert set(aa.attrs.keys()) == set(bb.attrs.keys()), f"{here} attrs"
                for attr in aa.attrs.keys():
                    va = aa.attrs[attr]
                    vb = bb.attrs[attr]
                    if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
                        assert np.array_equal(np.asarray(va), np.asarray(vb)), f"{here} attr {attr}"
                    else:
                        assert va == vb, f"{here} attr {attr}"
                _cmp(aa, bb, here)
            else:
                assert aa.shape == bb.shape, here
                assert aa.dtype == bb.dtype, here
                assert set(aa.attrs.keys()) == set(bb.attrs.keys()), f"{here} attrs"
                for attr in aa.attrs.keys():
                    va = aa.attrs[attr]
                    vb = bb.attrs[attr]
                    if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
                        assert np.array_equal(np.asarray(va), np.asarray(vb)), f"{here} attr {attr}"
                    else:
                        assert va == vb, f"{here} attr {attr}"
                if aa.dtype.kind in {"f"}:
                    assert np.allclose(aa[:], bb[:]), here
                else:
                    assert np.array_equal(aa[:], bb[:]), here

    with h5py.File(path_a, "r") as fa, h5py.File(path_b, "r") as fb:
        assert set(fa.attrs.keys()) == set(fb.attrs.keys()), "/ attrs"
        for attr in fa.attrs.keys():
            va = fa.attrs[attr]
            vb = fb.attrs[attr]
            if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
                assert np.array_equal(np.asarray(va), np.asarray(vb)), f"/ attr {attr}"
            else:
                assert va == vb, f"/ attr {attr}"
        _cmp(fa, fb, "")


def _property_shard_payload_from_states(halo_states, galaxy_states):
    halo_columns = export_mod.build_group_column_payload(
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
    galaxy_columns = export_mod.build_group_column_payload(
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
    return {
        "halo_columns": halo_columns,
        "galaxy_columns": galaxy_columns,
        "halo_lists": mpi_mod._build_property_list_blocks(halo_states, ("dmlist", "glist", "slist", "bhlist", "dlist")),
        "galaxy_lists": mpi_mod._build_property_list_blocks(
            galaxy_states,
            ("glist", "slist", "bhlist", "dlist", "cloud_index_list", "AHF_ancestor_haloIDs"),
        ),
    }


def _calculating_properties_result_with_summary(shard_path: Path, halo_states, galaxy_states):
    payload = _property_shard_payload_from_states(halo_states, galaxy_states)
    return {
        "shard_path": str(shard_path),
        "count_halos": int(len(halo_states)),
        "count_galaxies": int(len(galaxy_states)),
        "summary": mpi_mod._build_property_shard_summary(
            payload["halo_columns"],
            payload["galaxy_columns"],
        ),
    }


def test_ragged_index_block_round_trip():
    block = tables_mod.RaggedIndexBlock.from_sequences([[1, 2], [], [3]])
    assert np.array_equal(block.offsets, np.asarray([0, 2, 2, 3], dtype=np.int64))
    assert np.array_equal(block.get(0), np.asarray([1, 2], dtype=np.int64))
    assert np.array_equal(block.get(1), np.empty(0, dtype=np.int64))
    payload = block.to_payload()
    restored = tables_mod.RaggedIndexBlock.from_payload(payload)
    assert np.array_equal(restored.offsets, block.offsets)
    assert np.array_equal(restored.data, block.data)


def test_candidate_galaxy_table_round_trip():
    records = [
        {
            "AHF_haloID": 10,
            "AHF_parent_haloID": 0,
            "AHF_top_haloID": 10,
            "AHF_depth": 0,
            "AHF_ancestor_haloIDs": np.asarray([], dtype=np.int64),
            "glist": np.asarray([1, 2], dtype=np.int32),
            "slist": np.asarray([3], dtype=np.int32),
            "dmlist": np.asarray([], dtype=np.int32),
            "bhlist": np.asarray([4], dtype=np.int32),
            "dlist": np.asarray([], dtype=np.int32),
        },
        {
            "AHF_haloID": 11,
            "AHF_parent_haloID": 10,
            "AHF_top_haloID": 10,
            "AHF_depth": 1,
            "AHF_ancestor_haloIDs": np.asarray([10], dtype=np.int64),
            "glist": np.asarray([], dtype=np.int32),
            "slist": np.asarray([5, 6], dtype=np.int32),
            "dmlist": np.asarray([7], dtype=np.int32),
            "bhlist": np.asarray([], dtype=np.int32),
            "dlist": np.asarray([], dtype=np.int32),
        },
    ]
    table = tables_mod.candidate_records_to_table(records)
    restored = tables_mod.candidate_records_from_table_payload(table.to_payload())
    assert len(restored) == 2
    assert restored[0]["AHF_haloID"] == 10
    assert np.array_equal(restored[0]["glist"], np.asarray([1, 2], dtype=np.int32))
    assert np.array_equal(restored[1]["AHF_ancestor_haloIDs"], np.asarray([10], dtype=np.int64))
    assert np.array_equal(restored[1]["dmlist"], np.asarray([7], dtype=np.int32))


def test_direct_hdf5_snapshot_loading(tmp_path):
    snapshot = tmp_path / "snap.hdf5"
    _write_test_snapshot(snapshot)

    meta = hdf5_mod.load_snapshot_meta(str(snapshot))
    assert meta.snapshot_file == str(snapshot)
    assert meta.ptypes == ("gas", "dm", "star", "bh")
    assert meta.units["mass"] == "Msun"
    assert np.isclose(meta.boxsize, 200000.0)

    shard = hdf5_mod.load_particle_shard(str(snapshot))
    gas = shard.table("gas")
    dm = shard.table("dm")
    star = shard.table("star")
    bh = shard.table("bh")

    assert np.allclose(gas.get("pos")[0], np.asarray([2.0, 4.0, 6.0], dtype=np.float32))
    assert np.allclose(dm.get("mass"), np.asarray([6.0e10, 8.0e10], dtype=np.float32))
    assert gas.has("gnh")
    assert gas.has("gT")
    assert gas.has("gfHI")
    assert gas.has("gfH2")
    assert star.has("age")
    assert bh.has("bhmass")
    assert bh.has("bhmdot")


def test_build_ahf_node_table_maps_memberships(monkeypatch, tmp_path):
    snapshot = tmp_path / "snap.hdf5"
    _write_test_snapshot(snapshot)
    shard = hdf5_mod.load_particle_shard(str(snapshot))

    hierarchy = SimpleNamespace(
        parent_of={10: 0, 11: 10},
        children_of={10: [11], 11: []},
    )
    halos_df = pd.DataFrame(
        {
            "hid": np.asarray([10, 11], dtype=np.int64),
            "host_hid": np.asarray([0, 10], dtype=np.int64),
            "npart": np.asarray([4, 2], dtype=np.int64),
            "n_star": np.asarray([1, 1], dtype=np.int64),
        }
    )
    memberships = {
        10: np.asarray([[1001, 0], [2001, 1], [3001, 4], [4001, 5]], dtype=np.int64),
        11: np.asarray([[1002, 0], [3002, 4]], dtype=np.int64),
    }

    monkeypatch.setattr(hdf5_mod, "load_ahf_hierarchy", lambda _: hierarchy)
    monkeypatch.setattr(hdf5_mod, "load_ahf_halos_dataframe", lambda _: halos_df)
    monkeypatch.setattr(hdf5_mod, "load_ahf_particle_blocks", lambda *args, **kwargs: memberships)

    nodes = hdf5_mod.build_ahf_node_table("dummy.AHF_particles", shard)
    assert np.array_equal(nodes.halo_id, np.asarray([10, 11], dtype=np.int64))
    assert np.array_equal(nodes.parent_halo_id, np.asarray([0, 10], dtype=np.int64))
    assert np.array_equal(nodes.top_halo_id, np.asarray([10, 10], dtype=np.int64))
    assert np.array_equal(nodes.depth, np.asarray([0, 1], dtype=np.int64))
    assert np.array_equal(nodes.ancestors_for(0), np.empty(0, dtype=np.int64))
    assert np.array_equal(nodes.ancestors_for(1), np.asarray([10], dtype=np.int64))
    assert np.array_equal(nodes.members_for(0, "gas"), np.asarray([0], dtype=np.int64))
    assert np.array_equal(nodes.members_for(0, "dm"), np.asarray([0], dtype=np.int64))
    assert np.array_equal(nodes.members_for(0, "star"), np.asarray([0], dtype=np.int64))
    assert np.array_equal(nodes.members_for(0, "bh"), np.asarray([0], dtype=np.int64))
    assert np.array_equal(nodes.members_for(1, "gas"), np.asarray([1], dtype=np.int64))
    assert np.array_equal(nodes.members_for(1, "star"), np.asarray([1], dtype=np.int64))
    assert np.array_equal(nodes.star_count, np.asarray([1, 1], dtype=np.int64))
    assert np.array_equal(nodes.dm_count, np.asarray([1, 0], dtype=np.int64))
    assert np.array_equal(nodes.fof_candidates, np.asarray([4, 2], dtype=np.int64))


def test_build_task_payload_uses_direct_particle_tables(monkeypatch, tmp_path):
    snapshot = tmp_path / "snap.hdf5"
    _write_test_snapshot(snapshot)

    hierarchy = SimpleNamespace(
        parent_of={10: 0},
        children_of={10: []},
    )
    halos_df = pd.DataFrame(
        {
            "hid": np.asarray([10], dtype=np.int64),
            "host_hid": np.asarray([0], dtype=np.int64),
            "npart": np.asarray([5], dtype=np.int64),
            "n_star": np.asarray([1], dtype=np.int64),
        }
    )
    memberships = {
        10: np.asarray([[1002, 0], [2001, 1], [3001, 4], [4001, 5]], dtype=np.int64),
    }

    monkeypatch.setattr(hdf5_mod, "load_ahf_hierarchy", lambda _: hierarchy)
    monkeypatch.setattr(hdf5_mod, "load_ahf_halos_dataframe", lambda _: halos_df)
    monkeypatch.setattr(hdf5_mod, "load_ahf_particle_blocks", lambda *args, **kwargs: memberships)

    state = hdf5_mod.build_direct_state(str(snapshot), "dummy.AHF_particles")
    task = SimpleNamespace(node_id=10)
    payload = hdf5_mod.build_task_payload(
        state,
        task=task,
        fof_nHlim=0.0,
        fof_Tlim=1.0e9,
        fof_use_sfr_gate=True,
    )
    assert payload is not None
    assert np.array_equal(payload["gas_sel"], np.asarray([1], dtype=np.int32))
    assert np.array_equal(payload["star_sel"], np.asarray([0], dtype=np.int32))
    assert np.array_equal(payload["bh_sel"], np.asarray([0], dtype=np.int32))
    assert np.array_equal(payload["dm_sel"], np.asarray([0], dtype=np.int32))
    assert payload["eligible_pos"].shape == (3, 3)
    assert payload["eligible_vel"].shape == (3, 3)


def test_direct_calculating_properties_runtime_can_compute_and_save(monkeypatch, tmp_path):
    snapshot = tmp_path / "snap.hdf5"
    _write_test_snapshot(snapshot)

    hierarchy = SimpleNamespace(
        parent_of={10: 0},
        children_of={10: []},
    )
    halos_df = pd.DataFrame(
        {
            "hid": np.asarray([10], dtype=np.int64),
            "host_hid": np.asarray([0], dtype=np.int64),
            "npart": np.asarray([4], dtype=np.int64),
            "n_star": np.asarray([1], dtype=np.int64),
        }
    )
    memberships = {
        10: np.asarray([[1002, 0], [2001, 1], [3001, 4], [4001, 5]], dtype=np.int64),
    }

    monkeypatch.setattr(hdf5_mod, "load_ahf_hierarchy", lambda _: hierarchy)
    monkeypatch.setattr(hdf5_mod, "load_ahf_halos_dataframe", lambda _: halos_df)
    monkeypatch.setattr(hdf5_mod, "load_ahf_particle_blocks", lambda *args, **kwargs: memberships)

    state = hdf5_mod.build_direct_state(str(snapshot), "dummy.AHF_particles")
    galaxy_payloads = [
        {
            "AHF_haloID": 10,
            "AHF_parent_haloID": 0,
            "AHF_top_haloID": 10,
            "AHF_depth": 0,
            "AHF_ancestor_haloIDs": np.asarray([], dtype=np.int64),
            "glist": np.asarray([1], dtype=np.int32),
            "slist": np.asarray([0], dtype=np.int32),
            "dmlist": np.asarray([], dtype=np.int32),
            "bhlist": np.asarray([0], dtype=np.int32),
            "dlist": np.asarray([], dtype=np.int32),
        }
    ]

    sim = subhalo_mod._build_direct_calculating_properties_runtime(
        state,
        galaxy_payloads=galaxy_payloads,
        nproc=1,
    )
    assert getattr(sim, "_ahf_subhalo_streaming_save", False) is True
    subhalo_mod._compute_group_properties_subset(sim, group_type="halo", groups=list(sim.halo_list))
    subhalo_mod._compute_group_properties_subset(sim, group_type="galaxy", groups=list(sim.galaxy_list))
    subhalo_mod._complete_finalization_after_properties_direct(sim)

    outfile = tmp_path / "caesar_direct.hdf5"
    sim.save(str(outfile))

    with h5py.File(outfile, "r") as handle:
        assert int(handle.attrs["nhalos"]) == 1
        assert int(handle.attrs["ngalaxies"]) == 1
        assert bool(handle.attrs["skip_hash_check"])
        assert not bool(handle.attrs["load_haloid"])
        assert "_ahf_subhalo_streaming_save" not in handle.attrs
        sim_attrs = handle["simulation_attributes"].attrs
        assert float(sim_attrs["hubble_constant"]) == 0.5
        assert int(sim_attrs["effective_resolution"]) == 1
        assert bool(sim_attrs["baryons_present"])
        assert not bool(sim_attrs["unbind_halos"])
        assert "mean_interparticle_separation" in sim_attrs
        assert np.array_equal(handle["halo_data/AHF_haloID"][:], np.asarray([10], dtype=np.int64))
        assert np.array_equal(handle["galaxy_data/AHF_haloID"][:], np.asarray([10], dtype=np.int64))
        assert np.array_equal(handle["halo_data/caesar_parent_halo_index"][:], np.asarray([-1], dtype=np.int64))
        assert np.array_equal(handle["halo_data/caesar_top_halo_index"][:], np.asarray([0], dtype=np.int64))
        assert np.array_equal(handle["galaxy_data/parent_halo_index"][:], np.asarray([0], dtype=np.int64))
        assert np.array_equal(handle["galaxy_data/caesar_parent_halo_index"][:], np.asarray([-1], dtype=np.int64))
        assert np.array_equal(handle["galaxy_data/caesar_top_halo_index"][:], np.asarray([0], dtype=np.int64))
        assert "global_lists" in handle
        assert np.array_equal(handle["global_lists/halo_dmlist"][:], np.asarray([0, -1], dtype=np.int32))
        assert np.array_equal(handle["global_lists/halo_glist"][:], np.asarray([-1, 0], dtype=np.int32))
        assert np.array_equal(handle["global_lists/halo_slist"][:], np.asarray([0, -1], dtype=np.int32))
        assert np.array_equal(handle["global_lists/galaxy_glist"][:], np.asarray([-1, 0], dtype=np.int32))
        assert np.array_equal(handle["global_lists/galaxy_slist"][:], np.asarray([0, -1], dtype=np.int32))

    loaded = loader_mod.load(str(outfile), skip_hash_check=True)
    assert int(loaded.nhalos) == 1
    assert int(loaded.ngalaxies) == 1
    assert int(loaded.halos[0].AHF_haloID) == 10
    assert int(loaded.galaxies[0].AHF_haloID) == 10
    assert int(loaded.halos[0].caesar_parent_halo_index) == -1
    assert int(loaded.halos[0].caesar_top_halo_index) == 0
    assert int(loaded.galaxies[0].parent_halo_index) == 0
    assert int(loaded.galaxies[0].caesar_parent_halo_index) == -1
    assert int(loaded.galaxies[0].caesar_top_halo_index) == 0


def test_streaming_export_matches_original_caesar_format(monkeypatch, tmp_path):
    snapshot = tmp_path / "snap.hdf5"
    _write_test_snapshot(snapshot)

    hierarchy = SimpleNamespace(
        parent_of={10: 0},
        children_of={10: []},
    )
    halos_df = pd.DataFrame(
        {
            "hid": np.asarray([10], dtype=np.int64),
            "host_hid": np.asarray([0], dtype=np.int64),
            "npart": np.asarray([4], dtype=np.int64),
            "n_star": np.asarray([1], dtype=np.int64),
        }
    )
    memberships = {
        10: np.asarray([[1002, 0], [2001, 1], [3001, 4], [4001, 5]], dtype=np.int64),
    }

    monkeypatch.setattr(hdf5_mod, "load_ahf_hierarchy", lambda _: hierarchy)
    monkeypatch.setattr(hdf5_mod, "load_ahf_halos_dataframe", lambda _: halos_df)
    monkeypatch.setattr(hdf5_mod, "load_ahf_particle_blocks", lambda *args, **kwargs: memberships)

    state = hdf5_mod.build_direct_state(str(snapshot), "dummy.AHF_particles")
    galaxy_payloads = [
        {
            "AHF_haloID": 10,
            "AHF_parent_haloID": 0,
            "AHF_top_haloID": 10,
            "AHF_depth": 0,
            "AHF_ancestor_haloIDs": np.asarray([], dtype=np.int64),
            "glist": np.asarray([1], dtype=np.int32),
            "slist": np.asarray([0], dtype=np.int32),
            "dmlist": np.asarray([], dtype=np.int32),
            "bhlist": np.asarray([0], dtype=np.int32),
            "dlist": np.asarray([], dtype=np.int32),
        }
    ]

    sim_ref = subhalo_mod._build_direct_calculating_properties_runtime(
        state,
        galaxy_payloads=galaxy_payloads,
        nproc=1,
    )
    for idx, halo in enumerate(sim_ref.halo_list):
        halo._merge_id = idx
    for idx, gal in enumerate(sim_ref.galaxy_list):
        gal._merge_id = idx
    subhalo_mod._compute_group_properties_subset(sim_ref, group_type="halo", groups=list(sim_ref.halo_list))
    subhalo_mod._compute_group_properties_subset(sim_ref, group_type="galaxy", groups=list(sim_ref.galaxy_list))
    halo_states = [subhalo_mod._serialize_group_state(group, id_key="_merge_id") for group in sim_ref.halo_list]
    galaxy_states = [subhalo_mod._serialize_group_state(group, id_key="_merge_id") for group in sim_ref.galaxy_list]

    shard_path = tmp_path / "calculating_properties_rank00000.pkl"
    with shard_path.open("wb") as fh:
        pickle.dump(_property_shard_payload_from_states(halo_states, galaxy_states), fh, protocol=pickle.HIGHEST_PROTOCOL)

    sim_old = subhalo_mod._build_direct_calculating_properties_runtime(
        state,
        galaxy_payloads=galaxy_payloads,
        nproc=1,
    )
    subhalo_mod._compute_group_properties_subset(sim_old, group_type="halo", groups=list(sim_old.halo_list))
    subhalo_mod._compute_group_properties_subset(sim_old, group_type="galaxy", groups=list(sim_old.galaxy_list))
    subhalo_mod._complete_finalization_after_properties_direct(sim_old)
    pipeline_utils_mod.load_global_lists(sim_old)
    setattr(sim_old, "_ahf_subhalo_streaming_save", False)

    ref_out = tmp_path / "ref_caesar.hdf5"
    sim_old.save(str(ref_out))

    stream_out = tmp_path / "stream_caesar.hdf5"
    export_mod.write_catalogue_from_property_shards(
        snapshot_meta=state.snapshot,
        property_results=[_calculating_properties_result_with_summary(shard_path, halo_states, galaxy_states)],
        output_file=str(stream_out),
    )

    _assert_hdf5_equal(ref_out, stream_out)


def test_streaming_export_handles_sorted_multi_halo_lists(monkeypatch, tmp_path):
    snapshot = tmp_path / "snap.hdf5"
    _write_test_snapshot(snapshot)

    hierarchy = SimpleNamespace(
        parent_of={10: 0, 11: 0},
        children_of={10: [], 11: []},
    )
    halos_df = pd.DataFrame(
        {
            "hid": np.asarray([10, 11], dtype=np.int64),
            "host_hid": np.asarray([0, 0], dtype=np.int64),
            "npart": np.asarray([2, 4], dtype=np.int64),
            "n_star": np.asarray([1, 1], dtype=np.int64),
        }
    )
    memberships = {
        10: np.asarray([[1001, 0], [3001, 4]], dtype=np.int64),
        11: np.asarray([[1002, 0], [2001, 1], [2002, 1], [3002, 4]], dtype=np.int64),
    }

    monkeypatch.setattr(hdf5_mod, "load_ahf_hierarchy", lambda _: hierarchy)
    monkeypatch.setattr(hdf5_mod, "load_ahf_halos_dataframe", lambda _: halos_df)
    monkeypatch.setattr(hdf5_mod, "load_ahf_particle_blocks", lambda *args, **kwargs: memberships)

    state = hdf5_mod.build_direct_state(str(snapshot), "dummy.AHF_particles")
    galaxy_payloads = [
        {
            "AHF_haloID": 10,
            "AHF_parent_haloID": 0,
            "AHF_top_haloID": 10,
            "AHF_depth": 0,
            "AHF_ancestor_haloIDs": np.asarray([], dtype=np.int64),
            "glist": np.asarray([0], dtype=np.int32),
            "slist": np.asarray([0], dtype=np.int32),
            "dmlist": np.asarray([], dtype=np.int32),
            "bhlist": np.asarray([], dtype=np.int32),
            "dlist": np.asarray([], dtype=np.int32),
        },
        {
            "AHF_haloID": 11,
            "AHF_parent_haloID": 0,
            "AHF_top_haloID": 11,
            "AHF_depth": 0,
            "AHF_ancestor_haloIDs": np.asarray([], dtype=np.int64),
            "glist": np.asarray([1], dtype=np.int32),
            "slist": np.asarray([1], dtype=np.int32),
            "dmlist": np.asarray([], dtype=np.int32),
            "bhlist": np.asarray([], dtype=np.int32),
            "dlist": np.asarray([], dtype=np.int32),
        },
    ]

    sim_ref = subhalo_mod._build_direct_calculating_properties_runtime(
        state,
        galaxy_payloads=galaxy_payloads,
        nproc=1,
    )
    for idx, halo in enumerate(sim_ref.halo_list):
        halo._merge_id = idx
    for idx, gal in enumerate(sim_ref.galaxy_list):
        gal._merge_id = idx
    subhalo_mod._compute_group_properties_subset(sim_ref, group_type="halo", groups=list(sim_ref.halo_list))
    subhalo_mod._compute_group_properties_subset(sim_ref, group_type="galaxy", groups=list(sim_ref.galaxy_list))
    halo_states = [subhalo_mod._serialize_group_state(group, id_key="_merge_id") for group in sim_ref.halo_list]
    galaxy_states = [subhalo_mod._serialize_group_state(group, id_key="_merge_id") for group in sim_ref.galaxy_list]

    shard_path = tmp_path / "calculating_properties_rank00000.pkl"
    with shard_path.open("wb") as fh:
        pickle.dump(_property_shard_payload_from_states(halo_states, galaxy_states), fh, protocol=pickle.HIGHEST_PROTOCOL)

    sim_old = subhalo_mod._build_direct_calculating_properties_runtime(
        state,
        galaxy_payloads=galaxy_payloads,
        nproc=1,
    )
    subhalo_mod._compute_group_properties_subset(sim_old, group_type="halo", groups=list(sim_old.halo_list))
    subhalo_mod._compute_group_properties_subset(sim_old, group_type="galaxy", groups=list(sim_old.galaxy_list))
    subhalo_mod._complete_finalization_after_properties_direct(sim_old)
    pipeline_utils_mod.load_global_lists(sim_old)
    setattr(sim_old, "_ahf_subhalo_streaming_save", False)

    ref_out = tmp_path / "ref_caesar_multi.hdf5"
    sim_old.save(str(ref_out))

    stream_out = tmp_path / "stream_caesar_multi.hdf5"
    export_mod.write_catalogue_from_property_shards(
        snapshot_meta=state.snapshot,
        property_results=[_calculating_properties_result_with_summary(shard_path, halo_states, galaxy_states)],
        output_file=str(stream_out),
    )

    _assert_hdf5_equal(ref_out, stream_out)


def test_streaming_export_handles_scrambled_ragged_block_owner_order(monkeypatch, tmp_path):
    snapshot = tmp_path / "snap.hdf5"
    _write_test_snapshot(snapshot)

    hierarchy = SimpleNamespace(
        parent_of={10: 0, 11: 0},
        children_of={10: [], 11: []},
    )
    halos_df = pd.DataFrame(
        {
            "hid": np.asarray([10, 11], dtype=np.int64),
            "host_hid": np.asarray([0, 0], dtype=np.int64),
            "npart": np.asarray([2, 4], dtype=np.int64),
            "n_star": np.asarray([1, 1], dtype=np.int64),
        }
    )
    memberships = {
        10: np.asarray([[1001, 0], [3001, 4]], dtype=np.int64),
        11: np.asarray([[1002, 0], [2001, 1], [2002, 1], [3002, 4]], dtype=np.int64),
    }

    monkeypatch.setattr(hdf5_mod, "load_ahf_hierarchy", lambda _: hierarchy)
    monkeypatch.setattr(hdf5_mod, "load_ahf_halos_dataframe", lambda _: halos_df)
    monkeypatch.setattr(hdf5_mod, "load_ahf_particle_blocks", lambda *args, **kwargs: memberships)

    state = hdf5_mod.build_direct_state(str(snapshot), "dummy.AHF_particles")
    galaxy_payloads = [
        {
            "AHF_haloID": 10,
            "AHF_parent_haloID": 0,
            "AHF_top_haloID": 10,
            "AHF_depth": 0,
            "AHF_ancestor_haloIDs": np.asarray([], dtype=np.int64),
            "glist": np.asarray([0], dtype=np.int32),
            "slist": np.asarray([0], dtype=np.int32),
            "dmlist": np.asarray([], dtype=np.int32),
            "bhlist": np.asarray([], dtype=np.int32),
            "dlist": np.asarray([], dtype=np.int32),
        },
        {
            "AHF_haloID": 11,
            "AHF_parent_haloID": 0,
            "AHF_top_haloID": 11,
            "AHF_depth": 0,
            "AHF_ancestor_haloIDs": np.asarray([], dtype=np.int64),
            "glist": np.asarray([1], dtype=np.int32),
            "slist": np.asarray([1], dtype=np.int32),
            "dmlist": np.asarray([], dtype=np.int32),
            "bhlist": np.asarray([], dtype=np.int32),
            "dlist": np.asarray([], dtype=np.int32),
        },
    ]

    sim_ref = subhalo_mod._build_direct_calculating_properties_runtime(
        state,
        galaxy_payloads=galaxy_payloads,
        nproc=1,
    )
    for idx, halo in enumerate(sim_ref.halo_list):
        halo._merge_id = idx
    for idx, gal in enumerate(sim_ref.galaxy_list):
        gal._merge_id = idx
    subhalo_mod._compute_group_properties_subset(sim_ref, group_type="halo", groups=list(sim_ref.halo_list))
    subhalo_mod._compute_group_properties_subset(sim_ref, group_type="galaxy", groups=list(sim_ref.galaxy_list))
    halo_states = [subhalo_mod._serialize_group_state(group, id_key="_merge_id") for group in sim_ref.halo_list]
    galaxy_states = [subhalo_mod._serialize_group_state(group, id_key="_merge_id") for group in sim_ref.galaxy_list]

    payload = _property_shard_payload_from_states(halo_states, galaxy_states)

    def _scramble_block(block):
        owner_ids = np.asarray(block["owner_id"], dtype=np.int64)
        lengths = np.asarray(block["lengths"], dtype=np.int64)
        data = np.asarray(block["data"])
        if owner_ids.size <= 1:
            return dict(block)
        order = np.arange(owner_ids.size - 1, -1, -1, dtype=np.int64)
        starts = np.zeros(owner_ids.size, dtype=np.int64)
        if owner_ids.size > 1:
            starts[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
        ends = starts + lengths
        parts = [np.asarray(data[int(starts[idx]):int(ends[idx])]) for idx in order.tolist()]
        return {
            "owner_id": owner_ids[order],
            "lengths": lengths[order],
            "data": np.concatenate(parts) if parts else np.empty(0, dtype=data.dtype),
            "dtype": block.get("dtype"),
        }

    payload["halo_lists"] = {name: _scramble_block(block) for name, block in payload["halo_lists"].items()}
    payload["galaxy_lists"] = {name: _scramble_block(block) for name, block in payload["galaxy_lists"].items()}

    shard_path = tmp_path / "calculating_properties_rank00000.pkl"
    with shard_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    sim_old = subhalo_mod._build_direct_calculating_properties_runtime(
        state,
        galaxy_payloads=galaxy_payloads,
        nproc=1,
    )
    subhalo_mod._compute_group_properties_subset(sim_old, group_type="halo", groups=list(sim_old.halo_list))
    subhalo_mod._compute_group_properties_subset(sim_old, group_type="galaxy", groups=list(sim_old.galaxy_list))
    subhalo_mod._complete_finalization_after_properties_direct(sim_old)
    pipeline_utils_mod.load_global_lists(sim_old)
    setattr(sim_old, "_ahf_subhalo_streaming_save", False)

    ref_out = tmp_path / "ref_caesar_scrambled.hdf5"
    sim_old.save(str(ref_out))

    stream_out = tmp_path / "stream_caesar_scrambled.hdf5"
    export_mod.write_catalogue_from_property_shards(
        snapshot_meta=state.snapshot,
        property_results=[_calculating_properties_result_with_summary(shard_path, halo_states, galaxy_states)],
        output_file=str(stream_out),
    )

    _assert_hdf5_equal(ref_out, stream_out)
