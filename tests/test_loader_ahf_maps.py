from pathlib import Path
import importlib.util
import sys

import numpy as np


module_path = Path(__file__).resolve().parents[1] / "caesar" / "loader.py"
spec = importlib.util.spec_from_file_location("loader_mod", module_path)
loader_mod = importlib.util.module_from_spec(spec)
sys.modules["loader_mod"] = loader_mod
spec.loader.exec_module(loader_mod)


def test_build_ahf_maps_belongs_to_caesar():
    assert hasattr(loader_mod.CAESAR, "_build_ahf_maps")
    assert not hasattr(loader_mod.Cloud, "_build_ahf_maps")

    obj = loader_mod.CAESAR.__new__(loader_mod.CAESAR)
    obj._halo_data = {"AHF_haloID": np.array([11, -1, 42], dtype=np.int64)}
    obj._galaxy_data = {"AHF_haloID": np.array([101, -1, 202], dtype=np.int64)}
    obj.ngalaxies = 3

    obj._build_ahf_maps()

    assert obj._ahf_halo_id_to_index[11] == 0
    assert obj._ahf_halo_id_to_index[42] == 2
    assert 1 not in obj._ahf_halo_index_to_id
    assert obj._ahf_galaxy_id_to_index[101] == 0
    assert obj._ahf_galaxy_id_to_index[202] == 2
    assert obj._ahf_galaxy_ahf_ids == [101, -1, 202]
