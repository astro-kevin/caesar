from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _write_pipeline_snapshot(path: Path) -> None:
    n_dm_per_halo = 64
    dm1_ids = np.arange(2001, 2001 + n_dm_per_halo, dtype=np.int64)
    dm2_ids = np.arange(3001, 3001 + n_dm_per_halo, dtype=np.int64)

    dm1_coords = np.column_stack(
        [
            np.linspace(1000.000, 1000.063, n_dm_per_halo, dtype=np.float32),
            np.full(n_dm_per_halo, 1000.0, dtype=np.float32),
            np.full(n_dm_per_halo, 1000.0, dtype=np.float32),
        ]
    )
    dm2_coords = np.column_stack(
        [
            np.linspace(80000.000, 80000.063, n_dm_per_halo, dtype=np.float32),
            np.full(n_dm_per_halo, 80000.0, dtype=np.float32),
            np.full(n_dm_per_halo, 80000.0, dtype=np.float32),
        ]
    )

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
        gas["Coordinates"] = np.asarray([[1000.000, 1000.000, 1000.000], [80000.000, 80000.000, 80000.000]], dtype=np.float32)
        gas["Velocities"] = np.zeros((2, 3), dtype=np.float32)
        gas["Masses"] = np.asarray([1.0, 1.0], dtype=np.float32)
        gas["Density"] = np.asarray([1.0e-3, 1.2e-3], dtype=np.float32)
        gas["InternalEnergy"] = np.asarray([1.0, 1.0], dtype=np.float32)
        gas["ElectronAbundance"] = np.asarray([1.0, 1.0], dtype=np.float32)
        gas["StarFormationRate"] = np.asarray([1.0, 1.0], dtype=np.float32)
        gas["GrackleHI"] = np.asarray([0.7, 0.6], dtype=np.float32)
        gas["FractionH2"] = np.asarray([0.2, 0.3], dtype=np.float32)
        gas["Metallicity"] = np.asarray([[0.02], [0.03]], dtype=np.float32)

        dm = handle.create_group("PartType1")
        dm["ParticleIDs"] = np.concatenate([dm1_ids, dm2_ids])
        dm["Coordinates"] = np.vstack([dm1_coords, dm2_coords]).astype(np.float32)
        dm["Velocities"] = np.zeros((2 * n_dm_per_halo, 3), dtype=np.float32)
        dm["Masses"] = np.asarray([2.0] * (2 * n_dm_per_halo), dtype=np.float32)
        dm["Potential"] = -np.arange(1, 2 * n_dm_per_halo + 1, dtype=np.float32)

        star = handle.create_group("PartType4")
        star["ParticleIDs"] = np.asarray([4001, 4002], dtype=np.int64)
        star["Coordinates"] = np.asarray([[1000.010, 1000.000, 1000.000], [80000.010, 80000.000, 80000.000]], dtype=np.float32)
        star["Velocities"] = np.zeros((2, 3), dtype=np.float32)
        star["Masses"] = np.asarray([5.0, 5.0], dtype=np.float32)
        star["StellarFormationTime"] = np.asarray([0.25, 0.30], dtype=np.float32)
        star["Metallicity"] = np.asarray([[0.01], [0.015]], dtype=np.float32)

        bh = handle.create_group("PartType5")
        bh["ParticleIDs"] = np.asarray([5001], dtype=np.int64)
        bh["Coordinates"] = np.asarray([[1000.020, 1000.000, 1000.000]], dtype=np.float32)
        bh["Velocities"] = np.zeros((1, 3), dtype=np.float32)
        bh["Masses"] = np.asarray([7.0], dtype=np.float32)
        bh["BH_Mass"] = np.asarray([8.0], dtype=np.float32)
        bh["BH_Mdot"] = np.asarray([0.25], dtype=np.float32)


@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="mpiexec not available")
def test_run_mpi_pipeline_end_to_end(tmp_path):
    snapshot = tmp_path / "snap.hdf5"
    _write_pipeline_snapshot(snapshot)
    ahf_particles = tmp_path / "dummy.AHF_particles"
    ahf_particles.write_text("fixture\n", encoding="utf8")
    output = tmp_path / "out_caesar.hdf5"
    shard_dir = tmp_path / "shards"
    runner = tmp_path / "run_pipeline_fixture.py"

    runner.write_text(
        textwrap.dedent(
            f"""
            from __future__ import annotations

            import argparse
            import sys
            from pathlib import Path
            from types import SimpleNamespace

            import numpy as np
            import pandas as pd

            ROOT = Path({str(ROOT)!r})
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            import caesar.AHF_subhalo_mpi as mpi_mod
            import caesar.ahf_subhalo_hdf5 as hdf5_mod

            hierarchy = SimpleNamespace(
                parent_of={{10: 0, 11: 0}},
                children_of={{10: [], 11: []}},
            )
            halos_df = pd.DataFrame(
                {{
                    "hid": np.asarray([10, 11], dtype=np.int64),
                    "host_hid": np.asarray([0, 0], dtype=np.int64),
                    "npart": np.asarray([66, 67], dtype=np.int64),
                    "n_star": np.asarray([1, 1], dtype=np.int64),
                }}
            )
            memberships = {{
                10: np.asarray(
                    [[1001, 0]]
                    + [[int(v), 1] for v in np.arange(2001, 2001 + 64)]
                    + [[4001, 4], [5001, 5]],
                    dtype=np.int64,
                ),
                11: np.asarray(
                    [[1002, 0]]
                    + [[int(v), 1] for v in np.arange(3001, 3001 + 64)]
                    + [[4002, 4]],
                    dtype=np.int64,
                ),
            }}

            hdf5_mod.load_ahf_hierarchy = lambda _path: hierarchy
            hdf5_mod.load_ahf_halos_dataframe = lambda _path: halos_df
            hdf5_mod.load_ahf_particle_blocks = lambda *args, **kwargs: memberships
            mpi_mod.load_ahf_hierarchy = lambda _path: hierarchy
            mpi_mod.load_ahf_halos_dataframe = lambda _path: halos_df

            parser = argparse.ArgumentParser()
            parser.add_argument("--role", required=True)
            parser.add_argument("--snapshot", required=True)
            parser.add_argument("--ahf", required=True)
            parser.add_argument("--output", required=True)
            parser.add_argument("--shard-dir", required=True)
            args = parser.parse_args()

            mpi_mod.run_mpi(
                snapshot_file=args.snapshot,
                ahf_particles_file=args.ahf,
                output_file=args.output,
                nproc=1,
                role=args.role,
                min_stars=1,
                shard_dir=args.shard_dir,
                phase="pipeline",
            )
            """
        ),
        encoding="utf8",
    )

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    env["CAESAR_AHF_SUBHALO_PROGRESS_STYLE"] = "text"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    cmd = [
        shutil.which("mpiexec") or "mpiexec",
        "-n",
        "1",
        sys.executable,
        str(runner),
        "--role",
        "coordinator",
        "--snapshot",
        str(snapshot),
        "--ahf",
        str(ahf_particles),
        "--output",
        str(output),
        "--shard-dir",
        str(shard_dir),
        ":",
        "-n",
        "1",
        sys.executable,
        str(runner),
        "--role",
        "cpu_worker",
        "--snapshot",
        str(snapshot),
        "--ahf",
        str(ahf_particles),
        "--output",
        str(output),
        "--shard-dir",
        str(shard_dir),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, f"stdout:\\n{result.stdout}\\n\\nstderr:\\n{result.stderr}"
    assert output.is_file(), result.stdout

    with h5py.File(output, "r") as handle:
        assert int(handle.attrs["nhalos"]) == 2
        assert int(handle.attrs["ngalaxies"]) == 2
        sim_attrs = handle["simulation_attributes"].attrs
        assert float(sim_attrs["hubble_constant"]) == 0.5
        assert int(sim_attrs["effective_resolution"]) > 0
        assert bool(sim_attrs["baryons_present"])
        assert not bool(sim_attrs["unbind_halos"])
        assert "mean_interparticle_separation" in sim_attrs
        assert "halo_data/AHF_haloID" in handle
        assert "galaxy_data/AHF_haloID" in handle
        assert "galaxy_data/caesar_parent_halo_index" in handle
        assert "galaxy_data/caesar_top_halo_index" in handle
        assert "global_lists/halo_bhlist" in handle
        assert "global_lists/galaxy_bhlist" in handle
