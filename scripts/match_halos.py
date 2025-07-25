"""Utility script to match AHF halos to CAESAR galaxies."""

import os
import pandas as pd

import caesar
from caesar.halo_matching import (
    get_AHF_file,
    load_snapshot_to_namedata,
    read_file_to_structure,
    find_best_matches,
    summarize_matches,
    match_subhalos_to_galaxies,
)


def match_halos(sim, snapshot_file: str, ahf_file: str, csv_out: str) -> None:
    """Run halo matching and update ``sim`` with subhalo information."""

    ahf_data = read_file_to_structure(ahf_file)
    caesar_data = load_snapshot_to_namedata(snapshot_file, sim.galaxies)
    matches = find_best_matches(caesar_data, ahf_data)
    summarize_matches(matches, caesar_data, ahf_data)

    df = pd.DataFrame([{"CaesarID": m[0], "AHFID": m[1]} for m in matches])
    df.sort_values("CaesarID", inplace=True)
    df.to_csv(csv_out, index=False)

    match_subhalos_to_galaxies(
        sim, ahf_file=ahf_file, snapshot_file=snapshot_file
    )


def main() -> None:
    slurm_id = os.getenv("SLURM_ARRAY_TASK_ID")
    snap = int(slurm_id) if slurm_id is not None else 0

    directory_path_ahf = os.path.join("/disk04", "wcui", "AHF-Simba-m100n1024")
    directory_path_caesar = os.path.join("/disk04", "wcui", "caesar-AHF-Simba-m100n1024")
    directory_path_snap = os.path.join("/disk04", "rad", "sim", "m100n1024", "s50")
    directory_path_output = os.path.join("/disk04", "kevin", "m100n1024", "AHF", "caesar_matches")

    ahf_file = get_AHF_file(directory_path_ahf, snap)
    caesar_file = os.path.join(directory_path_caesar, f"Caesar_m100n1024_{snap:03d}.hdf5")
    snapshot_file = os.path.join(directory_path_snap, f"snap_m100n1024_{snap:03d}.hdf5")
    sim = caesar.load(caesar_file)
    csv_out = os.path.join(directory_path_output, f"Simba_M200_snap_{snap:03d}.csv")

    match_halos(sim, snapshot_file, ahf_file, csv_out)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
