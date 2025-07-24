import sys
overwrite = '--overwrite' in sys.argv

import os


# Get SLURM array task ID as both int and zero-padded string
slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
slurm_array_task_id_int = int(slurm_array_task_id)
slurm_array_task_id_padded = str(slurm_array_task_id_int).zfill(3)
# Construct the file path using the padded task ID

file_path = f'/disk04/kevin/m100n1024/AHF/particlelists/Simba_M200_snap_{slurm_array_task_id_padded}.h5'

if overwrite:
    print("Overwrite flag detected: overwriting existing file if present.")
else:
    print("No overwrite flag: will skip writing if file exists.")

if overwrite or not os.path.exists(file_path):
    import caesar
    import h5py
    from tqdm import tqdm
    from readgadget import readsnap

    caesar_infile = f'/disk04/wcui/caesar-AHF-Simba-m100n1024/Caesar_m100n1024_{slurm_array_task_id_padded}.hdf5'
    obj = caesar.load(caesar_infile)
    snapshot_file = f'/disk04/rad/sim/m100n1024/s50/snap_m100n1024_{slurm_array_task_id_padded}.hdf5'
    print("Loading STAR Particle IDs from snapshot...")
    particle_ids = readsnap(snapshot_file, 'pid', 'star')  # numpy array

    star = [i.slist for i in obj.galaxies]
    ids = [i.GroupID for i in obj.galaxies]

    print("Writing to file...")
    with h5py.File(file_path,'w') as hdf_file:
        for particles, name in tqdm(zip(star, ids), total=len(ids), desc="Saving to HDF5", unit="galaxy"):
            if particles is not None and len(particles) > 0:
                try:
                    star_pids = particle_ids[particles]
                except Exception as e:
                    print(f"Failed for GroupID {name}: {e}")
                    continue
                hdf_file.create_dataset(str(name), data=star_pids)

