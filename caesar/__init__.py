# Delay importing heavy modules that require ``yt``
try:
    from caesar.loader import load
    from caesar.main import CAESAR
    from caesar.driver import drive
except Exception:  # pragma: no cover - optional dependency
    load = None
    CAESAR = None
    drive = None
#from caesar.group_funcs import get_periodic_r

from caesar.old_loader import load as old_load
from caesar.halo_matching import (
    ParticleMembership,
    read_file_to_structure,
    load_hdf5_to_namedata,
    galaxies_to_namedata,
    load_snapshot_to_namedata,
    find_best_matches,
    get_caesar_file,
    get_AHF_file,
    summarize_matches,
    read_single_halo_from_file,
    match_subhalos_to_galaxies,
)

def quick_load(*args, **kwargs):
    import warnings
    warnings.warn('The quick-loader is now the default behavior. The -q and --quick flags will be removed soon.', stacklevel=2)
    return load(*args, **kwargs)

