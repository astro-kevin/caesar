import numpy as np

from typing import List

from yt.funcs import mylog


def _assert_galaxy_host_consistency(obj):
    """Ensure parent_halo_index and halo.galaxy_index_list agree for all halos.

    This is a sanity check that each top-level halo's membership is
    internally consistent before assigning centrals.
    """
    if not getattr(obj, '_has_galaxies', False):
        return

    nhalo = len(obj.halos)
    if nhalo == 0:
        return

    # 1) Collect parent_halo_index for all galaxies as a NumPy array
    parent = np.empty(len(obj.galaxies), dtype=np.int64)
    for gi, gal in enumerate(obj.galaxies):
        hid = getattr(gal, 'parent_halo_index', -1)
        try:
            parent[gi] = int(hid)
        except Exception:
            parent[gi] = -1

    # 2) For each halo, compare expected membership from parent_halo_index
    #    against the explicit galaxy_index_list
    for hid in range(nhalo):
        expected = np.where(parent == hid)[0]
        got = np.asarray(getattr(obj.halos[hid], 'galaxy_index_list', []), dtype=np.int64)

        expected_sorted = expected  # np.where returns sorted indices
        got_sorted = np.sort(got)

        if expected_sorted.shape != got_sorted.shape or not np.array_equal(expected_sorted, got_sorted):
            raise AssertionError(
                f"Galaxy/halo host mismatch for halo {hid}: "
                f"from parent_halo_index={expected_sorted.tolist()} vs galaxy_index_list={got_sorted.tolist()}"
            )

def assign_galaxies_to_halos(obj):
    """Assign galaxies to halos.

    This function compares galaxy_glist + galaxy_slist with halo_glist
    + halo_slist to determine which halo the majority of particles
    within each galaxy lie.  Finally we assign the .galaxies list to
    each halo.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Object containing the galaxies and halos lists.    

    """
    if not obj._has_galaxies:
        return

    mylog.info('Assigning galaxies to halos')

    override_hosts = None
    override_hosts_by_galaxy = [
        getattr(galaxy, '_ahf_host_halo_index', None) for galaxy in obj.galaxies
    ]
    if any(host is not None for host in override_hosts_by_galaxy):
        override_hosts = override_hosts_by_galaxy
    else:
        override_hosts = getattr(obj, '_ahf_galaxy_hosts', None)

    if override_hosts is not None and len(override_hosts) == obj.ngalaxies:
        for halo in obj.halos:
            halo.galaxy_index_list = []

        pending: List[int] = []
        for gi, galaxy in enumerate(obj.galaxies):
            host_index = override_hosts[gi]
            if host_index is not None and 0 <= host_index < len(obj.halos):
                galaxy.parent_halo_index = int(host_index)
                obj.halos[int(host_index)].galaxy_index_list.append(gi)
            else:
                galaxy.parent_halo_index = -1
                pending.append(gi)

        if pending:
            # Fallback to particle-based assignment only for the unresolved galaxies.
            h_glist = obj.global_particle_lists.halo_glist
            h_slist = obj.global_particle_lists.halo_slist
            for gi in pending:
                galaxy = obj.galaxies[gi]
                glist = h_glist[galaxy.glist]
                slist = h_slist[galaxy.slist]

                combined = np.hstack((glist, slist))
                valid = np.where(combined > -1)[0]
                combined = combined[valid]

                if len(combined) > 0:
                    parent_index = np.bincount(combined).argmax()
                    galaxy.parent_halo_index = parent_index
                    obj.halos[parent_index].galaxy_index_list.append(gi)

        obj._ahf_galaxy_hosts = []
        for galaxy in obj.galaxies:
            host_index = int(getattr(galaxy, 'parent_halo_index', -1))
            galaxy._ahf_host_halo_index = host_index
            obj._ahf_galaxy_hosts.append(host_index)

        obj._ahf_galaxy_ahf_ids = [
            int(getattr(galaxy, 'AHF_haloID', -1)) for galaxy in obj.galaxies
        ]
        obj._ahf_galaxy_top_ahf_ids = [
            int(getattr(galaxy, 'AHF_top_haloID', -1)) for galaxy in obj.galaxies
        ]
        try:
            from caesar.halo_matching import _update_ahf_galaxy_maps

            _update_ahf_galaxy_maps(obj, obj._ahf_galaxy_ahf_ids)
        except Exception:
            pass

        return

    h_glist = obj.global_particle_lists.halo_glist
    h_slist = obj.global_particle_lists.halo_slist
    for galaxy in obj.galaxies:
        glist = h_glist[galaxy.glist]
        slist = h_slist[galaxy.slist]

        combined = np.hstack((glist,slist))
        valid = np.where(combined > -1)[0]
        combined = combined[valid]

        galaxy.parent_halo_index = -1
        if len(combined) > 0:
            galaxy.parent_halo_index = np.bincount(combined).argmax()

    for halo in obj.halos:
        halo.galaxy_index_list = []

    for i in range(0,obj.ngalaxies):
        galaxy = obj.galaxies[i]
        if galaxy.parent_halo_index > -1:
            obj.halos[galaxy.parent_halo_index].galaxy_index_list.append(i)

    # assign_galaxies_to_halos is responsible for keeping
    # parent_halo_index and galaxy_index_list in sync; consistency is
    # enforced later by _assert_galaxy_host_consistency before
    # centrals are chosen.





def assign_clouds_to_galaxies(obj):
    """Assign clouds to galaxies.

    This function compares cloud_glist with galaxy_glist to determine
    which galaxy the majority of particles within each cloud lies.
    Finally we assign the .clouds list to each galaxy.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Object containing the galaxies and halos lists.

    """

    #NOTES FROM BOBBY: USE HASATTR() INSTEAD OF THIS _HAS_CLOUDS
    if not obj._has_clouds:
        return

    mylog.info('Assigning clouds to galaxies')
    

    g_glist = obj.global_particle_lists.galaxy_glist

    for cloud in obj.clouds:
        glist = g_glist[cloud.glist]

        combined = glist
        valid = np.where(combined > -1)[0]
        combined = combined[valid]
        
        cloud.parent_galaxy_index = -1
        if len(combined) > 0:
            cloud.parent_galaxy_index = np.bincount(combined).argmax()

    for galaxy in obj.galaxies:
        galaxy.cloud_index_list = []

    for i in range(0,obj.nclouds):
        cloud = obj.clouds[i]
        if cloud.parent_galaxy_index > -1:
            obj.galaxies[cloud.parent_galaxy_index].cloud_index_list.append(i)

            
def assign_central_galaxies(obj,central_mass_definition='stellar'):
    """Assign central galaxies.

    Iterate through halos and consider the most massive galaxy within
    a central and all other satellites.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Object containing the galaxies to assign centrals.  Halos 
        must already be assigned via `assign_galaxies_to_halos`.

    """
    if not obj._has_galaxies:
        return

    mylog.info('Assigning central galaxies')

    # Before assigning centrals, enforce that per-halo galaxy_index_list
    # matches parent_halo_index. Any mismatch here indicates a real
    # bug in host bookkeeping that we should not paper over.
    _assert_galaxy_host_consistency(obj)

    # Clear any previous central flags to avoid stale centrals after remapping
    try:
        for g in obj.galaxies:
            g.central = False
    except Exception:
        pass

    obj.central_galaxies   = []
    obj.satellite_galaxies = []

    # For each halo, choose the most massive galaxy among those already
    # assigned to the halo via galaxy_index_list. We rely on
    # assign_galaxies_to_halos (with optional assertions) to keep
    # galaxy_index_list and parent_halo_index consistent.
    for halo in obj.halos:
        if not hasattr(halo, 'galaxy_index_list') or len(halo.galaxy_index_list) == 0:
            continue

        masses = []
        for gi in halo.galaxy_index_list:
            m = obj.galaxies[gi].masses[central_mass_definition]
            # Convert YTQuantity or numpy scalar to float robustly
            try:
                val = float(getattr(m, 'd', m))
            except Exception:
                try:
                    val = float(getattr(m, 'value', m))
                except Exception:
                    val = 0.0
            masses.append(val)

        if not masses:
            continue

        central_local = int(np.argmax(np.asarray(masses)))
        central_gi = int(halo.galaxy_index_list[central_local])
        central_gal = obj.galaxies[central_gi]
        central_gal.central = True
        obj.central_galaxies.append(central_gal)

        # All other galaxies in this halo are satellites
        for gi in halo.galaxy_index_list:
            if gi == central_gi:
                continue
            sat_gal = obj.galaxies[gi]
            obj.satellite_galaxies.append(sat_gal)
