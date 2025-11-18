import numpy as np

from yt.funcs import mylog

from caesar.fof6d import fof6d
from caesar.fubar import get_mean_interparticle_separation
from caesar.pipeline_utils import reset_global_particle_IDs, load_global_lists


def run(obj):
    """FOF/SNAP member_search pipeline (haloid='fof' or 'snap')."""

    # set up number of processors
    obj.nproc = 1  # defaults to single core
    if 'nproc' in obj._kwargs:
        obj.nproc = int(obj._kwargs['nproc'])
    if obj.nproc != 1:
        import joblib

        if obj.nproc < 0:
            obj.nproc += joblib.cpu_count() + 1
        if obj.nproc == 0:
            obj.nproc = joblib.cpu_count()
    mylog.info('member_search() running on %d cores' % obj.nproc)
    obj.load_haloid = False
    if 'haloid' in obj._kwargs and 'snap' in obj._kwargs['haloid']:
        obj.load_haloid = True

    # Process halos
    halos = fof6d(obj, 'halo')  # instantiate a fof6d object
    halos.MIS = get_mean_interparticle_separation(obj).d  # also computes omega_baryon and related quantities
    halos.load_haloid()
    halos.obj.data_manager._member_search_init(
        select=halos.haloid
    )  # load particle info, but only those selected to be in a halo
    if not halos.plist_init():  # not enough halo particles found, nothing to do!
        return
    halos.load_lists()  # create halos, load particle index lists for halos
    if len(halos.obj.halo_list) == 0:  # no valid halos found
        mylog.warning('No valid halos found! Aborting member search')
        return

    from caesar.group import get_group_properties

    get_group_properties(halos, halos.obj.halo_list)  # compute halo properties
    if not obj.simulation.baryons_present:  # if no baryons, we're done
        return

    # Find galaxies, or load galaxy membership info
    if 'galid' in obj._kwargs and 'rockstar' in obj._kwargs['galid']:
        halos.load_rockstar_ids('bottomid')
        halos.obj.rockstar_flag = 2
    else:  # if unspecified use fof6d for galaxies/clouds
        fof6d_flag = True
        if 'fof6d_file' in obj._kwargs and obj._kwargs['fof6d_file'] is not None:
            fof6d_flag = halos.load_fof6dfile()  # load galaxy ID's from fof6d_file
        if fof6d_flag:
            halos.run_fof6d('galaxy')  # run fof6d on halos to find galaxies
            halos.save_fof6dfile()  # save fof6d info

    # Process galaxies
    galaxies = fof6d(obj, 'galaxy')  # instantiate a fof6d object
    galaxies.plist_init(parent=halos)  # get particle list for computing galaxy properties
    if galaxies.nparttot == 0:  # plist_init didn't find any particles in a galaxy
        mylog.warning('Not enough eligible galaxy particles found!')
        return
    galaxies.load_lists(parent=halos)  # create galaxy_list, load particle index lists for galaxies
    get_group_properties(galaxies, galaxies.obj.galaxy_list)  # compute galaxy properties
    if ('fsps_bands' in obj._kwargs) and obj._kwargs['fsps_bands'] is not None:
        from caesar.pyloser.pyloser import photometry

        galphot = photometry(obj, galaxies.obj.galaxy_list)
        galphot.run_pyloser()

    # Find and process clouds
    if ('fofclouds' in obj._kwargs) and obj._kwargs['fofclouds']:
        galaxies.run_fof6d('cloud')  # run fof6d to find cloudid's
        galaxies.load_lists('cloud')  # load particle index lists for galaxies
        clouds = fof6d(obj, 'cloud')  # instantiate a fof6d object
        clouds.plist_init(parent=galaxies)  # initialize comptutation of cloud properties
        if clouds.nparttot == 0:
            return  # plist_init didn't find enough particles to group
        galaxies.load_lists('cloud')
        get_group_properties(clouds, clouds.obj.cloud_list)  # compute cloud properties

    # reset particle lists to have original snapshot ID's; must do this after all group processing is finished
    reset_global_particle_IDs(obj)
    # load global lists
    load_global_lists(obj)

    return

