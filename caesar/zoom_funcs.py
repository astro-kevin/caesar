import numpy as np
from yt.funcs import mylog
import pdb
import math

def write_IC_mask(group, ic_ds, filename, search_factor, radius_type='total',print_extents=True):
    """Write MUSIC initial condition mask to disk.

    Parameters
    ----------
    group : :class:`group.Group`
        Group we are querying.
    ic_ds : yt dataset
        The initial condition dataset via ``yt.load()``.
    filename : str
        The filename of which to write the mask to.  If a full path is
        not supplied then it will be written in the current directory.
    search_factor : float, optional
        How far from the center to select DM particles
        (defaults to 2.5)
    print_extents : bool, optional
        Print MUSIC extents for cuboid after mask creation

    Examples
    --------
    >>> import yt
    >>> import caesar
    >>>
    >>> snap = 'my_snapshot.hdf5'
    >>> ic   = 'IC.dat'
    >>>
    >>> ds    = yt.load(snap)
    >>> ic_ds = yt.load(ic)
    >>>
    >>> obj = caesar.load('caesar_my_snapshot.hdf5', ds)
    >>> obj.galaxies[0].write_IC_mask(ic_ds, 'mymask.txt')

    """

    ic_dmpos = get_IC_pos(group, ic_ds,radius_type, search_factor=search_factor,
                          return_mask=True)

    mylog.info('Writing IC mask to %s' % filename)
    f = open(filename, 'w')
    for i in range(0, len(ic_dmpos)):
        f.write('%e %e %e\n' % (ic_dmpos[i,0], ic_dmpos[i,1], ic_dmpos[i,2]))
    f.close()

    def get_extents(dim):
        vals = np.sort(ic_dmpos[:,dim])
        break_point = -1.0
        for i in range(1,len(vals)):
            ddim = vals[i] - vals[i-1]
            if ddim > 0.1:
                break_point = vals[i-1] + (ddim/2.0)
                break
        if break_point != -1:
            l_indexes = np.where(ic_dmpos[:,dim] < break_point)[0]
            ic_dmpos[l_indexes, dim] += 1.0

        dmin = np.min(ic_dmpos[:,dim])
        dmax = np.max(ic_dmpos[:,dim])
        dext = dmax - dmin

        return dmin,dext

    if print_extents:
        xcen, xext = get_extents(0)
        ycen, yext = get_extents(1)
        zcen, zext = get_extents(2)

        mylog.info('MUSIC cuboid settings:')
        mylog.info('ref_center = %0.3f,%0.3f,%0.3f' % (xcen,ycen,zcen))
        mylog.info('ref_extent = %0.3f,%0.3f,%0.3f' % (xext,yext,zext))

        if xext >= 0.5 or yext >= 0.5 or zext >= 0.5:
            mylog.warning('REGION EXTENDS MORE THAN HALF OF YOUR VOLUME')


def get_IC_pos(group, ic_ds, radius_type,search_factor=2.5, return_mask=False):
    """Get the initial dark matter positions of a ``CAESAR`` halo.

    If called on a galaxy, it will return the IC DM positions of the
    parent halo.

    Parameters
    ----------
    group : :class:`group.Group`
        Group we are querying.
    ic_ds : yt dataset
        The initial condition dataset via ``yt.load()``
    search_factor : float, optional
        How far from the center to select DM particles (defaults to 2.5).
    return_mask : bool, optional
        Return initial condition positions from 0-->1 rather than raw
        data.  Useful for writing a MUSIC mask file.

    Returns
    -------
    ic_dmpos : np.ndarray
        DM positions of this object in the initial condition file.

    """
    from caesar.property_manager import ptype_aliases, get_property, DatasetType
    # from caesar.periodic_kdtree import PeriodicCKDTree
    from scipy.spatial import cKDTree

    ic_ds_type = ic_ds.__class__.__name__
    if ic_ds_type not in ptype_aliases:
        raise NotImplementedError('%s not yet supported' % ic_ds_type)
    
    if math.isclose(group.obj.yt_dataset.domain_width[0].d,ic_ds.domain_width[0].d,abs_tol=1.e-6) == False:
        raise Exception('IC and SNAP boxes do not match! (%f vs %f)' %
                        (ic_ds.domain_width[0].d,
                         group.obj.yt_dataset.domain_width[0].d))


    
    if math.isclose(ic_ds.length_unit.value,group.obj.yt_dataset.length_unit.value,abs_tol=1.e-6) == False:
        raise Exception('LENGTH UNIT MISMATCH! '\
                        'This may arise from loading the snap/IC '\
                        'incorrectly and WILL cause problems with '\
                        'the matching process. (%s vs %s)' %
                        (str(ic_ds.length_unit), str(group.obj.yt_dataset.length_unit)))

    if group.obj_type == 'halo':
        obj = group
    elif group.obj_type == 'galaxy':
        if group.halo is None:
            mylog.warning('Galaxy %d has no halo!' % group.GroupID)
            return
        obj = group.halo
    
    search_params = dict(
        pos = obj.pos.in_units('code_length').d,
        r   = obj.radii[radius_type].in_units('code_length').d * search_factor,
    )

    box    = ic_ds.domain_width[0].d
    box = np.array([box,box,box])

    dmpids = get_property(obj.obj, 'pid', 'dm').d
    dmpos  = get_property(obj.obj, 'pos', 'dm').d
    for i in range(3):
        dmpos[dmpos[:,i]>box[i], i] -= box[i]
        dmpos[dmpos[:,i]<0, i] += box[i]

    dm_TREE = cKDTree(dmpos, boxsize=box)

    valid = dm_TREE.query_ball_point(search_params['pos'], search_params['r'])
    search_params['ids'] = dmpids[valid]

    ic_ds_type = DatasetType(ic_ds)
    ic_dmpos   = ic_ds_type.get_property('dm', 'pos').d
    ic_dmpids  = ic_ds_type.get_property('dm', 'pid').d

    matches  = np.in1d(ic_dmpids, search_params['ids'], assume_unique=True)
    nmatches = len(np.where(matches)[0])
    nvalid   = len(valid)
    if nmatches != nvalid:
        raise Exception('Could not match all particles! '\
                        'Only %0.2f%% particles matched.' %
                        (float(nmatches)/float(nvalid) * 100.0))

    mylog.info('MATCHED %d particles from %s %d in %s' %
               (nmatches, obj.obj_type, obj.GroupID, ic_ds.basename))
    mylog.info('Returning %0.2f%% of the total DM from the sim' %
               (float(nmatches)/float(len(ic_dmpids)) * 100.0))

    matched_pos = ic_dmpos[matches]



    if return_mask:
        matched_pos /= box

    return matched_pos


def construct_lowres_tree(obj, lowres):
    """Construct a periodic KDTree for low-resolution particles.

    Parameters
    ----------
    group : :class:`group.Group`
        Group we are querying.
    lowres : list
        Particle types to be considered low-resolution.  Typically
        [2,3,5]

    Notes
    -----
    Assigns the dict ``_lowres`` to the :class:`main.CAESAR` object.

    """
    # obj = group.obj
    if hasattr(obj, '_lowres') and obj._lowres['ptypes'] == lowres:
        return
    mylog.info('Gathering low-res particles and constructing tree')
    from caesar.property_manager import get_property

    lr_pos  = np.empty((0,3))
    lr_mass = np.empty(0)

    pos_unit  = obj.halos[0].pos.units
    mass_unit = obj.halos[0].masses['total'].units

    for p in lowres:  # all low res particles to only build tree once
        if (p==2) or (p=='dm2'):
            cur_pos  = obj.data_manager.pos[obj.data_manager.dm2list]
            cur_mass = obj.data_manager.mass[obj.data_manager.dm2list]
        elif (p==3) or (p=='dm3'):
            cur_pos  = obj.data_manager.pos[obj.data_manager.dm3list]
            cur_mass = obj.data_manager.mass[obj.data_manager.dm3list]
        else: 
            mylog.error('It seems you are setting PartType%d as low resolution praticles, if so, please modify the code to proplerly load it!'%p)
            
        # ptype = 'PartType%d' % p
        # if ptype in obj.yt_dataset.particle_fields_by_type:
        #     cur_pos  = obj._ds_type.dd[ptype, 'particle_position'].to(pos_unit)
        #     cur_mass = obj._ds_type.dd[ptype, 'particle_mass'].to(mass_unit)

        lr_pos  = np.append(lr_pos,  cur_pos, axis=0)
        lr_mass = np.append(lr_mass, cur_mass, axis=0)

    # from caesar.periodic_kdtree import PeriodicCKDTree
    from scipy.spatial import cKDTree
    box    = obj.simulation.boxsize.to(pos_unit)
    box = np.array([box,box,box])
    for i in range(3):
        lr_pos[lr_pos[:,i]>box[i], i] -= box[i]
        lr_pos[lr_pos[:,i]<0, i] += box[i]

    obj._lowres = dict(
        TREE   = cKDTree(lr_pos, boxsize=box),
        MASS   = lr_mass,
        ptypes = lowres
    )


def all_object_contam_check(obj):
    # if obj._ds_type.ds_type != 'GizmoDataset' and obj._ds_type.ds_type != 'GadgetHDF5Dataset':
    #     return
    if not 'lowres' in obj._kwargs or obj._kwargs['lowres'] is None:
        return
    if not 'nproc' in obj._kwargs or obj._kwargs['nproc'] is None:
        nproc = 1
    else:
        nproc = obj._kwargs['nproc']

    lowres = obj._kwargs['lowres']
    if not isinstance(lowres, list):
        raise Exception('lowres must be a list!')

    construct_lowres_tree(obj, lowres)
    mylog.info('Checking all objects for contamination.  Lowres Types: %s' % lowres)
    if hasattr(obj, 'halos'):
        for h in obj.halos:
            h.contamination_check(obj._lowres, nproc=nproc, printer=False)
    if hasattr(obj, 'galaxies'):
        for g in obj.galaxies:
            if g.halo is None:
                g.contamination = -1
            else:
                g.contamination_check(obj._lowres, nproc=nproc, printer=False)
                # g.contamination = g.halo.contamination
