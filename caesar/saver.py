import os
import tempfile
import h5py
import numpy as np
import pdb
import six
from yt.units.yt_array import YTQuantity, YTArray
from yt.funcs import mylog
blacklist = [
    'G', 'initial_mass',
    'valid', 'vel_conversion',
    'unbound_particles', '_units',
    'unit_registry_json',
    'unbound_indexes',
    'lists','dicts'
]

######################################################################

def _write_dataset(key, data, hd):
    hd.create_dataset(key, data=data, compression=1)

def check_and_write_dataset(obj, key, hd):
    """General function for writing an HDF5 dataset.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object to save.
    key : str
        Name of dataset to write.
    hd : h5py.Group
        Open HDF5 group.

    """
    if not hasattr(obj, key): return
    if isinstance(getattr(obj, key), int): return
    _write_dataset(key, getattr(obj, key), hd)

######################################################################

def serialize_list(obj_list, key, hd):
    """Function that serializes a index list (glist/etc) for objects.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object.
    key : str
        Name of the index list.
    hd : h5py.Group
        Open HDF5 group.

    """
    if key in blacklist: return
    if not obj_list:
        return
    if not hasattr(obj_list[0], key): return
    data = _get_serialized_list(obj_list, key)
    _write_dataset(key, data, hd)

def _get_serialized_list(obj_list, key):
    tmp, index = [], 0
    for i in obj_list:
        current_list = getattr(i, key)
        n = len(current_list)
        tmp.extend(current_list)

        setattr(i, '%s_start' % key, index)
        index += n
        setattr(i, '%s_end'   % key, index)
    return tmp

######################################################################

def serialize_attributes(obj_list, hd, hd_dicts):
    """Function that goes through a list full of halos/galaxies/clouds and
    serializes their attributes.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object.
    hd : h5py.Group
        Open HDF5 group for lists.
    hd_dicts : h5py.Group
        Open HDF5 group for dictionaries.

    """
    if not obj_list:
        return

    for k,v in six.iteritems(obj_list[0].__dict__):
        if k in blacklist: continue

        if isinstance(v, dict):
            _write_dict(obj_list, k, v, hd_dicts)
        else:
            _write_attrib(obj_list, k, v, hd)

def _write_attrib(obj_list, k, v, hd):
    # Skip optional particle-type fields when the simulation did not load that species.
    owner = obj_list[0] if obj_list else None
    sim = getattr(owner, 'obj', None)
    dmgr = getattr(sim, 'data_manager', None) if sim is not None else None
    ptypes = getattr(dmgr, 'ptypes', []) if dmgr is not None else []
    if (k in ('dm2list', 'ndm2') and 'dm2' not in ptypes) or (
        k in ('dm3list', 'ndm3') and 'dm3' not in ptypes
    ):
        return

    unit = False
    if isinstance(v, YTQuantity):
        data = [getattr(i,k).d for i in obj_list]
        unit = True
    elif isinstance(v, YTArray):
        if np.shape(v)[0] == 3:
            data = np.vstack([getattr(i,k).d for i in obj_list])
        else:
            data = [getattr(i,k).d for i in obj_list]
        unit = True
    elif isinstance(v, np.ndarray) and np.shape(v)[0] == 3 and 'list' not in k:
        try:
            data = np.vstack([getattr(i,k) for i in obj_list])
        except Exception:
            mylog.warning('Saver unable to stack: %s %s %s',k,v,np.shape(v))
            return
    elif isinstance(v, (int, float, bool, np.number)):
        data = [getattr(i,k) for i in obj_list]
    else:
        return

    _write_dataset(k, data, hd)
    if unit:
        hd[k].attrs.create('unit', str(v.units).encode('utf8'))
            
def _write_dict(obj_list, k, v, hd):
    for kk,vv in six.iteritems(v):
        unit = False        
        if isinstance(vv, (YTQuantity, YTArray)):
            data = np.array([getattr(i,k)[kk].d for i in obj_list])
            unit = True
        else:
            data = np.array([getattr(i,k)[kk] for i in obj_list])            

        _write_dataset('%s.%s' % (k,kk), data, hd)
        if unit:
            hd['%s.%s' % (k,kk)].attrs.create('unit', str(vv.units).encode('utf8'))        

######################################################################

def _ensure_dm_bookkeeping(obj):
    """Guarantee halo DM bookkeeping exists when low-resolution species are enabled."""
    dm_ptypes = getattr(getattr(obj, 'data_manager', None), 'ptypes', [])
    if not dm_ptypes:
        return

    dataset = getattr(obj, 'yt_dataset', None)
    units = getattr(obj, 'units', {}) or {}
    mass_unit = units.get('mass', 'Msun')

    def _zero_mass():
        if dataset is not None:
            try:
                return dataset.quan(0.0, mass_unit)
            except Exception:
                pass
        return 0.0

    halos = getattr(obj, 'halos', [])
    if not halos:
        return

    for halo in halos:
        if 'dm2' in dm_ptypes:
            dm2list = getattr(halo, 'dm2list', None)
            if dm2list is None:
                dm2list = np.empty(0, dtype=np.int64)
            else:
                dm2list = np.asarray(dm2list, dtype=np.int64)
            halo.dm2list = dm2list
            halo.ndm2 = int(dm2list.size)
            if isinstance(getattr(halo, 'masses', None), dict):
                halo.masses.setdefault('dm2', _zero_mass())
        if 'dm3' in dm_ptypes:
            dm3list = getattr(halo, 'dm3list', None)
            if dm3list is None:
                dm3list = np.empty(0, dtype=np.int64)
            else:
                dm3list = np.asarray(dm3list, dtype=np.int64)
            halo.dm3list = dm3list
            halo.ndm3 = int(dm3list.size)
            if isinstance(getattr(halo, 'masses', None), dict):
                halo.masses.setdefault('dm3', _zero_mass())

######################################################################

def serialize_global_attribs(obj, hd):
    """Function that goes through a caesar object and saves general 
    attributes.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object.
    hd : h5py.File
        Open HDF5 dataset.

    """
    units = {}
    for k,v in six.iteritems(obj.__dict__):
        if k in blacklist: continue

        if isinstance(v, (YTQuantity, YTArray)):
            hd.attrs.create(k, v.d)
            units[k] = v.units
        elif isinstance(v, str):
            hd.attrs.create(k, v.encode('utf8'))
        elif isinstance(v, (int, float, bool, np.number)):
            hd.attrs.create(k, v)
        #else:
        #    print(k,type(v))

    if len(units) > 0:
        uhd = hd.create_group('global_attribute_units')
        for k,v in six.iteritems(units):
            uhd.attrs.create(k, str(v).encode('utf8'))
            
######################################################################

def _stream_group_id_dataset(
    *,
    hd,
    dataset_name: str,
    size: int,
    groups,
    list_attr: str,
    temp_dir: str,
):
    size = int(size)
    if size <= 0:
        return

    fd, tmp_path = tempfile.mkstemp(
        prefix=f"caesar_{dataset_name}_",
        suffix=".tmp",
        dir=temp_dir,
    )
    os.close(fd)
    try:
        arr = np.memmap(tmp_path, dtype=np.int32, mode="w+", shape=(size,))
        arr[:] = -1
        for group in groups:
            if not hasattr(group, list_attr):
                continue
            idx = np.asarray(getattr(group, list_attr), dtype=np.int64)
            if idx.size == 0:
                continue
            arr[idx] = int(getattr(group, "GroupID", -1))
        arr.flush()
        hd.create_dataset(dataset_name, data=arr, compression=1)
        del arr
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _write_streaming_global_lists(obj, hd, *, filename: str):
    from yt.funcs import mylog

    temp_dir = os.environ.get("CAESAR_STREAM_SAVE_TMPDIR")
    if not temp_dir:
        temp_dir = os.path.dirname(os.path.abspath(filename)) or "."
    os.makedirs(temp_dir, exist_ok=True)

    groups = [
        ("halo", getattr(obj, "halos", []), "dmlist", int(getattr(obj.simulation, "ndm", 0))),
        ("halo", getattr(obj, "halos", []), "glist", int(getattr(obj.simulation, "ngas", 0))),
        ("halo", getattr(obj, "halos", []), "slist", int(getattr(obj.simulation, "nstar", 0))),
        ("galaxy", getattr(obj, "galaxies", []), "glist", int(getattr(obj.simulation, "ngas", 0))),
        ("galaxy", getattr(obj, "galaxies", []), "slist", int(getattr(obj.simulation, "nstar", 0))),
        ("cloud", getattr(obj, "clouds", []), "glist", int(getattr(obj.simulation, "ngas", 0))),
    ]

    if getattr(getattr(obj, "data_manager", None), "blackholes", False):
        groups.extend(
            [
                ("halo", getattr(obj, "halos", []), "bhlist", int(getattr(obj.simulation, "nbh", 0))),
                ("galaxy", getattr(obj, "galaxies", []), "bhlist", int(getattr(obj.simulation, "nbh", 0))),
            ]
        )
    if getattr(getattr(obj, "data_manager", None), "dust", False):
        groups.extend(
            [
                ("halo", getattr(obj, "halos", []), "dlist", int(getattr(obj.simulation, "ndust", 0))),
                ("galaxy", getattr(obj, "galaxies", []), "dlist", int(getattr(obj.simulation, "ndust", 0))),
            ]
        )

    for group_type, group_list, list_attr, size in groups:
        dataset_name = f"{group_type}_{list_attr}"
        if not group_list or size <= 0:
            continue
        mylog.info("Streaming %s to global_lists/%s", dataset_name, dataset_name)
        _stream_group_id_dataset(
            hd=hd,
            dataset_name=dataset_name,
            size=size,
            groups=group_list,
            list_attr=list_attr,
            temp_dir=temp_dir,
        )
    
######################################################################
    
def save(obj, filename='test.hdf5'):
    """Function to save a CAESAR file to disk.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main caesar object to save.
    filename : str, optional
        Filename of the output file.

    Examples
    --------
    >>> obj.save('output.hdf5')
    
    """
    from yt.funcs import mylog

    if os.path.isfile(filename):
        mylog.warning('%s already present, overwriting!' % filename)
        os.remove(filename)
    mylog.info('Writing %s' % filename)
        
    outfile = h5py.File(filename, 'w')
    outfile.attrs.create('caesar', 315)
    
    
    unit_registry = obj.yt_dataset.unit_registry.to_json()
    outfile.attrs.create('unit_registry_json', unit_registry.encode('utf8'))

    serialize_global_attribs(obj, outfile)
    obj.simulation._serialize(obj, outfile)
    _ensure_dm_bookkeeping(obj)

    if hasattr(obj, 'halos') and obj.nhalos > 0:
        hd   = outfile.create_group('halo_data')
        hdd  = hd.create_group('lists')
        hddd = hd.create_group('dicts')

        # gather
        index_lists = ['dmlist']
        if 'gas' in obj.data_manager.ptypes: index_lists.append('glist')
        if 'star' in obj.data_manager.ptypes: index_lists.extend(['slist','galaxy_index_list'])
        if 'dm2' in obj.data_manager.ptypes: index_lists.append('dm2list')
        if 'dm3' in obj.data_manager.ptypes: index_lists.append('dm3list')
        if obj.data_manager.blackholes:
            index_lists.append('bhlist')
        if obj.data_manager.dust:
            index_lists.append('dlist')

        #write        
        for vals in index_lists:
            serialize_list(obj.halos, vals, hdd)
        serialize_attributes(obj.halos, hd, hddd)
  
    if hasattr(obj, 'galaxies') and obj.ngalaxies > 0:
        hd   = outfile.create_group('galaxy_data')
        hdd  = hd.create_group('lists')
        hddd = hd.create_group('dicts')

        # gather
        index_lists = ['glist', 'slist','cloud_index_list']
        if hasattr(obj.galaxies[0], 'AHF_ancestor_haloIDs'):
            index_lists.append('AHF_ancestor_haloIDs')
        if obj.data_manager.blackholes:
            index_lists.append('bhlist')
        if obj.data_manager.dust:
            index_lists.append('dlist')

        # write
        for vals in index_lists:
            serialize_list(obj.galaxies, vals, hdd)
        serialize_attributes(obj.galaxies, hd, hddd)


    if hasattr(obj, 'clouds') and obj.nclouds > 0:
        hd   = outfile.create_group('cloud_data')
        hdd  = hd.create_group('lists')
        hddd = hd.create_group('dicts')

        # gather
        index_lists = ['glist']
 
        # write
        for vals in index_lists:
            serialize_list(obj.clouds, vals, hdd)
        serialize_attributes(obj.clouds, hd, hddd)


        
    if hasattr(obj, 'global_particle_lists'):
        hd = outfile.create_group('global_lists')

        if getattr(obj, '_ahf_subhalo_streaming_save', False):
            _write_streaming_global_lists(obj, hd, filename=filename)
        else:
            # gather
            global_index_lists = ['halo_dmlist','halo_glist','halo_slist',
                                  'galaxy_glist','galaxy_slist','cloud_glist']
            if obj.data_manager.blackholes:
                global_index_lists.extend(['halo_bhlist','galaxy_bhlist'])
            if obj.data_manager.dust:
                global_index_lists.extend(['halo_dlist','galaxy_dlist'])

            # write
            for vals in global_index_lists:
                check_and_write_dataset(obj.global_particle_lists, vals, hd)
            
    outfile.close()
