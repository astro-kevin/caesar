import numpy as np
from yt.funcs import mylog
import os
import psutil
import time
from contextlib import contextmanager
from functools import wraps

# Global profiling data storage
_profile_data = {}
_profile_enabled = os.environ.get('CAESAR_PROFILE', '0') == '1'

# Flag to suppress memlog output (used during per-host property calculation)
_suppress_memlog = False


def memlog(msg):
    if _suppress_memlog:
        return
    process = psutil.Process(os.getpid())
    mylog.info('%s, RAM=%.4g GB'%(msg,process.memory_info()[0]/2.**30))


@contextmanager
def profile_section(name: str, enabled: bool = None):
    """Context manager for profiling code sections.

    Parameters
    ----------
    name : str
        Name of the code section being profiled
    enabled : bool, optional
        Override global profiling enable. If None, uses CAESAR_PROFILE env var.

    Usage
    -----
    >>> with profile_section('load_ahf_file'):
    ...     # code to profile
    ...     pass

    Output is logged as:
        [PROFILE] load_ahf_file: 12.34s, mem_delta=0.123GB
    """
    should_profile = enabled if enabled is not None else _profile_enabled

    if not should_profile:
        yield
        return

    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    start_mem = process.memory_info().rss

    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        end_mem = process.memory_info().rss
        delta_mem = (end_mem - start_mem) / 2**30

        _profile_data[name] = {
            'time': elapsed,
            'mem_delta_gb': delta_mem,
            'peak_mem_gb': end_mem / 2**30
        }
        mylog.info('[PROFILE] %s: %.2fs, mem_delta=%.3fGB', name, elapsed, delta_mem)


def profile_function(func):
    """Decorator for function-level profiling.

    Usage
    -----
    >>> @profile_function
    ... def my_expensive_function():
    ...     pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with profile_section(func.__name__):
            return func(*args, **kwargs)
    return wrapper


def get_profile_summary():
    """Return profiling summary sorted by time (descending).

    Returns
    -------
    dict
        Dictionary of profile data sorted by time, with keys as section names
        and values as dicts with 'time', 'mem_delta_gb', 'peak_mem_gb'.
    """
    return dict(sorted(_profile_data.items(), key=lambda x: -x[1]['time']))


def print_profile_summary():
    """Print a formatted profiling summary to the log."""
    if not _profile_data:
        mylog.info('[PROFILE] No profiling data collected')
        return

    summary = get_profile_summary()
    total_time = sum(v['time'] for v in summary.values())

    mylog.info('[PROFILE] ======== Summary ========')
    mylog.info('[PROFILE] Total profiled time: %.2fs', total_time)
    mylog.info('[PROFILE] Section breakdown:')
    for name, data in summary.items():
        pct = (data['time'] / total_time * 100) if total_time > 0 else 0
        mylog.info('[PROFILE]   %s: %.2fs (%.1f%%), mem_delta=%.3fGB',
                   name, data['time'], pct, data['mem_delta_gb'])
    mylog.info('[PROFILE] ============================')


def clear_profile_data():
    """Clear all collected profiling data."""
    global _profile_data
    _profile_data = {}

def rotator(vals, ALPHA=0, BETA=0):
    """Rotate particle set around given angles.

    Parameters
    ----------
    vals : np.array
        a Nx3 array typically consisting of
        either positions or velocities.
    ALPHA : float, optional
        First angle to rotate about
    BETA : float, optional
        Second angle to rotate about

    Examples
    --------
    >>> rotated_pos = rotator(positions, 32.3, 55.2)

    """
    c  = np.cos(ALPHA)
    s  = np.sin(ALPHA)
    Rx = np.array([
        [1.0,0.0,0.0],
        [0.0,  c, -s],
        [0.0,  s,  c]
    ])

    c  = np.cos(BETA)
    s  = np.sin(BETA)
    Ry = np.array([
        [  c,0.0, -s],
        [0.0,1.0,0.0],
        [  s,0.0,  c]
    ])

    # one value to rotate
    if len(np.shape(vals)) == 1:
        if ALPHA != 0:
            vals = np.dot(Rx, vals)
        if BETA != 0:
            vals = np.dot(Ry, vals)

    # rotating many values
    else:
        from .group_funcs import rotator as rotator_cython
        rotator_cython(np.asarray(vals), Rx, Ry, ALPHA, BETA)

    return vals


def calculate_local_densities(obj, group_list):
    """Calculate the local number and mass density of objects.

    Parameters
    ----------
    obj : SPHGR object
    group_list : list
        List of objects to perform this operation on.

    """
    if len(group_list) == 0:
        return

    try:
        from scipy.spatial import KDTree
        # from caesar.periodic_kdtree import PeriodicCKDTree
        #mylog.info('Calculating local densities')
    except:
        mylog.warning('Could not import scipy.spatial! '   \
                      'Please install scipy to allow for ' \
                      'local density calculations.')
        return

    pos  = np.array([i.pos for i in group_list])
    mass = np.array([i.masses['total'] for i in group_list])
    box  = obj.simulation.boxsize
    box  = np.array([box,box,box])
    np.mod(pos,box,out=pos)
    
    TREE = KDTree(pos, boxsize=obj.simulation.boxsize)

    if 'search_radius' in obj._kwargs:
        if isinstance(obj._kwargs['search_radius'],(int,float)):
            obj.simulation.search_radius = np.array([obj._kwargs['search_radius']])
        else:
            obj.simulation.search_radius = np.array(obj._kwargs['search_radius'])
        obj.simulation.search_radius = obj.yt_dataset.arr(obj.simulation.search_radius, obj.units['length'])

    for group in group_list:
        group.local_mass_density   = {}
        group.local_number_density = {}
        for search_radius in obj.simulation.search_radius:
            search_volume = 4.0/3.0 * np.pi * search_radius**3
            inrange = TREE.query_ball_point(group.pos, search_radius.d, workers=obj.nproc)
            total_mass = obj.yt_dataset.quan(np.sum(mass[inrange]), obj.units['mass'])
            rname = str(int(search_radius.d))
            group.local_mass_density[rname] = total_mass / search_volume
            group.local_number_density[rname] = float(len(inrange)) / search_volume


def info_printer(obj, group_type, top):
    """General method to print data.

    Parameters
    ----------
    obj : :class:`main.CAESAR`
        Main CAESAR object.
    group_type : {'halo','galaxy','cloud'}
        Type of group to print data for.
    top : int
        Number of objects to print.

    """
    from caesar.group import group_types
    if group_type == 'halo':
        group_list = obj.halos
    elif group_type == 'galaxy':
        group_list = obj.galaxies
    elif group_type == 'cloud':
        group_list = obj.clouds

    nobjs = len(group_list)
    if top > nobjs:
        top = nobjs

    if obj.simulation.cosmological_simulation:
        time = 'z=%0.3f' % obj.simulation.redshift
    else:
        time = 't=%0.3f' % obj.simulation.time

    output  = '\n'
    output += '## Largest %d %s\n' % (top, group_types[group_type])
    if hasattr(obj, 'data_file'): output += '## from: %s\n' % obj.data_file
    output += '## %d @ %s' % (nobjs, time)
    output += '\n\n'

    cnt = 1
    if group_type == 'halo':
        output += ' ID    Mdm       Mstar     Mgas      r         fgas   nrho\t|  CentralGalMstar\n'
        #         ' 0000  4.80e+09  4.80e+09  4.80e+09  7.64e-09  0.000  7.64e-09\t|  7.64e-09'
        output += ' ---------------------------------------------------------------------------------\n'
        for o in group_list:
            cgsm = -1
            if (hasattr(o,'central_galaxy')) & (hasattr(o.central_galaxy,'masses')): cgsm = o.central_galaxy.masses['stellar']
            output += ' %04d  %0.2e  %0.2e  %0.2e  %0.2e  %0.2e\t|  %0.2e \n' % \
                      (o.GroupID, o.masses['dm'], o.masses['stellar'],
                       o.masses['gas'],o.radii['total_half_mass'],
                       o.local_number_density['1000'], cgsm)
            cnt += 1
            if cnt > top: break
    elif group_type == 'galaxy':
        output += ' ID    Mstar     Mgas      SFR       r         fgas   nrho      Central\t|  Mhalo     HID\n'
        output += ' ----------------------------------------------------------------------------------------\n'
        #         ' 0000  4.80e+09  4.80e+09  4.80e+09  7.64e-09  0.000  7.64e-09  False
        for o in group_list:
            phm, phid = -1, -1
            if o.halo is not None: phm, phid = o.halo.masses['total'], o.halo.GroupID
            output += ' %04d  %0.2e  %0.2e  %0.2e  %0.2e  %0.2e  %s\t|  %0.2e  %d \n' % \
                      (o.GroupID, o.masses['stellar'], o.masses['gas'],
                       o.sfr, o.radii['total_half_mass'],
                       o.local_number_density['1000'], o.central,
                       phm, phid)
            cnt += 1
            if cnt > top: break
    elif group_type == 'cloud':
        output += ' ID    Mstar     Mgas      SFR       r         fgas   nrho      Central\t|  Mhalo     HID\n'
        output += ' ----------------------------------------------------------------------------------------\n'
        #         ' 0000  4.80e+09  4.80e+09  4.80e+09  7.64e-09  0.000  7.64e-09  False
        for o in group_list:
            halo = o.obj.galaxies[o.parent_galaxy_index].halo
            output += ' %04d  %0.2e  %0.2e  %0.2e  %0.2e   %0.2e  %s\t|  %0.2e  %d \n' % \
                      (o.GroupID, o.masses['stellar'], o.masses['gas'],
                       o.sfr, o.radii['total_half_mass'],
                       o.local_number_density['1000'], o.central,
                       halo.masses['dm'], halo.GroupID)
            cnt += 1
            if cnt > top: break



    print(output)
