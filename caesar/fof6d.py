
# 6-D FOF.
# Input is a snapshot file containing gas, stars, and BHs.
# Snapshot must contain a HaloID putting each particle in a halo (0=not in halo).
# First groups particles via an approximate FOF using  via quicksorts in successive directions.
# Then does a full 6-D FOF search on the approximate FOF groups
# Outputs SKID-format .grp binary file containing Npart, then galaxy IDs of all particles
#
# Romeel Dave, 25 Feb 2019

import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import sys
import os
import h5py
from yt.funcs import mylog
from caesar.utils import memlog
from caesar.property_manager import MY_DTYPE, get_property,has_ptype,ptype_ints
from caesar.group import MINIMUM_STARS_PER_GALAXY, MINIMUM_DM_PER_HALO, get_min_stars


class _PidLookup(object):
    """Compact PID -> index mapper backed by sorted numpy arrays."""

    __slots__ = ('sorted_pids', 'indices')

    def __init__(self, values: np.ndarray):
        arr = np.asarray(values, dtype=np.int64)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if arr.size == 0:
            self.sorted_pids = arr
            self.indices = np.empty(0, dtype=np.int64)
            return
        order = np.argsort(arr, kind='mergesort')
        self.sorted_pids = arr[order]
        self.indices = order.astype(np.int64, copy=False)

    def search(self, values: np.ndarray):
        """Return (indices, match_mask) for the supplied PID array."""
        arr = np.asarray(values, dtype=np.int64)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if arr.size == 0 or self.sorted_pids.size == 0:
            empty = np.empty(0, dtype=np.int64)
            mask = np.empty(0, dtype=bool)
            return empty, mask
        idx = np.searchsorted(self.sorted_pids, arr, side='left')
        mask = (idx < self.sorted_pids.size) & (self.sorted_pids[idx] == arr)
        return self.indices[idx[mask]], mask

class fof6d:

    def __init__(self, obj, group_type):
        self.obj = obj
        self.obj._args   = obj._args
        self.obj._kwargs = obj._kwargs
        self.obj_type = group_type
        self.Lbox = obj.simulation.boxsize.d
        self.counts = {}

        # set up number of processors
        self.nproc = obj.nproc

        # turn off unbinding; no need since fof6d already accounts for kinematics
        from caesar.group import group_types
        for gt in [group_type]:
            unbind_str = 'unbind_%s' % group_types[gt]
            setattr(self.obj.simulation, unbind_str, False)

    def load_haloid(self):
        # Get Halo IDs, either from snapshot or else run fof.
        # This will be a list of numpy arrays for each ptype
        if self.obj.load_haloid:
            from caesar.property_manager import get_haloid
            memlog('Using FOF Halo ID from snapshots')
            self.haloid = get_haloid(self.obj, self.obj.data_manager.ptypes, offset=-1)
        elif 'haloid' in self.obj._kwargs and 'fof' in self.obj._kwargs['haloid']:
            self.run_caesar_fof(self.obj)
        elif 'haloid' in self.obj._kwargs and 'rockstar' in self.obj._kwargs['haloid']:
            sys.exit('Sorry, reading from rockstar files not implemented yet')
        elif 'haloid' in self.obj._kwargs and 'AHF' in self.obj._kwargs['haloid']:
            haloid_flag = str(self.obj._kwargs['haloid']).lower()
            if haloid_flag == 'ahf-fast' and not self.obj._kwargs.get('AHF_use_subhalos', False):
                mylog.info('AHF-FAST detected; enabling AHF_use_subhalos')
                self.obj._kwargs['AHF_use_subhalos'] = True
            self.load_ahf_id()
        else:
            memlog('No Halo ID source specified -- running FOF.  This is the yt 3D FOF for halos and our homegrown 6D FOF for galaxies ...')
            try:
                self.run_caesar_fof(self.obj)
                self.obj._kwargs['haloid'] = 'fof'
            except:
                sys.exit("No Halo IDs found in snapshot -- please specify a source (haloid='fof' or 'snap')")
    def load_ahf_id(self):
        if 'haloid_file' in self.obj._kwargs and self.obj._kwargs['haloid'] is not None:
            haloid_file = self.obj._kwargs['haloid_file']
            haloid_flag = str(self.obj._kwargs.get('haloid', '')).lower()
            if os.path.isfile(haloid_file):
                memlog('Reading AHF halo IDs from %s'%(haloid_file))
                particles_file = haloid_file
                if particles_file.endswith('.gz'):
                    halos_file = particles_file.replace('particles.gz','halos')
                else:
                    halos_file = particles_file.replace('particles','halos')
                halo_info = np.loadtxt(halos_file, usecols=(0, 1), dtype=np.int64)

                if halo_info.size == 0:
                    sys.exit("No Halos in AHF halo file -- no need to run Caesar! Halos: %d" % halo_info.size)
                if halo_info.ndim == 1:
                    halo_info = halo_info.reshape(1, 2)

                total_halos = len(halo_info)

                host_mask = halo_info[:, 1] == 0
                mpi_warning_logged = False

                def _open_particles(path):
                    if path.endswith('.gz'):
                        import gzip
                        return gzip.open(path, 'rt')
                    return open(path, 'r')

                def _stream_particle_blocks(host_only=True):
                    fh = _open_particles(particles_file)

                    def _read_nonempty_line():
                        while True:
                            raw = fh.readline()
                            if not raw:
                                return None
                            stripped = raw.strip()
                            if stripped:
                                return stripped

                    def _skip_rows(count):
                        for _ in range(count):
                            fh.readline()

                    def _read_block(count):
                        if count <= 0:
                            return np.empty((0, 2), dtype=np.int64)
                        lines = []
                        filled = 0
                        while filled < count:
                            raw = fh.readline()
                            if not raw:
                                break
                            stripped = raw.strip()
                            if not stripped:
                                continue
                            lines.append(stripped)
                            filled += 1
                        if not lines:
                            return np.empty((0, 2), dtype=np.int64)
                        buf = "\n".join(lines)
                        arr = np.fromstring(buf, sep=' ', dtype=np.int64, count=2*len(lines))
                        if arr.size == 0:
                            return np.empty((0, 2), dtype=np.int64)
                        arr = arr.reshape(-1, 2)
                        return arr

                    first_line = _read_nonempty_line()
                    if first_line is None:
                        fh.close()
                        return
                    expected = int(first_line)
                    mpi_mode = expected != total_halos
                    if mpi_mode:
                        nonlocal mpi_warning_logged
                        if not mpi_warning_logged:
                            memlog('!!Warning!! reading AHF halo IDs from merged files!!')
                            mpi_warning_logged = True

                    try:
                        if not mpi_mode:
                            for idx in range(total_halos):
                                header = _read_nonempty_line()
                                if header is None:
                                    break
                                parts = header.split()
                                if len(parts) != 2:
                                    continue
                                npt = int(parts[0])
                                hid = int(parts[1])
                                if host_only and not host_mask[idx]:
                                    _skip_rows(npt)
                                    continue
                                block = _read_block(npt)
                                yield idx, hid, block
                        else:
                            remaining_in_block = expected
                            idx = 0
                            while idx < total_halos:
                                if remaining_in_block == 0:
                                    block_line = _read_nonempty_line()
                                    if block_line is None:
                                        break
                                    remaining_in_block = int(block_line)
                                    continue
                                header = _read_nonempty_line()
                                if header is None:
                                    break
                                parts = header.split()
                                if len(parts) != 2:
                                    continue
                                npt = int(parts[0])
                                hid = int(parts[1])
                                if host_only and not host_mask[idx]:
                                    _skip_rows(npt)
                                else:
                                    block = _read_block(npt)
                                    yield idx, hid, block
                                remaining_in_block -= 1
                                idx += 1
                    finally:
                        fh.close()

                use_subhalos = self.obj._kwargs.get('AHF_use_subhalos', False)

                if not use_subhalos:  # only use particles in distinct halos. this is default.

                    self.haloid = {}

                    lookup_map = {}
                    tmpp_by_ptype = {}
                    for p in self.obj.data_manager.ptypes:
                        if has_ptype(self.obj, p):
                            data = get_property(self.obj, 'pid', p).d.astype(np.int64)
                            tmpp = np.full(len(data), -1, dtype=np.int64)
                            self.haloid[p] = tmpp
                            lookup_map[ptype_ints[p]] = (_PidLookup(data), tmpp)
                            tmpp_by_ptype[p] = tmpp
                        else:
                            tmpp = np.empty(0, dtype=np.int64)
                            self.haloid[p] = tmpp
                            tmpp_by_ptype[p] = tmpp

                    nhid = 0
                    memlog('Reading simulation data IDs and mapping halo particle to them')
                    for _, hid, block in _stream_particle_blocks(host_only=True):
                        if block.size == 0:
                            continue
                        block = np.asarray(block, dtype=np.int64)
                        pid_vals = block[:, 0]
                        type_vals = block[:, 1]
                        ptypes_present, inv = np.unique(type_vals, return_inverse=True)
                        hid_val = int(hid)
                        for code in ptypes_present:
                            entry = lookup_map.get(int(code))
                            if entry is None:
                                continue
                            lookup, tmpp = entry
                            mask = inv == np.where(ptypes_present == code)[0][0]
                            if not np.any(mask):
                                continue
                            indices, matched_mask = lookup.search(pid_vals[mask])
                            if indices.size == 0:
                                continue
                            tmpp[indices] = hid_val
                            nhid += indices.size

                    if haloid_flag != 'ahf-fast':
                        assigned = []
                        for tmpp in tmpp_by_ptype.values():
                            if tmpp.size:
                                mapped = tmpp[tmpp >= 0]
                                if mapped.size:
                                    assigned.append(mapped)
                        all_halo_ids = (
                            np.concatenate(assigned).astype(np.int64, copy=False)
                            if assigned
                            else np.empty(0, dtype=np.int64)
                        )
                        self.obj.data_manager.haloid = all_halo_ids
                    memlog('Total halo particle IDs = %d' % (nhid))
                    return
                else: # use subhalo information as well, but very pain to remove these duplicated particles!!!!
                    parent_of = {int(row[0]): int(row[1]) for row in halo_info}
                    children_of = {}
                    for hid_val, host_val in parent_of.items():
                        children_of.setdefault(int(host_val), []).append(int(hid_val))

                    depth_cache = {}

                    def _depth(h):
                        h = int(h)
                        if h in depth_cache:
                            return depth_cache[h]
                        parent = parent_of.get(h, 0)
                        if parent <= 0 or parent == h:
                            depth_cache[h] = 0
                        else:
                            depth_cache[h] = _depth(parent) + 1
                        return depth_cache[h]

                    node_to_root = {}
                    host_to_nodes = {}

                    def _resolve_root(node):
                        cur = int(node)
                        path = []
                        while True:
                            parent = parent_of.get(cur, 0)
                            if parent <= 0 or parent == cur:
                                root = cur
                                break
                            path.append(cur)
                            cur = int(parent)
                        for item in path:
                            node_to_root[item] = root
                        node_to_root[int(node)] = root
                        return root

                    for hid_val in parent_of.keys():
                        root = node_to_root.get(int(hid_val))
                        if root is None:
                            root = _resolve_root(hid_val)
                        host_to_nodes.setdefault(root, set()).add(int(hid_val))

                    nodes_remaining = {root: len(nodes) for root, nodes in host_to_nodes.items()}
                    pending_members = {}
                    records = []

                    def _process_host(root_id, bucket):
                        members = {}
                        for hid_val, block in bucket.items():
                            arr = np.asarray(block, dtype=np.int64)
                            if arr.size == 0:
                                arr = np.empty((0, 2), dtype=np.int64)
                            else:
                                arr = np.atleast_2d(arr).reshape(-1, 2)
                            members[int(hid_val)] = arr

                        if not members:
                            return

                        stacked = []
                        for hid_val, pdata in members.items():
                            if pdata.size == 0:
                                continue
                            depth_val = _depth(hid_val)
                            depth_col = np.full(pdata.shape[0], depth_val, dtype=np.int16)
                            hid_col = np.full(pdata.shape[0], np.int64(hid_val), dtype=np.int64)
                            stacked.append(np.column_stack((pdata, depth_col, hid_col)))

                        if not stacked:
                            return

                        combined = np.vstack(stacked)  # cols: pid, ptype, depth, hid
                        order = np.lexsort((-combined[:, 2], combined[:, 0]))
                        combined = combined[order]

                        bary_mask = combined[:, 1] != 1
                        baryons = combined[bary_mask]
                        if baryons.size:
                            keep = np.ones(len(baryons), dtype=bool)
                            keep[1:] = baryons[1:, 0] != baryons[:-1, 0]
                            baryons = baryons[keep]

                        dm_rows = combined[~bary_mask]
                        reduced = []
                        if dm_rows.size:
                            reduced.append(dm_rows[:, [0, 1, 3]])
                        if baryons.size:
                            reduced.append(baryons[:, [0, 1, 3]])
                        if not reduced:
                            return
                        final_rows = np.vstack(reduced)

                        order = np.argsort(final_rows[:, 2], kind='mergesort')
                        final_rows = final_rows[order]
                        unique_hids, start_idx = np.unique(final_rows[:, 2], return_index=True)
                        bounds = np.append(start_idx, len(final_rows))
                        ms = get_min_stars(self.obj)

                        for i, hid_val in enumerate(unique_hids):
                            pdata = final_rows[bounds[i]:bounds[i + 1], :2]
                            if pdata.size == 0:
                                continue
                            dm_count = int(np.sum(pdata[:, 1] == 1))
                            star_count = int(np.sum(pdata[:, 1] == 4))
                            if dm_count < MINIMUM_DM_PER_HALO and star_count < ms:
                                local_children = [child for child in children_of.get(int(hid_val), []) if child in members]
                                if not local_children:
                                    continue
                            tmppd = np.zeros((pdata.shape[0], 3), dtype=np.int64)
                            tmppd[:, :2] = pdata
                            tmppd[:, 2] = np.int64(hid_val)
                            records.append(tmppd)

                    for _, hid_val, block in _stream_particle_blocks(host_only=False):
                        hid_int = int(hid_val)
                        root = node_to_root.get(hid_int)
                        if root is None:
                            continue
                        bucket = pending_members.setdefault(root, {})
                        bucket[hid_int] = np.array(block, copy=False)
                        nodes_remaining[root] -= 1
                        if nodes_remaining[root] <= 0:
                            _process_host(root, bucket)
                            pending_members.pop(root, None)
                            nodes_remaining.pop(root, None)

                    for root, bucket in pending_members.items():
                        _process_host(root, bucket)

                    if records:
                        hid_info = np.vstack(records)
                    else:
                        hid_info = np.empty((0, 3), dtype=np.int64)

                # Now load the simulation particle IDs # map back to the position
                self.haloid = {}
                lookup_map = {}
                tmpp_by_ptype = {}
                for p in self.obj.data_manager.ptypes:
                    if has_ptype(self.obj, p):
                        data = get_property(self.obj, 'pid', p).d.astype(np.int64)
                        tmpp = np.full(len(data), -1, dtype=np.int64)
                        self.haloid[p] = tmpp
                        lookup_map[ptype_ints[p]] = (_PidLookup(data), tmpp)
                        tmpp_by_ptype[p] = tmpp
                    else:
                        tmpp = np.empty(0, dtype=np.int64)
                        self.haloid[p] = tmpp
                        tmpp_by_ptype[p] = tmpp

                nhid = 0
                memlog('Reading simulation data IDs and mapping halo particle to them')
                if hid_info.size:
                    pid_vals = hid_info[:, 0]
                    type_vals = hid_info[:, 1]
                    halo_vals = hid_info[:, 2]
                    for code, (lookup, tmpp) in lookup_map.items():
                        mask = type_vals == code
                        if not np.any(mask):
                            continue
                        subset_pids = pid_vals[mask]
                        indices, matched_mask = lookup.search(subset_pids)
                        if indices.size == 0:
                            continue
                        tmpp[indices] = halo_vals[mask][matched_mask]
                        nhid += indices.size

                memlog('Total halo particle IDs = %d' % (nhid))
                if haloid_flag != 'ahf-fast':
                    assigned = []
                    for tmpp in tmpp_by_ptype.values():
                        if tmpp.size:
                            mapped = tmpp[tmpp >= 0]
                            if mapped.size:
                                assigned.append(mapped)
                    all_halo_ids = (
                        np.concatenate(assigned).astype(np.int64, copy=False)
                        if assigned
                        else np.empty(0, dtype=np.int64)
                    )
                    self.obj.data_manager.haloid = all_halo_ids
            else:
                sys.exit('No ID data file is found in %s' % haloid_file)   
        else:
            mylog.warning("With haloid='AHF' you must also specify a haloid_file containing halo[+subhalo] IDs.")
            sys.exit()

    def run_caesar_fof(self,obj):
        from caesar.fubar import get_mean_interparticle_separation,get_b,fof
        LL = get_mean_interparticle_separation(self.obj) * get_b(self.obj, 'halo')
        if 'haloid_file' in self.obj._kwargs and self.obj._kwargs['haloid'] is not None:
            haloid_file = self.obj._kwargs['haloid_file']
            if os.path.isfile(haloid_file):
                memlog('Reading 3D FOF Halo IDs from %s'%haloid_file)
                hf = h5py.File(haloid_file,'r')
                self.obj.data_manager.haloid = np.asarray(hf['all_haloids'])
                self.haloid = {}
                for p in self.obj.data_manager.ptypes:  # read haloid arrays for each ptype
                    self.haloid[p]=np.asarray(hf['haloids_%s'%p])
                # self.haloid = np.array(self.haloid, dtype=object)
                hf.close()
                return
        else:
            haloid_file = None
        memlog('Running 3D FOF to get Halo IDs, LL=%g'%LL)
        pos = np.empty((0,3),dtype=MY_DTYPE)
        ptype = np.empty(0,dtype=np.int32)
        for p in self.obj.data_manager.ptypes:  # get positions
            if not has_ptype(self.obj, p): continue
            data = get_property(self.obj, 'pos', p).to(self.obj.units['length'])
            pos = np.append(pos, data.d, axis=0)
            ptype = np.append(ptype, np.full(len(data), ptype_ints[p], dtype=np.int32), axis=0)
        haloid_all = fof(self.obj, pos, LL, group_type='halo')  # run FOF
        self.haloid = {}
        haloid = np.empty(0,dtype=np.int64)
        for p in self.obj.data_manager.ptypes:  # fill haloid arrays for each ptype
            if has_ptype(self.obj, p):
                data = haloid_all[ptype==ptype_ints[p]]
                datasel = data[data>=0]
            else:
                data = np.empty(0,dtype=np.int64)
                datasel = np.empty(0,dtype=np.int64)
            self.haloid[p] = data
            haloid = np.append(haloid,datasel,axis=0)
        # self.haloid = np.asarray(self.haloid, dtype=object)
        self.obj.data_manager.haloid = haloid
        if haloid_file is not None:
            memlog('Writing 3D FOF Halo IDs to %s' % haloid_file)
            with h5py.File(haloid_file,'w') as hf:
                hf.create_dataset('all_haloids',data=haloid, compression=1)
                for p in self.obj.data_manager.ptypes:  # write haloid arrays for each ptype
                    haloid_out = self.haloid[p]
                    hf.create_dataset('haloids_%s'%p,data=haloid_out, compression=1)
                hf.close()

    def plist_init(self,parent=None):
        # set up particle lists
        if self.obj_type == 'halo' or parent is None:
            grpid = self.obj.data_manager.haloid - 1
            if len(grpid[grpid>=0]) < MINIMUM_DM_PER_HALO:
                mylog.warning('Not enough halo particles for a single valid halo (%d < %d)'%(len(grpid[grpid>=0]),MINIMUM_DM_PER_HALO))
                return False
        else:
            grpid = parent.tags_fof6d
            # Use unified min_stars threshold
            ms = get_min_stars(self.obj)
            if len(grpid[grpid>=0]) < ms:
                self.nparttot = 0
                return False

        # sort by grpid
        self.nparttot = len(grpid)
        self.nparttype = {}
        for p in self.obj.data_manager.ptypes:
            self.nparttype[p] = len(grpid[self.obj.data_manager.ptype==ptype_ints[p]])
        sort_grpid = np.argsort(grpid)
        self.pid_sorted = np.arange(self.nparttot,dtype=np.int64)[sort_grpid]
        # self.grouplist = np.unique(grpid)  # list of objects (halo/galaxy/cloud) to process
        hid_sorted = grpid[sort_grpid]
        self.hid_bins = find_bins(hid_sorted,self.nparttot)
        self.grouplist = hid_sorted[self.hid_bins[:-1]] # list of objects (halo/galaxy/cloud) as haloid-1 in the same order
        return True

    def run_fof6d(self, target_type, nHlim=0.13, Tlim=1.e5, sfflag=True, minstars=MINIMUM_STARS_PER_GALAXY):

        from joblib import Parallel, delayed
        from caesar.fubar import get_b
        from caesar.group import group_types

        # initialize fof6d parameter
        self.fof_LL = self.MIS * get_b(self.obj, target_type)
        # default velocity-space linking factor (relative to local sigma)
        self.vel_LL = 1.0
        self.kerneltab = kernel_table(self.fof_LL)
        self.nHlim = nHlim  # only include gas above this nH limit (atoms/cm^3)
        self.Tlim = Tlim  # only include gas below this temperature
        self.sfflag = sfflag  # if True, always include particles with nonzero SF regardless of other crit
        self.minstars = minstars
        # set eligible galaxy gas (restore ISM gate for AHF as in upstream):
        # nH > nHlim and (T < Tlim or SFR > 0) when sfflag is True; otherwise nH > nHlim and T < Tlim
        # This applies to both standard and AHF modes for galaxy FOF.
        if self.sfflag:
            self.dense_crit = lambda gnh, gtemp, gsfr: (gnh > self.nHlim) & ((gtemp < self.Tlim) | (gsfr > 0))
        else:
            self.dense_crit = lambda gnh, gtemp, gsfr: (gnh > self.nHlim) & (gtemp < self.Tlim)

        # Record mode string for diagnostics below
        haloid_mode = ''
        try:
            haloid_mode = str(self.obj._kwargs.get('haloid', '')).upper()
        except Exception:
            haloid_mode = ''
        # Allow environment overrides for velocity gating
        try:
            _vel_env = os.environ.get('CAESAR_FOF6D_VEL_LL')
            if _vel_env is not None and _vel_env != '':
                self.vel_LL = float(_vel_env)
        except Exception:
            pass

        # Optional: in AHF modes, permit disabling the velocity gate entirely for debugging
        try:
            if haloid_mode in ('AHF', 'AHF-FAST') and target_type == 'galaxy':
                if os.environ.get('CAESAR_FOF6D_DISABLE_VEL', '0') == '1':
                    self.vel_LL = None
                    mylog.info('fof6d: AHF mode with spatial-only linking (velocity gate disabled)')
        except Exception:
            pass

        memlog(f'fof6d run for target={target_type}, mode={haloid_mode}, nHlim={self.nHlim}, Tlim={self.Tlim}, sfflag={self.sfflag}')

        # Optional: verify halos we process have gas present (before eligibility mapping)
        import os as _os
        do_halo_gas_report = _os.environ.get('CAESAR_FOF6D_HALO_GAS_REPORT', '0') == '1'
        do_halo_gas_assert = _os.environ.get('CAESAR_ASSERT_HALO_HAS_GAS', '0') == '1'
        do_progress = (_os.environ.get('CAESAR_FOF6D_PROGRESS', '0') == '1' or _os.environ.get('CAESAR_PROGRESS', '0') == '1')
        verbose_gas = _os.environ.get('CAESAR_FOF6D_HALO_GAS_VERBOSE', '0') == '1'
        try:
            _report_n = int(_os.environ.get('CAESAR_FOF6D_CHECK_N', '50'))
        except Exception:
            _report_n = 50
        if do_halo_gas_report or do_halo_gas_assert:
            from yt.funcs import mylog
            halos_missing_gas = 0
            total_eligible = 0
            it = range(len(self.obj.halo_list))
            if do_progress:
                try:
                    try:
                        from yt.extern.tqdm import tqdm as _tqdm
                    except Exception:
                        try:
                            from tqdm.auto import tqdm as _tqdm
                        except Exception:
                            from tqdm import tqdm as _tqdm  # type: ignore
                    it = _tqdm(it, desc='fof6d: scan halos for gas', leave=False)
                except Exception:
                    pass
            for ih in it:
                hi = self.obj.halo_list[ih].global_indexes
                if hi.size == 0:
                    continue
                ptypes_h = self.obj.data_manager.ptype[hi]
                gas_cnt = int(np.sum(ptypes_h == ptype_ints['gas']))
                star_cnt = int(np.sum(ptypes_h == ptype_ints['star']))
                will_process = (star_cnt >= self.minstars)
                if will_process:
                    total_eligible += 1
                    if gas_cnt == 0:
                        halos_missing_gas += 1
                if verbose_gas and ih < _report_n:
                    mylog.debug('fof6d halo gas report: halo=%d stars=%d gas=%d will_process=%s', ih, star_cnt, gas_cnt, str(will_process))
            mylog.info('fof6d halo gas summary: eligible_halos=%d with_no_gas=%d', total_eligible, halos_missing_gas)
            if do_halo_gas_assert and total_eligible > 0 and halos_missing_gas == total_eligible:
                mylog.error('Assertion: All eligible halos (with >= %d stars) have zero gas at halo stage', self.minstars)
                raise AssertionError('No gas present in any eligible halo before FOF; investigate selection/mapping')

        # collect indices for eligible particles
        memlog('Running fof6d on %d halos w/%d proc(s), LL=%g'%(len(self.obj.halo_list),self.nproc,self.fof_LL))
        g_inds = []  # indexes of (gas star BH dust) particle eligible for being in a group
        halos_with_any_gas = 0
        halos_with_no_gas = 0
        total_gas_candidates = 0
        total_star_candidates = 0
        len_hi = len_gi = 0
        it2 = range(len(self.obj.halo_list))
        if do_progress:
            try:
                try:
                    from yt.extern.tqdm import tqdm as _tqdm
                except Exception:
                    try:
                        from tqdm.auto import tqdm as _tqdm
                    except Exception:
                        from tqdm import tqdm as _tqdm  # type: ignore
                it2 = _tqdm(it2, desc='fof6d: select eligible', leave=False)
            except Exception:
                pass
        for ih in it2:
            eligible = setup_indexes(self,self.obj.halo_list[ih].global_indexes)
            g_inds.append(eligible)
            # diagnostics: count components
            if eligible.size > 0:
                ptypes_e = self.obj.data_manager.ptype[eligible]
                gas_c = int(np.sum(ptypes_e == ptype_ints['gas']))
                star_c = int(np.sum(ptypes_e == ptype_ints['star']))
                total_gas_candidates += gas_c
                total_star_candidates += star_c
                if gas_c > 0:
                    halos_with_any_gas += 1
                else:
                    halos_with_no_gas += 1
            len_hi += len(self.obj.halo_list[ih].global_indexes)
            len_gi += len(eligible)
        if len(self.obj.halo_list) > 0:
            memlog(
                f'fof6d eligibility: halos={len(self.obj.halo_list)}, '
                f'with_gas={halos_with_any_gas}, no_gas={halos_with_no_gas}, '
                f'gas_candidates={total_gas_candidates}, star_candidates={total_star_candidates}'
            )
        memlog('%d halo particles, %g%% eligible for galaxies'%(len_hi, np.round(100*len_gi/len_hi,2)))

        # get tags using fof6d
        ngs=0
        # Optional: pre-tag eligible gas count across sample for assert
        import os as _os
        do_tag_assert = _os.environ.get('CAESAR_ASSERT_TAGGED_GAS', '0') == '1'
        try:
            sample_n = int(_os.environ.get('CAESAR_FOF6D_CHECK_N', '50'))
        except Exception:
            sample_n = 50
        pre_tag_gas = None
        if do_tag_assert:
            pre_tag_gas = []
            for ih in range(min(len(self.obj.halo_list), sample_n)):
                ptypes_e = self.obj.data_manager.ptype[g_inds[ih]] if g_inds[ih].size>0 else np.array([], dtype=np.int32)
                pre_tag_gas.append(int(np.sum(ptypes_e == ptype_ints['gas'])))

        if self.nproc == 1:
            grp_tags = [None]*len(self.obj.halo_list)  # particle IDs for fof6d objects
            for ih in range(len(self.obj.halo_list)):
                grp_tags[ih],tmg = fof6d_halo(
                    len(self.obj.halo_list[ih].global_indexes),
                    len(g_inds[ih]),
                    self.obj.data_manager.pos[g_inds[ih]],
                    self.obj.data_manager.vel[g_inds[ih]],
                    self.minstars,
                    self.obj.simulation.boxsize.d,
                    self.fof_LL,
                    self.vel_LL,
                    self.kerneltab,
                )
                ngs+=tmg
        else:
            tmg = Parallel(n_jobs=self.nproc)(
                delayed(fof6d_halo)(
                    len(self.obj.halo_list[ih].global_indexes),
                    len(g_inds[ih]),
                    self.obj.data_manager.pos[g_inds[ih]],
                    self.obj.data_manager.vel[g_inds[ih]],
                    self.minstars,
                    self.obj.simulation.boxsize.d,
                    self.fof_LL,
                    self.vel_LL,
                    self.kerneltab,
                ) for ih in range(len(self.obj.halo_list))
            )
            grp_tags,ngs = zip(*tmg)
            grp_tags=list(grp_tags)
            ngs = np.sum(ngs)

        # adjust tags to be sequential in galaxy number overall (rather than within each halo)
        ngrp = 0
        memlog('total galaxies %d, total groups %d'%(ngs,len(grp_tags)))
        self.group_parents = np.zeros(ngs,dtype=np.int32)
        for ih in range(len(grp_tags)):
            grp_tags[ih] = np.where(grp_tags[ih]>=0, grp_tags[ih]+ngrp, -1)
            mygals = np.unique(grp_tags[ih])
            mygals = mygals[mygals>=0]
            self.group_parents[mygals] = ih
            ngrp += len(mygals)

        if ngrp != ngs:
            memlog('WARNING!! total galaxies number do not agree: %d, %d'%(ngs,ngrp))
        # collect overall tags
        self.tags_fof6d = np.zeros(self.nparttot,dtype=np.int64) - 1
        for igrp in range(len(grp_tags)):
            self.tags_fof6d[g_inds[igrp]] = grp_tags[igrp]

        memlog('Done fof6d, found %d %s'%(ngrp,group_types[target_type]))

        # Optional per-halo tag report for first N halos: how many gas/stars eligible vs tagged
        do_tag_report = _os.environ.get('CAESAR_FOF6D_GAS_REPORT', '0') == '1'
        try:
            report_n = int(_os.environ.get('CAESAR_FOF6D_CHECK_N', '50'))
        except Exception:
            report_n = 50
        if do_tag_report:
            from yt.funcs import mylog
            tot_gas_elig = tot_gas_tag = 0
            tot_star_elig = tot_star_tag = 0
            sampled = 0
            for ih in range(min(len(self.obj.halo_list), report_n)):
                elig = g_inds[ih]
                if elig.size == 0:
                    mylog.debug('fof6d tag report: halo=%d elig_total=0', ih)
                    continue
                ptypes_e = self.obj.data_manager.ptype[elig]
                gas_mask = (ptypes_e == ptype_ints['gas'])
                star_mask = (ptypes_e == ptype_ints['star'])
                tags = grp_tags[ih]
                gas_elig = int(gas_mask.sum())
                star_elig = int(star_mask.sum())
                gas_tag = int(np.sum(tags[gas_mask] >= 0)) if gas_elig>0 else 0
                star_tag = int(np.sum(tags[star_mask] >= 0)) if star_elig>0 else 0
                mylog.debug('fof6d tag report: halo=%d gas_elig=%d gas_tagged=%d star_elig=%d star_tagged=%d', ih, gas_elig, gas_tag, star_elig, star_tag)
                tot_gas_elig += gas_elig
                tot_gas_tag += gas_tag
                tot_star_elig += star_elig
                tot_star_tag += star_tag
                sampled += 1
            # Aggregate summary over sampled halos
            gas_rate = (tot_gas_tag / tot_gas_elig) if tot_gas_elig > 0 else 0.0
            star_rate = (tot_star_tag / tot_star_elig) if tot_star_elig > 0 else 0.0
            mylog.info('fof6d tag summary: halos_sampled=%d gas_elig=%d gas_tagged=%d (rate=%.3f) star_elig=%d star_tagged=%d (rate=%.3f)'
                       , sampled, tot_gas_elig, tot_gas_tag, gas_rate, tot_star_elig, tot_star_tag, star_rate)

        # Post-tag assert: if there was eligible gas in the sample halos but none got tagged
        if do_tag_assert and pre_tag_gas is not None:
            tagged_any = False
            for ih in range(min(len(self.obj.halo_list), sample_n)):
                if pre_tag_gas[ih] <= 0:
                    continue
                tags = grp_tags[ih]
                if tags is None or len(tags) == 0:
                    continue
                # Count how many gas candidates have tag >= 0
                elig = g_inds[ih]
                if elig.size == 0:
                    continue
                ptypes_e = self.obj.data_manager.ptype[elig]
                gas_mask = (ptypes_e == ptype_ints['gas'])
                if gas_mask.any():
                    gas_tags = tags[gas_mask]
                    if np.any(gas_tags >= 0):
                        tagged_any = True
                        break
            if any(v>0 for v in pre_tag_gas) and not tagged_any:
                from yt.funcs import mylog
                mylog.error('Assertion: 6D-FOF tagging did not assign any gas in sampled halos despite eligible gas present')
                raise AssertionError('6D-FOF tagging did not assign any gas in sampled halos despite eligible gas present')

    def load_lists(self,parent=None):
        # create valid caesar groups, populate index lists
        from caesar.group import create_new_group, group_types
        from caesar.property_manager import ptype_ints, has_ptype
        import os as _os
        grp_list = []
        if parent is not None:
            for ihalo in range(len(parent.obj.halo_list)):
                parent.obj.halo_list[ihalo].galaxy_index_list = []
        ngrp = 0
        zero_marker = 0
        keep_all = bool(getattr(self, 'keep_all_groups', False)) and self.obj_type == 'halo'
        _check = _os.environ.get('CAESAR_FOF6D_CHECK', '0') == '1'
        _assert_map = _os.environ.get('CAESAR_ASSERT_FOF6D', '0') == '1'
        try:
            _check_n = int(_os.environ.get('CAESAR_FOF6D_CHECK_N', '50'))
        except Exception:
            _check_n = 50
        _checked = 0

        for igrp in range(len(self.grouplist)):
            if self.grouplist[igrp] < 0:
                zero_marker = 1  # if there are particles with tag=-1, these will be in igrp=0 within hid_bins. In this case, group_parents should start their numbering at 1, since igrp=0 is not a valid object.  This should only happen for galaxies/clouds, not halos
                continue
            mygrp = create_new_group(self.obj, self.obj_type)
            my_indexes = self.pid_sorted[self.hid_bins[igrp]:self.hid_bins[igrp+1]]  # indexes for parts in this group
            # load indexes into lists for a given group, globally and for each particle type
            my_ptype = self.obj.data_manager.ptype[my_indexes]
            my_pos = self.obj.data_manager.pos[my_indexes]
            mygrp.global_indexes = my_indexes
            offset = 0
            mygrp.ngas = mygrp.nstar = mygrp.nbh = mygrp.ndust = mygrp.ndm = mygrp.ndm2 = mygrp.ndm3 = 0
            for p in self.obj.data_manager.ptypes:
                if not has_ptype(self.obj, p): continue
                if p == 'gas':
                    mygrp.glist = my_indexes[my_ptype==ptype_ints[p]]-offset
                    mygrp.ngas = len(mygrp.glist)
                elif p == 'star':
                    mygrp.slist = my_indexes[my_ptype==ptype_ints[p]]-offset
                    mygrp.nstar = len(mygrp.slist)
                elif p == 'bh':
                    mygrp.bhlist = my_indexes[my_ptype==ptype_ints[p]]-offset
                    mygrp.nbh = len(mygrp.bhlist)
                elif p == 'dust':
                    mygrp.dlist = my_indexes[my_ptype==ptype_ints[p]]-offset
                    mygrp.ndust = len(mygrp.dlist)
                elif p == 'dm':
                    mygrp.dmlist = my_indexes[my_ptype==ptype_ints[p]]-offset
                    mygrp.ndm = len(mygrp.dmlist)
                elif p == 'dm2':
                    mygrp.dm2list = my_indexes[my_ptype==ptype_ints[p]]-offset
                    mygrp.ndm2 = len(mygrp.dm2list)
                elif p == 'dm3':
                    mygrp.dm3list = my_indexes[my_ptype==ptype_ints[p]]-offset
                    mygrp.ndm3 = len(mygrp.dm3list)
                offset += self.nparttype[p]

            # (demo logging disabled)

            # Optional lightweight correctness checks for mapping (first N groups only)
            if ( _check or _assert_map ) and _checked < _check_n and self.obj_type == 'galaxy':
                try:
                    gas_concat = my_indexes[my_ptype==ptype_ints['gas']]
                    exp_gas = int(gas_concat.size)
                except Exception:
                    exp_gas = 0
                got_gas = int(getattr(mygrp, 'ngas', 0))
                if exp_gas != got_gas:
                    from yt.funcs import mylog
                    # Validate a small sample via concat_to_selected to avoid memory blowup
                    sample = gas_concat[:min(8, gas_concat.size)] if exp_gas>0 else gas_concat
                    try:
                        sel_map = self.obj.data_manager.concat_to_selected('gas', sample)
                        mylog.warning('fof6d load_lists gas mismatch: group=%d exp=%d got=%d sample_sel=%s', igrp, exp_gas, got_gas, sel_map)
                    except Exception as _e:
                        mylog.warning('fof6d load_lists gas mismatch: group=%d exp=%d got=%d (map err: %s)', igrp, exp_gas, got_gas, _e)
                    if _assert_map:
                        raise AssertionError(f'FOF6D mapping mismatch: group={igrp} exp_gas={exp_gas} got_gas={got_gas}')
                _checked += 1
            include_group = mygrp._valid or keep_all
            if include_group:
                if keep_all and not mygrp._valid:
                    setattr(mygrp, '_forced_include', True)
                mygrp.obj_type = self.obj_type
                if self.obj_type == 'halo':
                    haloid_mode = str(self.obj._kwargs.get('haloid', '')).upper()
                    if haloid_mode in ('AHF', 'AHF-FAST'):
                        mygrp.AHF_haloID = self.grouplist[igrp] + 1 # recover to original ID, see line 156
                if parent is not None:
                    ihalo = parent.group_parents[igrp-zero_marker]
                    mygrp.parent_halo_index = ihalo
                    parent.obj.halo_list[ihalo].galaxy_index_list.append(ngrp)
                    ngrp += 1
                grp_list.append(mygrp)
            #else:
            #    print('Not selected halo -- ', self.grouplist[igrp] + 1, 'with ngas: ',len(mygrp.glist), 'nstar:', len(mygrp.slist),'ndm:',len(mygrp.dmlist))

        if self.obj_type == 'halo':
            self.obj.halo_list = grp_list
            self.counts[self.obj_type] = len(self.obj.halo_list)
            self.obj.group_types.append(self.obj_type)
        if self.obj_type == 'galaxy':
            self.obj.galaxy_list = grp_list
            self.counts[self.obj_type] = len(self.obj.galaxy_list)
            self.obj.group_types.append(self.obj_type)
            # Summary diagnostics: how many galaxies have gas immediately after 6D-FOF list build?
            if _os.environ.get('CAESAR_FOF6D_SUMMARY', '0') == '1' or _assert_map:
                ngals = len(self.obj.galaxy_list)
                with_gas = sum(1 for g in self.obj.galaxy_list if getattr(g, 'glist', []) is not None and len(g.glist) > 0)
                from yt.funcs import mylog
                mylog.info('fof6d load_lists summary: galaxies=%d with_gas=%d without_gas=%d', ngals, with_gas, ngals - with_gas)
                if _assert_map and with_gas == 0 and ngals > 0:
                    mylog.error('Assertion: FOF6D produced %d galaxies but none have gas', ngals)
                    raise AssertionError('FOF6D produced galaxies but none have gas; failing fast for investigation')
        if self.obj_type == 'cloud':
            self.obj.cloud_list = grp_list
            self.counts[self.obj_type] = len(self.obj.cloud_list)
            self.obj.group_types.append(self.obj_type)

        memlog('Found %d valid %s, loaded indexes'%(self.counts[self.obj_type],group_types[self.obj_type]))


    def load_fof6dfile(self):
        import os
        from yt.funcs import mylog
        fof6d_file = self.obj._kwargs['fof6d_file']
        if os.path.isfile(fof6d_file):
            mylog.info('Reading galaxy membership from fof6d file %s'%fof6d_file)
        else:
            mylog.info('fof6d file %s not found! Running fof6d' % fof6d_file)
            return True
        hf = h5py.File(fof6d_file,'r')
        self.tags_fof6d = np.asarray(hf['fof6d_tags'])
        self.group_parents = np.asarray(hf['group_parents'])
        hf.close()
        if len(self.tags_fof6d)!=self.nparttot:
            mylog.warning('fof6d_file invalid! len(fof6d_tags) does not match length of particle list in halos (%d != %d) and/or len(group_parents) does not match length of halo list (%d != %d) -- RUNNING FOF6D'%(len(self.tags_fof6d),self.nparttot,len(self.group_parents),self.counts[self.obj_type]))
            return True
        return False

    def save_fof6dfile(self):
        if 'fof6d_file' not in self.obj._kwargs or self.obj._kwargs['fof6d_file'] is None:
            return
        fof6d_file = self.obj._kwargs['fof6d_file']
        memlog('Writing fof6d info to %s' % fof6d_file)
        all_tags = self.tags_fof6d
        group_parents = self.group_parents
        with h5py.File(fof6d_file,'w') as hf:  # overwrites existing fof6d group file
            hf.create_dataset('fof6d_tags',data=all_tags, compression=1)
            hf.create_dataset('group_parents',data=group_parents, compression=1)
            hf.close()

    '''
    def link_to_parent(self,parent=None):
        # Set up cross-matching between parent and children
        if parent == None: return
        for ih in range(len(parent.grouplist)):
            group_nums = np.unique(parent.tags_fof6d[ih])
            if parent.obj_type == 'halo':
                if self.obj_type == 'galaxy':
                    self.obj.halo_list[ih].galaxy_index_list = np.empty(0, dtype=np.int32)
                elif self.obj_type == 'cloud':
                    self.obj.halo_list[ih].cloud_index_list = np.empty(0, dtype=np.int32)
            if parent.obj_type == 'galaxy':
                self.obj.galaxy_list[ih].cloud_index_list = np.empty(0, dtype=np.int32)
            for igrp in group_nums[group_nums>=0]:
                if parent.obj_type == 'halo':
                    mygrp.parent_halo_index = ih
                    if self.obj_type == 'galaxy':
                        self.obj.halo_list[ih].galaxy_index_list = np.append(self.obj.halo_list[ih].galaxy_index_list,self.counts[target_type])
                    if self.obj_type == 'cloud':
                        self.obj.halo_list[ih].cloud_index_list = np.append(self.obj.halo_list[ih].cloud_index_list,self.counts[target_type])
    '''


#=========================================================
# 6DFOF ROUTINES
#=========================================================

def find_bins(sorted_list,last_value):
    # find particle indexes in sorted halos list (from alimanfoo/find_runs.py)
    loc_run_start = np.empty(len(sorted_list), dtype=bool)
    loc_run_start[0] = True
    np.not_equal(sorted_list[:-1], sorted_list[1:], out=loc_run_start[1:])
    sorted_bins = np.nonzero(loc_run_start)[0]
    sorted_bins = np.append(sorted_bins,last_value)
    return sorted_bins

def setup_indexes(self,halo_indexes):
    ''' Collect indexes of eligible gas/stars/bh/dust from among particles in a given halo
    NOTE: The indexes returned are tagged to the particles within a given halo '''
    from caesar.property_manager import ptype_ints
    # first quickly check if there are enough stars
    my_ptype = self.obj.data_manager.ptype[halo_indexes]
    star_indexes = halo_indexes[my_ptype == ptype_ints['star']]
    if len(star_indexes) < self.minstars:
        return np.zeros(1,dtype=np.int32)
    # Optional: validate STAR mapping consistency (concat -> selected -> concat round-trip)
    try:
        import os as _os
        if _os.environ.get('CAESAR_VALIDATE_STAR_MAP', '0') == '1':
            ssel = self.obj.data_manager.concat_to_selected('star', star_indexes) if star_indexes.size > 0 else np.array([], dtype=np.int64)
            rt = self.obj.data_manager.selected_to_concat('star', ssel) if ssel.size > 0 else np.array([], dtype=np.int64)
            ok = (rt.size == star_indexes.size) and np.array_equal(np.sort(rt), np.sort(star_indexes))
            from yt.funcs import mylog
            mylog.debug('fof6d star-map check: concat=%d selected=%d roundtrip_ok=%s', int(star_indexes.size), int(ssel.size), str(ok))
            if _os.environ.get('CAESAR_ASSERT_STAR_MAP', '0') == '1' and star_indexes.size > 0 and (ssel.size == 0 or not ok):
                raise AssertionError('Star index mapping inconsistency: concat > 0 but selected empty or round-trip failed')
    except Exception:
        pass
    # collect particles for fof6d: first apply dense gas cut
    gas_indexes = halo_indexes[my_ptype == ptype_ints['gas']]
    # Map concatenated gas indices to selected indices using DataManager utility
    if gas_indexes.size > 0:
        gpos = self.obj.data_manager.concat_to_selected('gas', gas_indexes)
        # Assert: if gas is present in the concatenated slice but mapping returns empty
        # this indicates a mapping/selection inconsistency.
        import os as _os
        if _os.environ.get('CAESAR_ASSERT_ELIGIBLE_GAS', '0') == '1' and gpos.size == 0:
            raise AssertionError('Eligible gas present in halo slice but selected index map is empty')
        # Optional validation: round-trip mapping and gating statistics
        if _os.environ.get('CAESAR_VALIDATE_GAS_MAP', '0') == '1':
            try:
                from yt.funcs import mylog
                gas_concat_cnt = int(gas_indexes.size)
                gas_sel_cnt = int(gpos.size)
                rt = self.obj.data_manager.selected_to_concat('gas', gpos)
                rt_ok = (rt.size == gas_indexes.size) and np.array_equal(np.sort(rt), np.sort(gas_indexes))
                mylog.debug('fof6d gas-map check: concat=%d selected=%d roundtrip_ok=%s', gas_concat_cnt, gas_sel_cnt, str(rt_ok))
                if _os.environ.get('CAESAR_ASSERT_GAS_MAP', '0') == '1' and gas_concat_cnt > 0 and (gas_sel_cnt == 0 or not rt_ok):
                    raise AssertionError('Gas index mapping inconsistency: concat > 0 but selected empty or round-trip failed')
            except Exception:
                pass
        gtemp = self.obj.data_manager.gT[gpos]
        gsfr = self.obj.data_manager.gsfr[gpos]
        gnh = self.obj.data_manager.gnh[gpos]
    else:
        gtemp = self.obj.data_manager.gT[:0]
        gsfr = self.obj.data_manager.gsfr[:0]
        gnh = self.obj.data_manager.gnh[:0]
    select_dense_gas = self.dense_crit(gnh, gtemp, gsfr)
    # (gas-gate per-halo logging disabled)
    dense_indexes = gas_indexes[select_dense_gas]
    # add in other particle types
    bh_indexes = halo_indexes[my_ptype == ptype_ints['bh']]
    dust_indexes = halo_indexes[my_ptype == ptype_ints['dust']]
    # Optional: validate DM mapping as well (it is not used for eligible set but useful to detect systemic issues)
    try:
        import os as _os
        if _os.environ.get('CAESAR_VALIDATE_DM_MAP', '0') == '1' and 'dm' in self.obj.data_manager.ptypes:
            dm_indexes = halo_indexes[my_ptype == ptype_ints['dm']]
            dsel = self.obj.data_manager.concat_to_selected('dm', dm_indexes) if dm_indexes.size > 0 else np.array([], dtype=np.int64)
            rt = self.obj.data_manager.selected_to_concat('dm', dsel) if dsel.size > 0 else np.array([], dtype=np.int64)
            ok = (rt.size == dm_indexes.size) and np.array_equal(np.sort(rt), np.sort(dm_indexes))
            from yt.funcs import mylog
            mylog.debug('fof6d dm-map check: concat=%d selected=%d roundtrip_ok=%s', int(dm_indexes.size), int(dsel.size), str(ok))
            if _os.environ.get('CAESAR_ASSERT_DM_MAP', '0') == '1' and dm_indexes.size > 0 and (dsel.size == 0 or not ok):
                raise AssertionError('DM index mapping inconsistency: concat > 0 but selected empty or round-trip failed')
    except Exception:
        pass
    all_indexes = np.concatenate((dense_indexes,star_indexes,bh_indexes,dust_indexes),axis=None).astype(np.int32)
    # concatenate everything in the proper order and return
    return all_indexes

def fof6d_halo(nparthalo,npart,pos,vel,minstars,Lbox,fof_LL,vel_LL,kerneltab):
    ''' Routine to find galaxies within a given halo using fof6d '''

    #initialize fof6d
    fof6d_tags = np.zeros(npart,dtype=np.int64)-1  # default is that no particles are in galaxies
    if npart <= minstars:  # no possible valid galaxies; we're done
        return fof6d_tags, 0
    mypos = np.copy(pos)
    myvel = np.copy(vel)
    mypos = mypos.T # transpose since fof6d routines expect [npart,ndim]
    myvel = myvel.T

    # run fof6d
    groups = [[0,npart]]  # group to process has the entire list of particles in halo
    pindex = np.arange(npart,dtype=np.int32) # index to keep track of particle sorting
    myhaloID = np.zeros(npart,dtype=np.int32)  # fof6d doing one halo at a time; arbitrarily assign to halo 0
    for idir in range(len(mypos)):  # sort in each direction, find groups within sorted list
        if len(groups) > 0: groups = fof_sorting_old(groups,mypos,myvel,myhaloID,pindex,fof_LL,Lbox,idir,mingrp=minstars)
    if len(groups) == 0:
        return fof6d_tags,0  # found no valid groups after sorting
    fof6d_results = [None]*len(groups)
    for igrp in range(len(groups)):
        if groups[igrp][1]-groups[igrp][0] < minstars: continue
        fof6d_results[igrp] = fof6d_main(igrp,groups,mypos.T[groups[igrp][0]:groups[igrp][1]],myvel.T[groups[igrp][0]:groups[igrp][1]],kerneltab,0.,Lbox,minstars,fof_LL,vel_LL)

    # insert galaxy IDs into particle lists
    nfof = 0
    galindex = np.zeros(npart,dtype=int)-1
    for igrp in range(len(groups)):
        if fof6d_results[igrp] is None: continue  # no valid galaxies
        istart = groups[igrp][0]  # starting particle index for group igrp
        iend = groups[igrp][1]
        galindex[istart:iend] = np.where(fof6d_results[igrp][1]>=0,fof6d_results[igrp][1]+nfof,-1)  # for particles in galaxies, increment galaxy ID with counter (nfof)
        nfof += fof6d_results[igrp][0]

    # reset back into original particle order and collect tags for particles in this halo
    for i in range(npart):
        fof6d_tags[pindex[i]] = galindex[i]
    # returns the group to which each particle belongs (-1 if not in group)
    return fof6d_tags,nfof

def fof6d_main(igrp,groups,poslist,vellist,kerneltab,t0,Lbox,mingrp,fof_LL,vel_LL=None,nfof=0):
    # find neighbors of all particles within fof_LL
    istart = groups[igrp][0]  # starting particle index for group igrp
    iend = groups[igrp][1]
    nactive = iend-istart
    if nactive < mingrp: return [0,[]]
    neigh = NearestNeighbors(radius=fof_LL)  # set up neighbor finder
    neigh.fit(poslist)  # do neighbor finding
    nlist = neigh.radius_neighbors(poslist)  # get neighbor properties (radii, indices)

    # compute velocity criterion for neighbors, based on local velocity dispersion
    if vel_LL is not None:
        LLinv = 1./fof_LL
        sigma = np.zeros(nactive)
        siglist = []  # list of boolean arrays storing whether crit is satisfied for each neighbor pair
        # compute local velocity dispersion from neighbors
        for i in range(nactive):
            ngblist = nlist[1][i]  # list of indices of neighbors
            rlist = nlist[0][i]  # list of radii of neighbors
            # compute kernel-weighted velocity dispersion
            wt = kernel(rlist*LLinv,kerneltab)
            dv = np.linalg.norm(vellist[ngblist]-vellist[i],axis=1)
            sigma[i] = np.sqrt(np.sum(wt*dv*dv)/np.sum(wt))
            siglist.append((dv <= vel_LL*sigma[i]))
    else:
        # if velocity criterion not used, then all particles satisfy it by default
        siglist = []  # list of boolean arrays storing whether vel disp crit is satisfied
        for i in range(nactive):
            ngbnum = len(nlist[1][i])
            sigs = np.ones(ngbnum,dtype=bool)  # array of True's
            siglist.append(sigs)

    # determine counts within fof_LL, set up ordering of most dense to least
    ncount = np.zeros(nactive,dtype=int)
    for i in range(len(ncount)):
        ncount[i] = len(nlist[1][i])  # count number of neighbors for each particle
    dense_order = np.argsort(-ncount)  # find ordering of most dense to least

    # main loop to do FOF
    galind = np.zeros(nactive,dtype=int)-1
    linked = []
    galcount = 0
    for ipart in range(nactive):
        densest = dense_order[ipart]  # get next densest particle
        galind_ngb = galind[nlist[1][densest]]  # indices of neighbors' galaxies
        galind_ngb = np.where(siglist[densest],galind_ngb,-1)  # apply velocity criterion here
        if len(galind_ngb[galind_ngb>=0]) > 0:  # if it already has neighbors (incl itself) in a galaxy...
            galmin = np.unique(galind_ngb[galind_ngb>=0])  # find sorted, unique indices of neighbors' gals
            galind[nlist[1][densest]] = min(galmin)  # put all neighbors in lowest-# galaxy
            for i in range(1,len(galmin)):  # link all other galaxies to lowest
                if linked[galmin[i]]==-1:
                    linked[galmin[i]] = min(galmin)  # link all other galaxies to lowest index one
                else:
                    linked[galmin[i]] = min(linked[galmin[i]],min(galmin))  # connect all other galaxies to lowest index
        else:  # it has no neighbors in a galaxy, so create a new one
            galind[nlist[1][densest]] = galcount  # put all neighbors in a new galaxy
            linked.append(galcount) # new galaxy is linked to itself
            galcount += 1

    # handle linked galaxies by resetting indices of their particles to its linked galaxy
    for i in range(galcount-1,-1,-1):
        if linked[i] != i:
            assert linked[i]<i,'Trouble: mis-ordered linking %d > %d'%(i,linked[i])
            for j in range(iend-istart):
                if galind[j] == i: galind[j] = linked[i]

    # assign indices of particles to FOF groups having more than mingrp particles
    pcount,bin_edges = np.histogram(galind,bins=galcount)  # count particles in each galaxy
    for i in range(iend-istart):  # set indices of particles in groups with <mingrp members to -1
        if pcount[galind[i]] < mingrp: galind[i] = -1
    if len(galind[galind>=0])==0: return 0,galind  # if there are no valid groups left, return
    galind_unique = np.unique(galind[galind>=0])  # find unique groups
    galind_inv = np.zeros(max(galind_unique)+1,dtype=int)
    for i in range(len(galind_unique)):
        galind_inv[galind_unique[i]] = i  # create mapping from original groups to unique set
    for i in range(iend-istart):
        if galind[i]>=0: galind[i] = galind_inv[galind[i]] # re-assign group indices sequentially
    galcount = max(galind)+1
    #raw_input('press ENTER to continue')

    '''
    # check: are there groups that are too large?
    for i in range(galcount):
        npgal = len(galind[galind==i])
        galpos = np.array([poslist[j] for j in range(len(poslist)) if galind[j]==i])
        galpos = galpos.T
        #print npgal,galpos
        galsize = np.array([max(galpos[0])-min(galpos[0]),max(galpos[1])-min(galpos[1]),max(galpos[2])-min(galpos[2])])
        galsize = np.where(galsize>Lbox/2,Lbox-galsize,galsize)
        toolarge = 300
        if nfof < 10 and nfof > 0: print 'in fof6d:',nfof+i,i,npgal,np.mean(galpos[0]),np.mean(galpos[1]),np.mean(galpos[2]),galsize
        if galsize[0]>toolarge or galsize[1]>toolarge or galsize[2]>toolarge: print 'Too large?',igrp,iend-istart,galsize,galpos.T
    '''

    # Compile result to return: number of new groups found, and a galaxy index for particles from istart:iend
    result = [galcount]
    result.append(galind)
    #progress_bar(1.*groups[igrp][1]/groups[len(groups)-1][1],barLength=50,t=time.time()-t0)

    return result

def fof_sorting(groups,pos,vel,haloID,pindex,fof_LL,Lbox,mingrp,idir):
    oldgroups = groups[:]  # stores the groups found from the previous sorting direction
    npart = len(pindex)
    groups = [[0,npart]]  # (re-)initialize the group as containing all particles
    grpcount = 0
    for igrp in range(len(oldgroups)):  # loop over old groups, sort within each, find new groups
        # sort particles within group in given direction
        istart = oldgroups[igrp][0]  # starting particle index for group igrp
        iend = oldgroups[igrp][1]
        sort_parts(pos,vel,haloID,pindex,istart,iend,idir,'pos')  # sort group igrp in given direction
        # create new groups by looking for breaks of dx[idir]>fof_LL between particle positions
        oldpos = pos[idir][istart]
        groups[grpcount][0] = istart  # set start of new group to be same as for old group
        for i in range(istart,iend):
            if periodic(pos[idir][i],oldpos,Lbox) > fof_LL or i == iend-1:
                groups[grpcount][1] = i  # set end of old group to be i
                groups.append([i,i])  # add new group, starting at i; second index is a dummy that will be overwritten when the end of the group is found
                grpcount += 1
            oldpos = pos[idir][i]
    assert grpcount>0,'fof6d : Found no groups or unable to separate groups via sorting; exiting. %d %d %d %d'%(idir,len(groups),len(oldgroups),npart)
    # Remove groups that have less than mingrp particles (gas+star)
    oldgroups = groups[:]
    groups = []
    for igrp in range(len(oldgroups)):
        istart = oldgroups[igrp][0]
        iend = oldgroups[igrp][1]
        if iend-istart >= mingrp: groups.append(oldgroups[igrp])
    return groups

# progress bar, from https://stackoverflow.com/questions/3160699/python-progress-bar
def progress_bar(progress,barLength=10,t=None):
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done!\r\n"
    block = int(round(barLength*progress))
    if t is None: text = "\r[{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), np.round(progress*100,2), status)
    else: text = "\r[{0}] {1}% [t={2} s] {3}".format( "#"*block + "-"*(barLength-block), np.round(progress*100,2), np.round(t,2), status)
    sys.stdout.write(text)
    sys.stdout.flush()

# set up kernel table
def kernel_table(fof_LL,ntab=1000):
    kerneltab = np.zeros(ntab+1)
    hinv = 1./fof_LL
    norm = 0.31832*hinv**3
    for i in range(ntab):
        r = 1.*i/ntab
        q = 2*r*hinv
        if q > 2: kerneltab[i] = 0.0
        elif q > 1: kerneltab[i] = 0.25*norm*(2-q)**3
        else: kerneltab[i] = norm*(1-1.5*q*q*(1-0.5*q))
    return kerneltab

# kernel table lookup
def kernel(r_over_h,kerneltab):
    ntab = len(kerneltab)-1
    rtab = ntab*r_over_h+0.5
    itab = rtab.astype(int)
    return kerneltab[itab]


#=========================================================
# DEFUNCT FOF6D ROUTINES
#=========================================================

# 6-D FOF.
# Input is a snapshot file containing gas, stars, and BHs.
# Snapshot must contain a HaloID putting each particle in a halo (0=not in halo).
# First groups particles via an approximate FOF using  via quicksorts in successive directions.
# Then does a full 6-D FOF search on the approximate FOF groups
# Outputs SKID-format .grp binary file containing Npart, then galaxy IDs of all particles
#
# Romeel Dave, 25 Feb 2019

from astropy import constants as const

#BASEDIR = sys.argv[1]
#snapnum = sys.argv[2]

'''
MODEL = sys.argv[1]
WIND = sys.argv[2]
SNAP = int(sys.argv[3])
if len(sys.argv)==5: nproc = int(sys.argv[4])
else: nproc = 1
'''
nproc = 1

#BASEDIR = '/cosma/home/dc-dave2/data/%s/%s'%(MODEL,WIND)

# FOF options
mingrp = 16
LL_factor = 0.02  # linking length is this times the mean interparticle spacing
vel_LL = 1.0  # velocity space linking length factor, multiplies local velocity dispersion

#=========================================================
# MISCELLANEOUS ROUTINES
#=========================================================

# Loads gas, star, BH information from snapshot.  Requires HaloID for all particles.
def loadsnap(snap,t0):
    import pygadgetreader as pygr
    redshift = pygr.readheader(snap,'redshift')
    h = pygr.readheader(snap,'h')

    # get gas Halo IDs so we can select only gas particles in halos
    ghalo = np.array(pygr.readsnap(snap,'HaloID','gas'),dtype=int)  # Halo ID of gas; 0=not in halo
    ngastot = len(ghalo)
    gas_select = (ghalo>0)
    ngastot = np.uint64(len(ghalo))
    pindex = np.arange(ngastot,dtype=np.uint64)  # keep an index for the original order of the particles

    # Load in gas info for selecting gas particles
    gnh = pygr.readsnap(snap,'rho','gas',units=1)[gas_select]*h*h*0.76*(1+redshift)**3/const.m_p.to('g').value   # number density in phys H atoms/cm^3
    gsfr = pygr.readsnap(snap,'sfr','gas',units=1)[gas_select]
    gtemp = pygr.readsnap(snap,'u','gas',units=1)[gas_select]  # temperature in K
    # Apply additional selection criteria on gas
    dense_gas_select = ((gnh>0.13)&((gtemp<1.e5)|(gsfr>0))) # dense, cool, or SF gas only
    gas_select[gas_select>0] = (gas_select[gas_select>0] & dense_gas_select)

    # load in selected gas
    gpos = pygr.readsnap(snap,'pos','gas',units=1)[gas_select]/h  # positions in ckpc
    gvel = pygr.readsnap(snap,'vel','gas',units=1,suppress=1)[gas_select] # vel in physical km/s
    ghalo = np.array(pygr.readsnap(snap,'HaloID','gas'),dtype=int)[gas_select]  # Halo ID of gas; 0=not in halo

    # load in all stars+BHs; don't bother with halo selection since most will be in halos anyways
    shalo = np.array(pygr.readsnap(snap,'HaloID','star'),dtype=int)  # Halo ID of gas; 0=not in halo
    star_select = (shalo==2)
    spos = pygr.readsnap(snap,'pos','star',units=1)[star_select]/h  # star positions in ckpc
    svel = pygr.readsnap(snap,'vel','star',units=1,suppress=1)[star_select] # star vels in physical km/s
    shalo = np.array(pygr.readsnap(snap,'HaloID','star'),dtype=int)[star_select]  # Halo ID of stars
    nstartot = np.uint64(len(spos))
    try:
        bhhalo = np.array(pygr.readsnap(snap,'HaloID','bndry'),dtype=int)  # Halo ID of gas; 0=not in halo
        bh_select = (bhhalo==2)
        bpos = pygr.readsnap(snap,'pos','bndry',units=1)[bh_select]/h  # BH positions in ckpc
        bvel = pygr.readsnap(snap,'vel','bndry',units=1,suppress=1)[bh_select] # BH vels in physical km/s
        bhalo = np.array(pygr.readsnap(snap,'HaloID','bndry'),dtype=int)[bh_select]  # Halo ID of BHs
        nbhtot = np.uint64(len(bpos))
    except:
        print('fof6d : Creating one fake BH particle at origin (not in a halo) to avoid crash.')
        bpos = [[0,0,0]]
        bvel = [[0,0,0]]
        bhalo = [0]
        nbhtot = 0
    # set up combined arrays for positions and velocities, along with indexing
    pos = np.vstack((gpos,spos,bpos)).T  # transpose gives pos[0] as list of x positions, pos[1] as y, etc
    vel = np.vstack((gvel,svel,bvel)).T  # same for vel
    haloID = np.concatenate((ghalo,shalo,bhalo))  # compile list of halo IDs
    pindex = np.concatenate((pindex[gas_select],np.arange(ngastot,ngastot+nstartot+nbhtot,dtype=np.uint64)))
    print('fof6d_old : Loaded %d (of %d) gas + %d stars + %g bh = %d total particles [t=%.2f s]'%(len(gpos),ngastot,len(spos),len(bpos),len(pindex),time.time()-t0))
    return pos,vel,haloID,pindex,ngastot,nstartot,nbhtot,gas_select

# Returns array galindex to original particles order in snapshot, and splits into separate arrays
# for gas, stars, and BHs.  Returns a list of galaxy IDs for all gas, stars, and BHs in
# snapshot, with galaxy ID = -1 for particles not in a fof6d galaxy
def reset_order_old(galindex,pindex,ngas,nstar,nbh,snap):
    import pygadgetreader as pygr
    # create galaxy ID arrays for gas, stars, BHs
    gindex = np.zeros(ngas,dtype=np.int64)-1
    sindex = np.zeros(nstar,dtype=np.int64)-1
    bindex = np.zeros(nbh,dtype=np.int64)-1
    # loop through all searched particles and place galaxy IDs into appropriate array
    for i in range(len(pindex)):
        if pindex[i] < ngas:
            gindex[pindex[i]] = galindex[i]
        elif pindex[i] < ngas+nstar:
            sindex[pindex[i]-ngas] = galindex[i]
        else:
            bindex[pindex[i]-(ngas+nstar)] = galindex[i]

    if False:  # a check for debugging
        ghalo = np.array(pygr.readsnap(snap,'HaloID','gas'),dtype=int)  # Halo ID of gas; 0=not in halo
        shalo = np.array(pygr.readsnap(snap,'HaloID','star'),dtype=int)  # Halo ID of gas; 0=not in halo
        bhalo = np.array(pygr.readsnap(snap,'HaloID','bndry'),dtype=int)  # Halo ID of gas; 0=not in halo
        assert len(ghalo)==len(gindex),'Gas lengths not equal! %d != %d'%(len(ghalo),len(gindex))
        assert len(shalo)==len(sindex),'Star lengths not equal! %d != %d'%(len(shalo),len(sindex))
        assert len(bhalo)==len(bindex),'BH lengths not equal! %d != %d'%(len(bhalo),len(bindex))
        for i in range(len(gindex)):
            if ghalo[i] == 0 and gindex[i] >= 0: sys.exit('Found particle in galaxy but not in halo! i=%d %d %d'%(i,ghalo[i],gindex[i]))
        for i in range(len(sindex)):
            if shalo[i] == 0 and sindex[i] >= 0: sys.exit('Found particle in galaxy but not in halo! i=%d %d %d'%(i,shalo[i],sindex[i]))
        for i in range(len(bindex)):
            if bhalo[i] == 0 and bindex[i] >= 0: sys.exit('Found particle in galaxy but not in halo! i=%d %d %d'%(i,bhalo[i],bindex[i]))

    return gindex,sindex,bindex

def periodic(x1,x2,L):  # periodic distance between scalars x1 and k2
    dx = x1-x2
    if dx>0.5*L: return L-dx
    else: return dx


#=========================================================
# 6DFOF ROUTINES
#=========================================================

def fof6d_old(igrp,groups,poslist,vellist,kerneltab,t0,Lbox,fof_LL,vel_LL=None,nfof=0):
    # find neighbors of all particles within fof_LL
    istart = groups[igrp][0]  # starting particle index for group igrp
    iend = groups[igrp][1]
    nactive = iend-istart
    if nactive < mingrp: return [0,[]]
    neigh = NearestNeighbors(radius=fof_LL)  # set up neighbor finder
    neigh.fit(poslist)  # do neighbor finding
    nlist = neigh.radius_neighbors(poslist)  # get neighbor properties (radii, indices)

    # compute velocity criterion for neighbors, based on local velocity dispersion
    if vel_LL is not None:
        LLinv = 1./fof_LL
        sigma = np.zeros(nactive)
        siglist = []  # list of boolean arrays storing whether crit is satisfied for each neighbor pair
        # compute local velocity dispersion from neighbors
        for i in range(nactive):
            ngblist = nlist[1][i]  # list of indices of neighbors
            rlist = nlist[0][i]  # list of radii of neighbors
            # compute kernel-weighted velocity dispersion
            wt = kernel(rlist*LLinv,kerneltab)
            dv = np.linalg.norm(vellist[ngblist]-vellist[i],axis=1)
            sigma[i] = np.sqrt(np.sum(wt*dv*dv)/np.sum(wt))
            siglist.append((dv <= vel_LL*sigma[i]))
    else:
        # if velocity criterion not used, then all particles satisfy it by default
        siglist = []  # list of boolean arrays storing whether vel disp crit is satisfied
        for i in range(nactive):
            ngbnum = len(nlist[1][i])
            sigs = np.ones(ngbnum,dtype=bool)  # array of True's
            siglist.append(sigs)

    # determine counts within fof_LL, set up ordering of most dense to least
    ncount = np.zeros(nactive,dtype=int)
    for i in range(len(ncount)):
        ncount[i] = len(nlist[1][i])  # count number of neighbors for each particle
    dense_order = np.argsort(-ncount)  # find ordering of most dense to least

    # main loop to do FOF
    galind = np.zeros(nactive,dtype=int)-1
    linked = []
    galcount = 0
    for ipart in range(nactive):
        densest = dense_order[ipart]  # get next densest particle
        galind_ngb = galind[nlist[1][densest]]  # indices of neighbors' galaxies
        galind_ngb = np.where(siglist[densest],galind_ngb,-1)  # apply velocity criterion here
        if len(galind_ngb[galind_ngb>=0]) > 0:  # if it already has neighbors (incl iteself) in a galaxy...
            galmin = np.unique(galind_ngb[galind_ngb>=0])  # find sorted, unique indices of neighbors' gals
            galind[nlist[1][densest]] = min(galmin)  # put all neighbors in lowest-# galaxy
            for i in range(1,len(galmin)):  # link all other galaxies to lowest
                if linked[galmin[i]]==-1:
                    linked[galmin[i]] = min(galmin)  # link all other galaxies to lowest index one
                else:
                    linked[galmin[i]] = min(linked[galmin[i]],min(galmin))  # connect all other galaxies to lowest index
        else:  # it has no neighbors in a galaxy, so create a new one
            galind[nlist[1][densest]] = galcount  # put all neighbors in a new galaxy
            linked.append(galcount) # new galaxy is linked to itself
            #print 'particle %d creating galaxy %d with neighbors %s'%(densest,galcount,nlist[1][densest])
            galcount += 1

    '''
    # Do final linking
    nreset = 1
    while nreset > 0:
        nreset = 0
        for ipart in range(len(ncount)):
            for k in range(len(nlist[1][ipart])):
                j = nlist[1][ipart][k]
                if galind[ipart] > galind[j]:
                    linked[galind[ipart]] = galind[j]
                    galind[ipart] = galind[j]
                    #print nlist[0][ipart]
                    #print nlist[1][ipart]
                    #print 'resetting part %d group to %d to match %d'%(ipart,galind[ipart],j)
                    nreset += 1
                    #print 'Trouble: Part %d (%s) in gal %d has neighbor %d (%d/%d) (%s) in gal %d, r=%g %d %d'%(ipart,poslist[ipart],galind[ipart],j,k,len(nlist[1][ipart]),poslist[j],galind[j],nlist[0][ipart][k],linked[galind[ipart]],linked[galind[j]])
                    #raw_input("Press Enter to continue ...")
    '''

    '''
    for ipart in range(nactive):  # check that objects don't have neighbors that should have been linked but aren't
        for k in range(len(nlist[1][ipart])):
            j = nlist[1][ipart][k]
            if galind[ipart] != galind[j]:
                print 'Trouble: Part %d in gal %d has neighbor %d (%d/%d) in gal %d, r=%g %d %d'%(ipart,galind[ipart],j,k,len(nlist[1][ipart]),galind[j],nlist[0][ipart][k],linked[galind[ipart]],linked[galind[j]])
    '''

    # handle linked galaxies by resetting indices of their particles to its linked galaxy
    for i in range(galcount-1,-1,-1):
        if linked[i] != i:
            assert linked[i]<i,'Trouble: mis-ordered linking %d > %d'%(i,linked[i])
            for j in range(iend-istart):
                if galind[j] == i: galind[j] = linked[i]

    # assign indices of particles to FOF groups having more than mingrp particles
    pcount,bin_edges = np.histogram(galind,bins=galcount)  # count particles in each galaxy
    for i in range(iend-istart):  # set indices of particles in groups with <mingrp members to -1
        if pcount[galind[i]] < mingrp: galind[i] = -1
    if len(galind[galind>=0])==0: return 0,galind  # if there are no valid groups left, return
    galind_unique = np.unique(galind[galind>=0])  # find unique groups
    galind_inv = np.zeros(max(galind_unique)+1,dtype=int)
    for i in range(len(galind_unique)):
        galind_inv[galind_unique[i]] = i  # create mapping from original groups to unique set
    for i in range(iend-istart):
        if galind[i]>=0: galind[i] = galind_inv[galind[i]] # re-assign group indices sequentially
    galcount = max(galind)+1
    #raw_input('press ENTER to continue')

    '''
    # check: are there groups that are too large?
    for i in range(galcount):
        npgal = len(galind[galind==i])
        galpos = np.array([poslist[j] for j in range(len(poslist)) if galind[j]==i])
        galpos = galpos.T
        #print npgal,galpos
        galsize = np.array([max(galpos[0])-min(galpos[0]),max(galpos[1])-min(galpos[1]),max(galpos[2])-min(galpos[2])])
        galsize = np.where(galsize>Lbox/2,Lbox-galsize,galsize)
        toolarge = 300
        if nfof < 10 and nfof > 0: print 'in fof6d:',nfof+i,i,npgal,np.mean(galpos[0]),np.mean(galpos[1]),np.mean(galpos[2]),galsize
        if galsize[0]>toolarge or galsize[1]>toolarge or galsize[2]>toolarge: print 'Too large?',igrp,iend-istart,galsize,galpos.T
    '''

    # Compile result to return: number of new groups found, and a galaxy index for particles from istart:iend
    result = [galcount]
    result.append(galind)
    progress_bar(1.*groups[igrp][1]/groups[len(groups)-1][1],barLength=50,t=time.time()-t0)

    return result

#=========================================================
# FOFRAD ROUTINES
#=========================================================

# sort particles by position in direction idir0, for particles from istart:iend
def sort_parts(pos,vel,haloID,pindex,istart,iend,idir0,key='pos'):
    if key == 'pos':
        sort_ind = np.argsort(pos[idir0][istart:iend])  # sort in desired direction
    elif key == 'haloID':
        sort_ind = np.argsort(haloID[istart:iend])  # sort by halo ID (idir0 is irrelevant)
    idir1 = (idir0+1)%3  # these are the other two directions
    idir2 = (idir0+2)%3
    pos[idir0][istart:iend] = pos[idir0][istart:iend][sort_ind]  # keep all arrays likewise sorted
    pos[idir1][istart:iend] = pos[idir1][istart:iend][sort_ind]
    pos[idir2][istart:iend] = pos[idir2][istart:iend][sort_ind]
    vel[idir0][istart:iend] = vel[idir0][istart:iend][sort_ind]
    vel[idir1][istart:iend] = vel[idir1][istart:iend][sort_ind]
    vel[idir2][istart:iend] = vel[idir2][istart:iend][sort_ind]
    haloID[istart:iend] = haloID[istart:iend][sort_ind]
    pindex[istart:iend] = pindex[istart:iend][sort_ind]  # keep particle indices in the same order

def fof_sorting_old(groups,pos,vel,haloID,pindex,fof_LL,Lbox,idir,mingrp=16):
    oldgroups = groups[:]  # stores the groups found from the previous sorting direction
    npart = len(pindex)
    groups = [[0,npart]]  # (re-)initialize the group as containing all particles
    grpcount = 0
    for igrp in range(len(oldgroups)):  # loop over old groups, sort within each, find new groups
        # sort particles within group in given direction
        istart = oldgroups[igrp][0]  # starting particle index for group igrp
        iend = oldgroups[igrp][1]
        sort_parts(pos,vel,haloID,pindex,istart,iend,idir,'pos')  # sort group igrp in given direction
        # create new groups by looking for breaks of dx[idir]>fof_LL between particle positions
        oldpos = pos[idir][istart]
        groups[grpcount][0] = istart  # set start of new group to be same as for old group
        for i in range(istart,iend):
            if periodic(pos[idir][i],oldpos,Lbox) > fof_LL or i == iend-1:
                groups[grpcount][1] = i  # set end of old group to be i
                groups.append([i,i])  # add new group, starting at i; second index is a dummy that will be overwritten when the end of the group is found
                grpcount += 1
            oldpos = pos[idir][i]
    #assert grpcount>0,'fof6d : Unable to separate groups via sorting; exiting.'
    # Remove groups that have less than mingrp particles (gas+star)
    oldgroups = groups[:]
    groups = []
    for igrp in range(len(oldgroups)):
        istart = oldgroups[igrp][0]
        iend = oldgroups[igrp][1]
        if iend-istart >= mingrp: groups.append(oldgroups[igrp])
    return groups

def fofrad_old(snap,nproc,mingrp,LL_factor,vel_LL):
    import pygadgetreader as pygr
    t0 = time.time()
    Lbox = pygr.readheader(snap,'boxsize')
    h = pygr.readheader(snap,'h')
    Lbox = Lbox/h  # to kpc
    #Omega = pygr.readheader(snap,'O0')
    #Lambda = pygr.readheader(snap,'Ol')
    n_side = int(pygr.readheader(snap,'dmcount')**(1./3.)+0.5)
    MIS = Lbox/n_side
    fof_LL = LL_factor*MIS
    print('fof6d : Guessing %d particles per side, fof_LL=%g ckpc' % (n_side,fof_LL))
    kerneltab = kernel_table(fof_LL)

    # Load particles positions and velocities from snapshot
    pos,vel,haloID,pindex,ngastot,nstartot,nbhtot,gas_select = loadsnap(snap,t0)

    # initialize groups as containing all the particles in each halo
    npart = len(pindex)
    sort_parts(pos,vel,haloID,pindex,0,npart,0,'haloID')
    groups = []
    hstart = 0
    for i in range(1,len(haloID)):  # set up groups[]
        if haloID[i] == 0:
            hstart = i
            continue
        if haloID[i]>haloID[i-1] and haloID[i-1]>0:
            if i-hstart>=mingrp: groups.append([hstart,i])
            hstart = i
    '''
    # use all particles, not just ones in halos
    groups = [[0,npart]]  # initial group has the entire list of particles
    '''

    # within each group (i.e. halo), create sub-groups of particles via directional sorting, ala FOFRAD
    print('fof6d_old : FOF via sorting beginning with %d halos/groups [t=%.2f s]'%(len(groups),time.time()-t0))
    for idir in range(len(pos)):  # sort in each direction, find groups within sorted list
        groups = fof_sorting_old(groups,pos,vel,haloID,pindex,fof_LL,Lbox,idir)
        print('fof6d_old : Axis %d approx FOF via sorting found %d groups [t=%.2f s]'%(idir,len(groups),time.time()-t0))
        if(len(groups)==0): break

    # for each sub-group, do a proper 6D FOF search to find galaxies
    galindex = np.zeros(npart,dtype=int)-1
    results = [None]*len(groups)
    if nproc > 1:
        npart_par = 2000 # optimization: groups with > this # of parts done via Parallel()
        igpar = []
        igser = []
        for igrp in range(len(groups)):
            if groups[igrp][1]-groups[igrp][0] > npart_par:
                igpar.append(igrp)
            else:
                igser.append(igrp)
        igpar = np.array(igpar,dtype=int)
        print('fof6d : Doing %d groups with npart<%d on single core [t=%.2f s]'%(len(igser),npart_par,time.time()-t0))
        for igrp in igser: results[igrp] = fof6d_old(igrp,groups,pos.T[groups[igrp][0]:groups[igrp][1]],vel.T[groups[igrp][0]:groups[igrp][1]],kerneltab,t0,Lbox,fof_LL,vel_LL)
        print('\nfof6d : Doing %d groups with npart>%d on %d cores (progressbar approximate) [t=%.2f s]'%(len(igpar),npart_par,nproc,time.time()-t0))
        if len(igpar)>0:
            from joblib import Parallel, delayed
            results_par = Parallel(n_jobs=nproc)(delayed(fof6d)(igrp,groups,pos.T[groups[igrp][0]:groups[igrp][1]],vel.T[groups[igrp][0]:groups[igrp][1]],kerneltab,t0,Lbox,fof_LL,vel_LL) for igrp in igpar)

        for i in range(len(igpar)): results[igpar[i]] = results_par[i]
    else:
        print('fof6d : Doing %d groups on single core [t=%.2f s]'%(len(groups),time.time()-t0))
        for igrp in range(len(groups)):
            results[igrp] = fof6d_old(igrp,groups,pos.T[groups[igrp][0]:groups[igrp][1]],vel.T[groups[igrp][0]:groups[igrp][1]],kerneltab,t0,Lbox,fof_LL,vel_LL)

    # insert galaxy IDs into particle lists
    nfof = 0
    for igrp in range(len(groups)):
        result = results[igrp]
        istart = groups[igrp][0]  # starting particle index for group igrp
        iend = groups[igrp][1]
        galindex[istart:iend] = np.where(result[1]>=0,result[1]+nfof,-1)  # for particles in groups, increment group ID with counter (nfof)
        #if nfof<1 and len(result[1][result[1]>=0])>0:
        #        print 'inserting IDs',igrp,nfof,result[0],groups[igrp][1]-groups[igrp][0]
        #        for igal in range(result[0]):
        #            print 'galaxies found:',igal+nfof,len(result[1][result[1]==igal]),np.mean(posgrp[result[1]==igal].T[0]),np.mean(posgrp[result[1]==igal].T[1]),np.mean(posgrp[result[1]==igal].T[2])
        #            print len(posgrp[result[1]==igal]),posgrp[result[1]==igal]
        nfof += result[0]

    gindex,sindex,bindex = reset_order_old(galindex,pindex,ngastot,nstartot,nbhtot,snap)

    if 0:
        spos = pygr.readsnap(snap,'pos','star',units=1)/h  # star positions in ckpc
        for ifof in range(nfof):
            sgrp = spos[ifof==sindex]
            sgrp = sgrp.T
            if ifof<2:
                print('final check',ifof,np.mean(sgrp[0]),np.mean(sgrp[1]),np.mean(sgrp[2]),max(sgrp[0])-min(sgrp[0]),max(sgrp[1])-min(sgrp[1]),max(sgrp[2])-min(sgrp[2]))
                print (len(sgrp.T),sgrp.T)

    print('\nfof6d: Found %d galaxies, with %d gas+%d star+%d BH = %d particles [t=%.2f s]'%(nfof,len(gindex[gindex>=0]),len(sindex[sindex>=0]),len(bindex[bindex>=0]),len(gindex[gindex>=0])+len(sindex[sindex>=0])+len(bindex[bindex>=0]),time.time()-t0))

    return gindex,sindex,bindex,t0

#=========================================================
# MAIN DRIVER ROUTINE
#=========================================================

def run_fof_6d(snapfile,mingrp,LL_factor,vel_LL,nproc):

    #pdb.set_trace()
    #snapfile = '%s/snapshot_%03d.hdf5'%(BASEDIR,int(snapnum))
    if not os.path.isfile(snapfile):
        sys.exit('Snapfile %s does not exist'%snapfile)
    else: print('fof6d : Doing snapfile: %s'%snapfile)

# Set up multiprocessing
    import multiprocessing
    if nproc == 0:   # use all available cores
        num_cores = multiprocessing.cpu_count()
    if nproc != 1:   # if multi-core, set up Parallel processing
        num_cores = multiprocessing.cpu_count()
        if nproc < 0: print('fof6d : Using %d cores (all but %d)'%(num_cores+nproc+1,-nproc-1) )
        if nproc > 1:
            print('progen : Using %d of %d cores'%(nproc,num_cores))
        else: print('progen : Using single core')
        if nproc>8: print('fof6d : FYI you are using nproc=%d. nproc>8 tends to give minimal or negative benefit.'%nproc)

    # find friends of friends groups
    gas_index,star_index,bh_index,t0 = fofrad_old(snapfile,nproc,mingrp,LL_factor,vel_LL)   # returns galaxy indices for *all* gas, stars, bh


    #return statements
    nparts = np.array([len(gas_index),len(star_index),len(bh_index)])
    return nparts,gas_index,star_index,bh_index

    '''
    # output csv file to be read into Caesar
    outfile = '%s/Groups/fof6d_%03d.hdf5'%(BASEDIR,int(snapnum))
    with h5py.File(outfile, 'w') as hf:
        nparts = np.array([len(gas_index),len(star_index),len(bh_index)])
        hf.create_dataset('nparts',data=nparts)
        hf.create_dataset('gas_index',data=gas_index)
        hf.create_dataset('star_index',data=star_index)
        hf.create_dataset('bh_index',data=bh_index)
    print('fof6d : Outputted galaxy IDs for gas, stars, and BHs to %s -- FOF6D DONE [t=%.2f s]'%(outfile,time.time()-t0))

   '''
