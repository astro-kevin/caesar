#!/bin/env python

'''
This script reduces a set of snapshots in SNAPLIST for simulation SIM
located in SNAPDIR.  You can also set the number of cores used.

Romeel Davé, 23 Apr 2020
'''

import os
import sys
import yt
import caesar

def reduce(SNAPLIST, SIM, SNAPDIR, CAESARLOC='Groups', FOF6DLOC='Groups', NPROC=16):
    '''
    SNAPLIST: list of the snapshot numbers
    SIM: Sim name, e.g. m50n512, used in snapshot name (assumes Simba naming convention)
    SNAPDIR = Directory where snapshots are located
    CAESARLOC = Subdirectory to write Caesar catalogs
    FOF6DLOC = Subdirectory to look for / write fof6d files
    '''    
    for j in SNAPLIST:
        # path to the file
        SNAP    = '%s/snap_%s_%03d.hdf5' % (SNAPDIR,SIM,j)
        if not os.path.exists(SNAP):
            print(SNAP, "does not exist")
            continue
        CAESARFILE = '%s/%s/%s_%03d.hdf5' % (SNAPDIR, CAESARLOC, SIM, j)
        FOF6DFILE = '%s/%s/fof6d_%s_%03d.hdf5'%(SNAPDIR,FOF6DLOC,SIM,j)
        HALOID = 'snap'  # options atm are 'snap' (Halo ID's from snapshot) or 'fof' (yt's 3DFOF)
        SELFSHIELD   = False  # True=compute self-shielded mass; False= use fHI,fH2 from snap
        if 'fh_qr' in SNAPDIR:  # Mufasa snaps have no HaloId, need self-shielding correction
            HALOID = 'fof'
            SELFSHIELD = True
    
        ds  = yt.load(SNAP)
        obj = caesar.CAESAR(ds)
        redshift = obj.simulation.redshift
        # Always use the modern FOF/SNAP member_search path (FOF_SNAP module).
        obj.member_search(haloid=HALOID, fof6d_file=FOF6DFILE, nproc=NPROC)

        obj.save(CAESARFILE)
