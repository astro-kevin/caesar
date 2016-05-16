import numpy as np
from .group import create_new_group
from .property_getter import get_property, get_particles_for_FOF, get_high_density_gas_indexes
from .property_getter import ptype_ints
from .utils import calculate_local_densities

from yt.extern import six
from yt.extern.tqdm import tqdm
from yt.units.yt_array import uconcatenate, YTArray
from yt.data_objects.octree_subset import YTPositionArray
from yt.utilities.lib.contour_finding import ParticleContourTree
from yt.geometry.selection_routines import AlwaysSelector
#from yt.analysis_modules.halo_finding.rockstar.rockstar_groupies import RockstarGroupiesInterface

def fof(obj, positions, LL):
    pct = ParticleContourTree(LL)

    pos = YTPositionArray(obj.yt_dataset.arr(positions, obj.units['length']))
    #pos = YTPositionArray(pdata['pos'])
    ot  = pos.to_octree()

    #ot  = [c._current_chunk.objs[0] for c in obj._dd.chunks([], 'all')][0]
    #group_tags = pct.identify_contours(ot.oct_handler, ot.domain_ind, pdata['pos'],
    #                                   np.arange(0,len(pdata['pos']),dtype=np.int64),
    #                                   0,0)

    group_tags = pct.identify_contours(
        ot,
        ot.domain_ind(AlwaysSelector(None)),
        positions,
        np.arange(0,len(positions),dtype=np.int64),
        0,0
    )

    """
    ## RS
    ds  = obj.yt_dataset
    rgi = RockstarGroupiesInterface(ds)
    rgi.setup_rockstar(ds.mass_unit * ds.parameters['MassTable'][1], force_res = LL / 0.2 / 50.)
    ind = np.argsort(group_tags)
    print 'running rs'
    
    print pdata.keys()
    #pcounts = rgi.make_rockstar_fof(ind, group_tags, pdata['pos'], pdata['vel'], pdata['mass'], pdata['ptype'])
    pcounts = rgi.make_rockstar_fof(ind, group_tags, pdata['pos'], pdata['vel'])
    print 'pcounts:',pcounts
    #rgi.output_halos()
    halos = rgi.return_halos()
    #import ipdb; ipdb.set_trace()
    return halos
    """
    
    return group_tags



def get_ptypes(obj, find_type):
    ptypes = ['dm','gas','star']
    
    if 'blackholes' in obj._kwargs and obj._kwargs['blackholes']:
        ptypes.append('bh')
    
    if find_type == 'galaxy':
        ptypes.remove('dm')
    if obj._ds_type.grid:
        ptypes.remove('gas')
    return ptypes


def get_mean_interparticle_separation(obj):
    if hasattr(obj, 'mean_interparticle_separation'):
        return obj.mean_interparticle_separation
    
    UT = obj.yt_dataset.time_unit.to('s/h')/obj.yt_dataset.scale_factor
    UL = obj.yt_dataset.length_unit.to('cmcm/h')
    UM = obj.yt_dataset.mass_unit.to('g/h')
    GRAV = obj.yt_dataset.quan(6.672e-8, 'cmcm**3/(g*s**2)')
    
    G = GRAV / UL**3 * UM * UT**2  ## to system units
    Hubble = obj.yt_dataset.quan(3.2407789e-18, 'h/s') * UT

    dmmass = get_property(obj, 'mass', 'dm').to('code_mass')
    ndm    = len(dmmass)
    dmmass = np.sum(dmmass)

    gmass = get_property(obj, 'mass', 'gas').to('code_mass')
    smass = get_property(obj, 'mass', 'star').to('code_mass')
    bmass = np.sum(gmass) + np.sum(smass)

    Om = obj.yt_dataset.cosmology.omega_matter
    Ob = (bmass / (bmass + dmmass) * Om).d
    rhodm = ((Om - Ob) * 3.0 * Hubble**2 / (8.0 * np.pi * G)).d
    rhodm = obj.yt_dataset.quan(rhodm, 'code_mass/code_length**3')

    mips = ((dmmass / ndm / rhodm)**(1./3.)).to(obj.units['length'])

    obj.mean_interparticle_separation = mips
    return obj.mean_interparticle_separation
    
def fubar(obj, find_type, **kwargs):
    # REPLACE find_type with group_type

    LL = get_mean_interparticle_separation(obj) * 0.2

    pos = obj.data_manager.pos

    if find_type == 'galaxy':
        LL *= 0.2

        # here we want to perform FOF on high density gas + stars
        high_rho_indexes = get_high_density_gas_indexes(obj)
        pos  = np.concatenate((
            pos[obj.data_manager.glist][high_rho_indexes],
            pos[obj.data_manager.slist],
            pos[obj.data_manager.bhlist]
        ))        
        
    fof_tags = fof(obj, pos, LL)

    if find_type == 'galaxy':
        gtags = np.full(obj.ngas, -1, dtype=np.int64)
        gtags[high_rho_indexes] = fof_tags[0:len(high_rho_indexes)]
        fof_tags = np.concatenate((gtags,fof_tags[len(high_rho_indexes)::]))
        
    tag_sort = np.argsort(fof_tags)

    unique_groupIDs = np.unique(fof_tags)    
    groupings = {}
    for GroupID in unique_groupIDs:
        if GroupID < 0: continue
        groupings[GroupID] = create_new_group(obj, find_type)

    tags = fof_tags
        
    nparts = len(tags)
    for i in range(0,nparts):
        index = tag_sort[i]
        tag   = tags[index]
        if tag < 0: continue
        groupings[tag]._append_global_index(index)

    for v in tqdm(groupings.itervalues(),
                  total=len(groupings),
                  desc='Processing %s' % find_type):
        v._process_group()

    group_list = []
    for v in six.itervalues(groupings):
        if not v._valid: continue
        group_list.append(v)

    # sort by mass
    group_list.sort(key = lambda x: x.masses['total'], reverse=True)
    for i in range(0,len(group_list)):
        group_list[i].GroupID = i

        
    # initialize global lists
    glist  = np.full(obj.ngas,  -1, dtype=np.int32)
    slist  = np.full(obj.nstar, -1, dtype=np.int32)
    dmlist = np.full(obj.ndm,   -1, dtype=np.int32)

    for group in group_list:
        glist[group.glist]   = group.GroupID
        slist[group.slist]   = group.GroupID
        dmlist[group.dmlist] = group.GroupID

        if not hasattr(group, 'unbound_indexes'):
            continue
        
        glist[group.unbound_indexes[ptype_ints['gas']]]  = -2
        slist[group.unbound_indexes[ptype_ints['star']]] = -2
        dmlist[group.unbound_indexes[ptype_ints['dm']]]  = -2
            
    setattr(obj.global_particle_lists, '%s_glist'  % find_type, glist)
    setattr(obj.global_particle_lists, '%s_slist'  % find_type, slist)
    setattr(obj.global_particle_lists, '%s_dmlist' % find_type, dmlist)       
    
    calculate_local_densities(obj, group_list)
    
    if find_type == 'halo':
        obj.halos  = group_list
        obj.nhalos = len(obj.halos)
    elif find_type == 'galaxy':
        obj.galaxies  = group_list
        obj.ngalaxies = len(obj.galaxies)
