import numpy as np


plist_dict = dict(
    gas='glist',
    star='slist',
    bh='bhlist',
    dust='dlist',
    dm='dmlist',
    dm2='dm2list',
    dm3='dm3list',
)


def reset_global_particle_IDs(obj):
    """Map particle lists from currently loaded ID's back to full snapshot indices."""

    from caesar.property_manager import has_ptype, get_property, has_property

    offset = np.zeros(len(obj.data_manager.ptypes) + 1, dtype=np.int64)
    for ip, p in enumerate(obj.data_manager.ptypes):
        if not has_ptype(obj, p):
            continue
        if p == 'bh':
            if has_property(obj, p, 'bhmass'):
                count = len(get_property(obj, 'bhmass', p))  # some data formats (eg SWIFT) don't have mass for BH
            else:
                count = len(get_property(obj, 'mass', p))
        else:
            count = len(get_property(obj, 'mass', p))
        if p == 'gas':
            offset[ip + 1] = offset[ip] + obj.simulation.ngas
            obj.simulation.ngas = count
        elif p == 'star':
            offset[ip + 1] = offset[ip] + obj.simulation.nstar
            obj.simulation.nstar = count
        elif p == 'bh':
            offset[ip + 1] = offset[ip] + obj.simulation.nbh
            obj.simulation.nbh = count
        elif p == 'dust':
            offset[ip + 1] = offset[ip] + obj.simulation.ndust
            obj.simulation.ndust = count
        elif p == 'dm':
            offset[ip + 1] = offset[ip] + obj.simulation.ndm
            obj.simulation.ndm = count
        elif p == 'dm2':
            offset[ip + 1] = offset[ip] + obj.simulation.ndm2
            obj.simulation.ndm2 = count
        elif p == 'dm3':
            offset[ip + 1] = offset[ip] + obj.simulation.ndm3
            obj.simulation.ndm3 = count

    # Summary diagnostics (optional): capture pre-map gas counts per galaxy
    import os as _os

    do_reset_check = _os.environ.get('CAESAR_RESET_CHECK', '0') == '1'
    do_reset_assert = _os.environ.get('CAESAR_ASSERT_RESET', '0') == '1'
    pre_gas_counts = None
    if do_reset_check and 'galaxy' in obj.group_types:
        try:
            pre_gas_counts = [
                len(getattr(g, 'glist', [])) if getattr(g, 'glist', None) is not None else 0
                for g in obj.galaxy_list
            ]
        except Exception:
            pre_gas_counts = None

    # reset lists
    from caesar.property_manager import has_ptype as _has_ptype

    for group_type in obj.group_types:
        group_list = 'obj.%s_list' % group_type
        for ip, p in enumerate(obj.data_manager.ptypes):
            if not _has_ptype(obj, p):
                continue
            for group in eval(group_list):
                part_list = 'group.%s' % plist_dict[p]
                mylist = eval(part_list)
                mylist = obj.data_manager.indexes[mylist + offset[ip]]
                if p == 'gas':
                    group.glist = mylist
                if p == 'star':
                    group.slist = mylist
                if p == 'bh':
                    group.bhlist = mylist
                if p == 'dust':
                    group.dlist = mylist
                if p == 'dm':
                    group.dmlist = mylist
                if p == 'dm2':
                    group.dm2list = mylist
                if p == 'dm3':
                    group.dm3list = mylist

    # Post-map diagnostics: compare gas counts before vs after mapping
    if (do_reset_check or do_reset_assert) and pre_gas_counts is not None and 'galaxy' in obj.group_types:
        post_gas_counts = []
        try:
            post_gas_counts = [
                len(getattr(g, 'glist', [])) if getattr(g, 'glist', None) is not None else 0
                for g in obj.galaxy_list
            ]
        except Exception:
            post_gas_counts = []
        from yt.funcs import mylog

        pre_with = sum(1 for v in pre_gas_counts if v > 0)
        post_with = sum(1 for v in post_gas_counts if v > 0)
        mylog.info(
            'reset_global_particle_IDs: galaxies with gas before=%d after=%d (total=%d)',
            pre_with,
            post_with,
            len(post_gas_counts),
        )
        if do_reset_assert and pre_with > 0 and post_with == 0:
            mylog.error(
                'Assertion: Gas lost after reset mapping (pre_with=%d, post_with=%d)',
                pre_with,
                post_with,
            )
            raise AssertionError(
                'Gas lost after reset_global_particle_IDs: nonzero pre-map gas count dropped to zero'
            )

    return


def load_global_lists(obj):
    """Populate global reverse maps (glist/slist/etc) for each group type."""

    from caesar.property_manager import has_ptype

    for group_type in obj.group_types:
        glist = np.full(obj.simulation.ngas, -1, dtype=np.int32)
        slist = np.full(obj.simulation.nstar, -1, dtype=np.int32)
        bhlist = np.full(obj.simulation.nbh, -1, dtype=np.int32)
        dlist = np.full(obj.simulation.ndust, -1, dtype=np.int32)
        dmlist = np.full(obj.simulation.ndm, -1, dtype=np.int32)
        if 'dm2' in obj.data_manager.ptypes:
            dm2list = np.full(obj.simulation.ndm2, -1, dtype=np.int32)
        if 'dm3' in obj.data_manager.ptypes:
            dm3list = np.full(obj.simulation.ndm3, -1, dtype=np.int32)

        group_list = 'obj.%s_list' % group_type
        for group in eval(group_list):
            for p in obj.data_manager.ptypes:
                if not has_ptype(obj, p):
                    continue
                part_list = 'group.%s' % plist_dict[p]
                if p == 'gas':
                    glist[eval(part_list)] = group.GroupID
                if p == 'star':
                    slist[eval(part_list)] = group.GroupID
                if p == 'bh':
                    bhlist[eval(part_list)] = group.GroupID
                if p == 'dust':
                    dlist[eval(part_list)] = group.GroupID
                if p == 'dm':
                    dmlist[eval(part_list)] = group.GroupID
                if p == 'dm2':
                    dm2list[eval(part_list)] = group.GroupID
                if p == 'dm3':
                    dm3list[eval(part_list)] = group.GroupID

        setattr(obj.global_particle_lists, '%s_glist' % group_type, glist)
        setattr(obj.global_particle_lists, '%s_slist' % group_type, slist)
        setattr(obj.global_particle_lists, '%s_bhlist' % group_type, bhlist)
        setattr(obj.global_particle_lists, '%s_dlist' % group_type, dlist)
        # If we have an exclusive DM override for galaxies, use it for the global reverse map
        if group_type == 'galaxy' and hasattr(obj, '_exclusive_galaxy_dmlist'):
            setattr(obj.global_particle_lists, '%s_dmlist' % group_type, obj._exclusive_galaxy_dmlist)
        else:
            setattr(obj.global_particle_lists, '%s_dmlist' % group_type, dmlist)
        if 'dm2' in obj.data_manager.ptypes:
            setattr(obj.global_particle_lists, '%s_dm2list' % group_type, dm2list)
        if 'dm3' in obj.data_manager.ptypes:
            setattr(obj.global_particle_lists, '%s_dm3list' % group_type, dm3list)

    return

