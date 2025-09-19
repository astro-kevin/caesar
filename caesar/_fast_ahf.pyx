# cython: language_level=3

def iter_memberships(str path, object needed, bint load_dm=True):
    """Yield ParticleMembership objects for IDs in ``needed``.

    This Cython-backed version mirrors ``_iter_memberships_stream`` but keeps
    the streaming behaviour so memory stays bounded.
    """
    from caesar.halo_matching import ParticleMembership, _open_ahf_particles

    cdef object fh = _open_ahf_particles(path)
    cdef object line, parts, pparts
    cdef int remaining = 0
    cdef int current_hid = -1
    cdef object pm = None

    try:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if remaining == 0 and len(parts) == 2:
                if pm is not None and pm.id in needed:
                    yield pm

                try:
                    remaining = int(parts[0])
                    current_hid = int(parts[1])
                except Exception:
                    remaining = 0
                    pm = None
                    continue

                if current_hid in needed:
                    pm = ParticleMembership(current_hid)
                else:
                    pm = None
                continue

            if remaining > 0:
                remaining -= 1
                if pm is None:
                    continue

                pparts = line.split('\t')
                if len(pparts) != 2:
                    continue

                try:
                    pid = int(pparts[0])
                    ptype = int(pparts[1])
                except Exception:
                    continue

                if ptype == 0:
                    pm.parttype0.add(pid)
                elif ptype == 4:
                    pm.parttype4.add(pid)
                elif ptype == 5:
                    pm.parttype5.add(pid)
                elif load_dm and ptype == 1:
                    pm.parttype1.add(pid)

        if pm is not None and pm.id in needed:
            yield pm
    finally:
        fh.close()


def read_memberships_for_ids(str path, object needed, bint load_dm=True):
    """Return a dict mapping halo IDs to ParticleMembership objects."""
    cdef dict out = {}
    for pm in iter_memberships(path, needed, load_dm):
        out[pm.id] = pm
    return out
