from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from caesar.AHF_FAST_loader import (
    load_ahf_halos_dataframe,
    load_ahf_hierarchy,
    load_ahf_particle_blocks,
)
from caesar.ahf_subhalo_tables import (
    AHFNodeTable,
    ParticleShard,
    ParticleTable,
    RaggedIndexBlock,
    SnapshotMeta,
)


PTYPE_TO_GROUP = {
    "gas": "PartType0",
    "dm": "PartType1",
    "dm2": "PartType2",
    "dm3": "PartType3",
    "star": "PartType4",
    "bh": "PartType5",
}

PTYPE_CODE_TO_NAME = {
    0: "gas",
    1: "dm",
    2: "dm2",
    3: "dm3",
    4: "star",
    5: "bh",
}


@dataclass(frozen=True)
class PidLookup:
    particle_ids: np.ndarray
    rows: np.ndarray

    def map(self, particle_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        ids = np.asarray(particle_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0 or self.particle_ids.size == 0:
            return np.empty(0, dtype=np.int64)
        pos = np.searchsorted(self.particle_ids, ids)
        valid = (pos >= 0) & (pos < self.particle_ids.size) & (self.particle_ids[pos] == ids)
        if not np.any(valid):
            return np.empty(0, dtype=np.int64)
        return self.rows[pos[valid]].astype(np.int64, copy=False)


@dataclass
class AHFSubhaloDirectState:
    snapshot: SnapshotMeta
    particles: ParticleShard
    nodes: AHFNodeTable

    def to_payload(self) -> Dict[str, object]:
        return {
            "snapshot": self.snapshot.to_payload(),
            "particles": self.particles.to_payload(),
            "nodes": self.nodes.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "AHFSubhaloDirectState":
        return cls(
            snapshot=SnapshotMeta.from_payload(dict(payload["snapshot"])),
            particles=ParticleShard.from_payload(dict(payload["particles"])),
            nodes=AHFNodeTable.from_payload(dict(payload["nodes"])),
        )


def _gas_nH_cm3_from_density(density_code: np.ndarray, *, h: float, z: float, XH: float = 0.76) -> np.ndarray:
    density_code = np.asarray(density_code, dtype=np.float64)
    if density_code.size == 0:
        return np.empty(0, dtype=np.float32)
    msun = 1.98847e33
    kpc = 3.085677581e21
    mp = 1.67262192369e-24
    rho_cgs = density_code * (1.0e10 * msun) * (float(h) ** 2) / (kpc**3)
    rho_phys = rho_cgs * (1.0 + float(z)) ** 3
    return np.asarray(rho_phys * float(XH) / mp, dtype=np.float32)


def _gas_T_K_from_u_ne(u_code: np.ndarray, ne: np.ndarray, *, XH: float = 0.76) -> np.ndarray:
    u_code = np.asarray(u_code, dtype=np.float64)
    ne = np.asarray(ne, dtype=np.float64)
    if u_code.size == 0 or ne.size == 0:
        return np.empty(0, dtype=np.float32)
    gamma = 5.0 / 3.0
    kb = 1.380649e-16
    mp = 1.67262192369e-24
    y = (1.0 - float(XH)) / (4.0 * float(XH))
    mu = (1.0 + 4.0 * y) / (1.0 + y + ne)
    temp = (gamma - 1.0) * u_code * (1.0e5**2) * mu * mp / kb
    return np.asarray(temp, dtype=np.float32)


def _field_or_none(group: h5py.Group, names: Sequence[str]) -> Optional[np.ndarray]:
    for name in names:
        if name in group:
            return group[name][:]
    return None


def _scalarize_field(values: Optional[np.ndarray], *, dtype=np.float32) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[1] > 0:
        arr = arr[:, 0]
    return np.asarray(arr, dtype=dtype)


def _group_particle_count(handle: h5py.File, group_name: str) -> int:
    if group_name not in handle:
        return 0
    group = handle[group_name]
    if "ParticleIDs" in group:
        return int(group["ParticleIDs"].shape[0])
    if "Coordinates" in group:
        return int(group["Coordinates"].shape[0])
    return 0


def load_snapshot_meta(snapshot_file: str) -> SnapshotMeta:
    with h5py.File(snapshot_file, "r") as handle:
        header = handle["Header"].attrs
        boxsize = float(header.get("BoxSize", 0.0))
        hubble = float(header.get("HubbleParam", 1.0))
        redshift = float(header.get("Redshift", 0.0))
        scale_factor = float(header.get("Time", 1.0 / (1.0 + redshift)))
        omega_matter = float(header.get("Omega0", 0.0))
        omega_lambda = float(header.get("OmegaLambda", 0.0))
        omega_curvature = float(max(0.0, 1.0 - omega_matter - omega_lambda))
        cosmology = FlatLambdaCDM(H0=100.0 * hubble, Om0=omega_matter if omega_matter > 0.0 else 0.3)
        time_gyr = float(cosmology.age(redshift).value)
        hz_km_s_mpc = float(cosmology.H(redshift).value)
        hz_s = hz_km_s_mpc * 1.0e5 / 3.0856775814913673e24
        g_cgs = 6.6743e-8
        critical_density = (3.0 * hz_s * hz_s) / (8.0 * np.pi * g_cgs)
        critical_density *= (3.0856775814913673e21**3) / 1.98847e33

        ptypes = []
        fields_by_ptype: Dict[str, tuple[str, ...]] = {}
        particle_counts: Dict[str, int] = {}
        for ptype, group_name in PTYPE_TO_GROUP.items():
            if group_name not in handle:
                continue
            ptypes.append(ptype)
            fields_by_ptype[ptype] = tuple(sorted(handle[group_name].keys()))
            particle_counts[ptype] = _group_particle_count(handle, group_name)

    return SnapshotMeta(
        snapshot_file=str(snapshot_file),
        boxsize=float(boxsize / max(hubble, 1.0e-30)),
        redshift=redshift,
        scale_factor=scale_factor,
        hubble_constant=hubble,
        omega_matter=omega_matter,
        omega_lambda=omega_lambda,
        omega_curvature=omega_curvature,
        time_gyr=time_gyr,
        critical_density_msun_kpc3=float(critical_density),
        ptypes=tuple(ptypes),
        units={
            "mass": "Msun",
            "length": "kpccm",
            "velocity": "km/s",
            "time": "yr",
            "temperature": "K",
        },
        fields_by_ptype=fields_by_ptype,
        particle_counts=particle_counts,
    )


def load_particle_shard(snapshot_file: str) -> ParticleShard:
    meta = load_snapshot_meta(snapshot_file)
    tables: Dict[str, ParticleTable] = {}

    with h5py.File(snapshot_file, "r") as handle:
        header = handle["Header"].attrs
        mass_table = np.asarray(header.get("MassTable", np.zeros(6, dtype=np.float64)), dtype=np.float64)
        h = float(meta.hubble_constant)
        z = float(meta.redshift)

        for ptype, group_name in PTYPE_TO_GROUP.items():
            if group_name not in handle:
                continue
            group = handle[group_name]
            pcode = int(group_name.replace("PartType", ""))
            particle_ids = np.asarray(group["ParticleIDs"][:], dtype=np.int64)
            n = int(particle_ids.size)
            coords = np.asarray(group["Coordinates"][:], dtype=np.float32) / max(h, 1.0e-30)
            vels = np.asarray(group["Velocities"][:], dtype=np.float32)
            masses_raw = group["Masses"][:] if "Masses" in group else np.full(n, mass_table[pcode], dtype=np.float64)
            masses = np.asarray(masses_raw, dtype=np.float64) * (1.0e10 / max(h, 1.0e-30))
            pot = _field_or_none(group, ("Potential", "Potentials"))

            fields: Dict[str, np.ndarray] = {
                "particle_id": particle_ids,
                "global_index": np.arange(n, dtype=np.int64),
                "pos": np.asarray(coords, dtype=np.float32),
                "vel": np.asarray(vels, dtype=np.float32),
                "mass": np.asarray(masses, dtype=np.float32),
            }
            if pot is not None:
                fields["pot"] = np.asarray(pot, dtype=np.float32)

            if ptype == "gas":
                rho = _field_or_none(group, ("Density", "Densities"))
                sfr = _field_or_none(group, ("StarFormationRate", "StarFormationRates"))
                ne = _field_or_none(group, ("ElectronAbundance", "ElectronNumberDensities"))
                u = _field_or_none(group, ("InternalEnergy",))
                temp = _field_or_none(group, ("Temperature", "Temperatures"))
                metallicity = _scalarize_field(_field_or_none(group, ("Metallicity", "GFM_Metallicity")))
                gfhi = _field_or_none(group, ("GrackleHI", "NeutralHydrogenAbundance", "AtomicHydrogenMasses"))
                gfh2 = _field_or_none(group, ("FractionH2", "MolecularHydrogenMasses"))
                dustmass = _field_or_none(group, ("Dust_Masses",))
                if rho is not None:
                    fields["gnh"] = _gas_nH_cm3_from_density(rho, h=h, z=z)
                if sfr is not None:
                    fields["gsfr"] = np.asarray(sfr, dtype=np.float32)
                if metallicity is not None:
                    fields["gZ"] = np.asarray(metallicity, dtype=np.float32)
                if temp is not None:
                    fields["gT"] = np.asarray(temp, dtype=np.float32)
                elif u is not None and ne is not None:
                    fields["gT"] = _gas_T_K_from_u_ne(u, ne)
                if gfhi is not None:
                    fields["gfHI"] = np.asarray(gfhi, dtype=np.float32)
                if gfh2 is not None:
                    fields["gfH2"] = np.asarray(gfh2, dtype=np.float32)
                if dustmass is not None:
                    fields["dustmass"] = np.asarray(dustmass, dtype=np.float64) * (1.0e10 / max(h, 1.0e-30))

            if ptype == "star":
                metallicity = _scalarize_field(_field_or_none(group, ("Metallicity", "GFM_Metallicity")))
                aform = _field_or_none(group, ("StellarFormationTime", "GFM_StellarFormationTime", "BirthScaleFactors"))
                if metallicity is not None:
                    fields["sZ"] = np.asarray(metallicity, dtype=np.float32)
                if aform is not None:
                    aform_arr = np.asarray(aform, dtype=np.float64)
                    valid = aform_arr > 0.0
                    age = np.zeros_like(aform_arr, dtype=np.float64)
                    if np.any(valid):
                        cosmology = FlatLambdaCDM(H0=100.0 * h, Om0=meta.omega_matter if meta.omega_matter > 0.0 else 0.3)
                        zform = (1.0 / np.maximum(aform_arr[valid], 1.0e-12)) - 1.0
                        age[valid] = float(meta.time_gyr) - cosmology.age(zform).value
                    fields["age"] = np.asarray(age, dtype=np.float32)

            if ptype == "bh":
                bhmass = _field_or_none(group, ("BH_Mass", "SubgridMasses"))
                bhmdot = _field_or_none(group, ("BH_Mdot", "AccretionRates"))
                if bhmass is not None:
                    fields["bhmass"] = np.asarray(bhmass, dtype=np.float64) * (1.0e10 / max(h, 1.0e-30))
                if bhmdot is not None:
                    fields["bhmdot"] = np.asarray(bhmdot, dtype=np.float32)

            tables[ptype] = ParticleTable(fields=fields)

    return ParticleShard(meta=meta, tables=tables)


def build_pid_lookups(particles: ParticleShard) -> Dict[str, PidLookup]:
    lookups: Dict[str, PidLookup] = {}
    for ptype, table in particles.tables.items():
        particle_ids = np.asarray(table.get("particle_id", np.empty(0, dtype=np.int64)), dtype=np.int64)
        if particle_ids.size == 0:
            continue
        order = np.argsort(particle_ids)
        lookups[ptype] = PidLookup(
            particle_ids=particle_ids[order],
            rows=np.asarray(table.get("global_index", np.arange(particle_ids.size, dtype=np.int64)), dtype=np.int64)[order],
        )
    return lookups


def build_ahf_node_table(
    ahf_particles_file: str,
    particles: ParticleShard,
) -> AHFNodeTable:
    hierarchy = load_ahf_hierarchy(ahf_particles_file)
    halos_df = load_ahf_halos_dataframe(ahf_particles_file)
    all_nodes = sorted(int(v) for v in hierarchy.parent_of.keys())
    memberships_raw = load_ahf_particle_blocks(
        ahf_particles_file,
        needed_nodes=all_nodes,
        load_dm=True,
    )
    pid_lookups = build_pid_lookups(particles)

    npart_map = {
        int(hid): int(npart)
        for hid, npart in zip(halos_df["hid"].to_numpy(), halos_df["npart"].to_numpy())
    }

    def _ancestor_chain(node_id: int) -> np.ndarray:
        out = []
        seen = set()
        cur = int(node_id)
        while True:
            parent = int(hierarchy.parent_of.get(cur, 0) or 0)
            if parent <= 0 or parent in seen:
                break
            out.append(parent)
            seen.add(parent)
            cur = parent
        return np.asarray(out, dtype=np.int64)

    member_rows: Dict[str, list[np.ndarray]] = {ptype: [] for ptype in particles.ptypes}
    halo_id = np.empty(len(all_nodes), dtype=np.int64)
    parent_id = np.empty(len(all_nodes), dtype=np.int64)
    top_id = np.empty(len(all_nodes), dtype=np.int64)
    depth = np.empty(len(all_nodes), dtype=np.int64)
    star_count = np.empty(len(all_nodes), dtype=np.int64)
    dm_count = np.empty(len(all_nodes), dtype=np.int64)
    fof_candidates = np.empty(len(all_nodes), dtype=np.int64)
    ancestor_rows = []

    for i, node_id in enumerate(all_nodes):
        halo_id[i] = int(node_id)
        parent_id[i] = int(hierarchy.parent_of.get(int(node_id), 0) or 0)
        ancestors = _ancestor_chain(int(node_id))
        ancestor_rows.append(ancestors)
        depth[i] = int(ancestors.size)
        top_id[i] = int(ancestors[-1]) if ancestors.size > 0 else int(node_id)
        if ancestors.size > 0:
            top_id[i] = int(ancestors[-1])
        else:
            top_id[i] = int(node_id)

        by_type: Dict[str, np.ndarray] = {ptype: np.empty(0, dtype=np.int64) for ptype in particles.ptypes}
        raw = np.asarray(memberships_raw.get(int(node_id), np.empty((0, 2), dtype=np.int64)), dtype=np.int64)
        if raw.size > 0:
            if raw.ndim != 2 or raw.shape[1] != 2:
                raw = raw.reshape(-1, 2)
            pids = raw[:, 0]
            ptypes = raw[:, 1]
            for pcode, ptype_name in PTYPE_CODE_TO_NAME.items():
                lookup = pid_lookups.get(ptype_name)
                if lookup is None:
                    continue
                mask = ptypes == int(pcode)
                if np.any(mask):
                    mapped = lookup.map(pids[mask])
                    by_type[ptype_name] = np.unique(mapped.astype(np.int64, copy=False))

        for ptype in particles.ptypes:
            member_rows.setdefault(ptype, []).append(by_type.get(ptype, np.empty(0, dtype=np.int64)))

        star_count[i] = int(by_type.get("star", np.empty(0, dtype=np.int64)).size)
        dm_total = 0
        for name in ("dm", "dm2", "dm3"):
            dm_total += int(by_type.get(name, np.empty(0, dtype=np.int64)).size)
        dm_count[i] = int(dm_total)
        fof_candidates[i] = int(max(1, npart_map.get(int(node_id), 0)))

    return AHFNodeTable(
        halo_id=halo_id,
        parent_halo_id=parent_id,
        top_halo_id=top_id,
        depth=depth,
        star_count=star_count,
        dm_count=dm_count,
        fof_candidates=fof_candidates,
        ancestors=RaggedIndexBlock.from_sequences(ancestor_rows),
        members_by_type={
            str(ptype): RaggedIndexBlock.from_sequences(rows)
            for ptype, rows in member_rows.items()
        },
    )


def build_direct_state(snapshot_file: str, ahf_particles_file: str) -> AHFSubhaloDirectState:
    particles = load_particle_shard(snapshot_file)
    nodes = build_ahf_node_table(ahf_particles_file, particles)
    return AHFSubhaloDirectState(snapshot=particles.meta, particles=particles, nodes=nodes)


def dense_gas_selected(
    particles: ParticleShard,
    gas_indices: np.ndarray,
    *,
    fof_nHlim: float,
    fof_Tlim: float,
    fof_use_sfr_gate: bool,
) -> np.ndarray:
    gas_idx = np.asarray(gas_indices, dtype=np.int64)
    if gas_idx.size == 0 or not particles.has_ptype("gas"):
        return np.empty(0, dtype=np.int64)
    table = particles.table("gas")
    gnh = table.get("gnh")
    gtemp = table.get("gT")
    gsfr = table.get("gsfr")
    if gnh is None or gtemp is None or gsfr is None:
        return gas_idx.astype(np.int64, copy=False)
    nh = np.asarray(gnh[gas_idx], dtype=np.float64)
    temp = np.asarray(gtemp[gas_idx], dtype=np.float64)
    sfr = np.asarray(gsfr[gas_idx], dtype=np.float64)
    if fof_use_sfr_gate:
        mask = (nh > float(fof_nHlim)) & ((temp < float(fof_Tlim)) | (sfr > 0.0))
    else:
        mask = (nh > float(fof_nHlim)) & (temp < float(fof_Tlim))
    return gas_idx[mask].astype(np.int64, copy=False)


def resolve_node_membership(
    state: AHFSubhaloDirectState,
    *,
    node_id: int,
    fof_nHlim: float,
    fof_Tlim: float,
    fof_use_sfr_gate: bool,
) -> Dict[str, np.ndarray]:
    idx = state.nodes.index_of(int(node_id))
    gas_sel = dense_gas_selected(
        state.particles,
        state.nodes.members_for(idx, "gas"),
        fof_nHlim=float(fof_nHlim),
        fof_Tlim=float(fof_Tlim),
        fof_use_sfr_gate=bool(fof_use_sfr_gate),
    )
    return {
        "gas_sel": np.asarray(gas_sel, dtype=np.int64),
        "star_sel": state.nodes.members_for(idx, "star"),
        "bh_sel": state.nodes.members_for(idx, "bh"),
        "dm_sel": state.nodes.members_for(idx, "dm"),
        "dm2_sel": state.nodes.members_for(idx, "dm2"),
        "dm3_sel": state.nodes.members_for(idx, "dm3"),
    }


def build_task_payload(
    state: AHFSubhaloDirectState,
    *,
    task,
    fof_nHlim: float,
    fof_Tlim: float,
    fof_use_sfr_gate: bool,
) -> Optional[Dict[str, object]]:
    membership = resolve_node_membership(
        state,
        node_id=int(task.node_id),
        fof_nHlim=float(fof_nHlim),
        fof_Tlim=float(fof_Tlim),
        fof_use_sfr_gate=bool(fof_use_sfr_gate),
    )
    gas_sel = np.asarray(membership["gas_sel"], dtype=np.int32)
    star_sel = np.asarray(membership["star_sel"], dtype=np.int32)
    bh_sel = np.asarray(membership["bh_sel"], dtype=np.int32)
    dm_sel = np.asarray(
        np.unique(
            np.concatenate(
                [
                    np.asarray(membership["dm_sel"], dtype=np.int64),
                    np.asarray(membership["dm2_sel"], dtype=np.int64),
                    np.asarray(membership["dm3_sel"], dtype=np.int64),
                ]
            )
        ),
        dtype=np.int32,
    )

    gas_pos = (
        np.asarray(state.particles.table("gas").get("pos")[gas_sel], dtype=np.float32)
        if gas_sel.size > 0 and state.particles.has_ptype("gas")
        else np.empty((0, 3), dtype=np.float32)
    )
    gas_vel = (
        np.asarray(state.particles.table("gas").get("vel")[gas_sel], dtype=np.float32)
        if gas_sel.size > 0 and state.particles.has_ptype("gas")
        else np.empty((0, 3), dtype=np.float32)
    )
    star_pos = (
        np.asarray(state.particles.table("star").get("pos")[star_sel], dtype=np.float32)
        if star_sel.size > 0 and state.particles.has_ptype("star")
        else np.empty((0, 3), dtype=np.float32)
    )
    star_vel = (
        np.asarray(state.particles.table("star").get("vel")[star_sel], dtype=np.float32)
        if star_sel.size > 0 and state.particles.has_ptype("star")
        else np.empty((0, 3), dtype=np.float32)
    )
    bh_pos = (
        np.asarray(state.particles.table("bh").get("pos")[bh_sel], dtype=np.float32)
        if bh_sel.size > 0 and state.particles.has_ptype("bh")
        else np.empty((0, 3), dtype=np.float32)
    )
    bh_vel = (
        np.asarray(state.particles.table("bh").get("vel")[bh_sel], dtype=np.float32)
        if bh_sel.size > 0 and state.particles.has_ptype("bh")
        else np.empty((0, 3), dtype=np.float32)
    )

    eligible_pos = np.concatenate([gas_pos, star_pos, bh_pos], axis=0)
    eligible_vel = np.concatenate([gas_vel, star_vel, bh_vel], axis=0)
    if eligible_pos.shape[0] == 0:
        return None

    return {
        "gas_sel": gas_sel,
        "star_sel": star_sel,
        "bh_sel": bh_sel,
        "dm_sel": dm_sel,
        "ng": int(gas_sel.size),
        "ns": int(star_sel.size),
        "nb": int(bh_sel.size),
        "eligible_pos": eligible_pos,
        "eligible_vel": eligible_vel,
    }


def build_halo_record(state: AHFSubhaloDirectState, *, node_id: int) -> Dict[str, object]:
    idx = state.nodes.index_of(int(node_id))
    return {
        "AHF_haloID": int(state.nodes.halo_id[idx]),
        "AHF_parent_haloID": int(state.nodes.parent_halo_id[idx]),
        "AHF_top_haloID": int(state.nodes.top_halo_id[idx]),
        "AHF_depth": int(state.nodes.depth[idx]),
        "AHF_ancestor_haloIDs": state.nodes.ancestors_for(idx).astype(np.int64, copy=False),
        "glist": state.nodes.members_for(idx, "gas", dtype=np.int64),
        "slist": state.nodes.members_for(idx, "star", dtype=np.int64),
        "dmlist": np.unique(
            np.concatenate(
                [
                    state.nodes.members_for(idx, "dm", dtype=np.int64),
                    state.nodes.members_for(idx, "dm2", dtype=np.int64),
                    state.nodes.members_for(idx, "dm3", dtype=np.int64),
                ]
            )
        ).astype(np.int64, copy=False),
        "bhlist": state.nodes.members_for(idx, "bh", dtype=np.int64),
        "dlist": state.nodes.members_for(idx, "dust", dtype=np.int64),
    }
