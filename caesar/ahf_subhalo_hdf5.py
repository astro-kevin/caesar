from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
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


def _snapshot_meta_to_attrs(meta_group: h5py.Group, meta: SnapshotMeta) -> None:
    meta_group.attrs["snapshot_file"] = str(meta.snapshot_file)
    meta_group.attrs["boxsize"] = float(meta.boxsize)
    meta_group.attrs["redshift"] = float(meta.redshift)
    meta_group.attrs["scale_factor"] = float(meta.scale_factor)
    meta_group.attrs["hubble_constant"] = float(meta.hubble_constant)
    meta_group.attrs["omega_matter"] = float(meta.omega_matter)
    meta_group.attrs["omega_lambda"] = float(meta.omega_lambda)
    meta_group.attrs["omega_curvature"] = float(meta.omega_curvature)
    meta_group.attrs["time_gyr"] = float(meta.time_gyr)
    meta_group.attrs["critical_density_msun_kpc3"] = float(meta.critical_density_msun_kpc3)
    meta_group.attrs["ptypes_json"] = json.dumps(list(meta.ptypes))
    meta_group.attrs["units_json"] = json.dumps(dict(meta.units))
    meta_group.attrs["fields_by_ptype_json"] = json.dumps({str(k): list(v) for k, v in meta.fields_by_ptype.items()})
    meta_group.attrs["particle_counts_json"] = json.dumps({str(k): int(v) for k, v in meta.particle_counts.items()})


def _snapshot_meta_from_attrs(meta_group: h5py.Group) -> SnapshotMeta:
    return SnapshotMeta(
        snapshot_file=_store_attr_text(meta_group.attrs["snapshot_file"]),
        boxsize=float(meta_group.attrs["boxsize"]),
        redshift=float(meta_group.attrs["redshift"]),
        scale_factor=float(meta_group.attrs["scale_factor"]),
        hubble_constant=float(meta_group.attrs["hubble_constant"]),
        omega_matter=float(meta_group.attrs["omega_matter"]),
        omega_lambda=float(meta_group.attrs["omega_lambda"]),
        omega_curvature=float(meta_group.attrs["omega_curvature"]),
        time_gyr=float(meta_group.attrs["time_gyr"]),
        critical_density_msun_kpc3=float(meta_group.attrs["critical_density_msun_kpc3"]),
        ptypes=tuple(str(v) for v in json.loads(_store_attr_text(meta_group.attrs["ptypes_json"]))),
        units={str(k): str(v) for k, v in dict(json.loads(_store_attr_text(meta_group.attrs["units_json"]))).items()},
        fields_by_ptype={
            str(k): tuple(str(x) for x in v)
            for k, v in dict(json.loads(_store_attr_text(meta_group.attrs["fields_by_ptype_json"]))).items()
        },
        particle_counts={
            str(k): int(v)
            for k, v in dict(json.loads(_store_attr_text(meta_group.attrs["particle_counts_json"]))).items()
        },
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


def _store_attr_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf8")
    return str(value)


def write_snapshot_meta_store(path: str | Path, snapshot_meta: SnapshotMeta) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(snapshot_meta.to_payload(), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return target


def load_snapshot_meta_store(path: str | Path) -> SnapshotMeta:
    source = Path(path)
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    return SnapshotMeta.from_payload(dict(payload))


def _local_member_row_maps(
    state: AHFSubhaloDirectState,
    *,
    node_indexes: Sequence[int],
) -> Dict[str, np.ndarray]:
    rows_by_ptype: Dict[str, np.ndarray] = {}
    for ptype in state.particles.ptypes:
        rows = [
            np.asarray(state.nodes.members_for(int(node_index), str(ptype), dtype=np.int64), dtype=np.int64)
            for node_index in node_indexes
        ]
        if rows:
            merged = np.unique(np.concatenate(rows).astype(np.int64, copy=False))
        else:
            merged = np.empty(0, dtype=np.int64)
        rows_by_ptype[str(ptype)] = merged
    return rows_by_ptype


def _build_particle_substore(
    particles: ParticleShard,
    *,
    rows_by_ptype: Mapping[str, np.ndarray],
) -> tuple[ParticleShard, Dict[str, Dict[int, int]]]:
    tables: Dict[str, ParticleTable] = {}
    local_maps: Dict[str, Dict[int, int]] = {}

    for ptype in particles.ptypes:
        rows = np.asarray(rows_by_ptype.get(str(ptype), np.empty(0, dtype=np.int64)), dtype=np.int64)
        table = particles.table(str(ptype))
        local_maps[str(ptype)] = {int(global_row): int(local_row) for local_row, global_row in enumerate(rows.tolist())}
        fields: Dict[str, np.ndarray] = {}
        for field_name, values in table.fields.items():
            arr = np.asarray(values)
            if rows.size > 0:
                fields[str(field_name)] = np.asarray(arr[rows])
            else:
                fields[str(field_name)] = np.empty((0,) + tuple(arr.shape[1:]), dtype=arr.dtype)
        if "global_index" not in fields:
            fields["global_index"] = np.asarray(rows, dtype=np.int64)
        tables[str(ptype)] = ParticleTable(fields=fields)

    particle_counts = {
        str(ptype): int(len(tables[str(ptype)]))
        for ptype in tables.keys()
    }
    submeta = SnapshotMeta(
        snapshot_file=str(particles.meta.snapshot_file),
        boxsize=float(particles.meta.boxsize),
        redshift=float(particles.meta.redshift),
        scale_factor=float(particles.meta.scale_factor),
        hubble_constant=float(particles.meta.hubble_constant),
        omega_matter=float(particles.meta.omega_matter),
        omega_lambda=float(particles.meta.omega_lambda),
        omega_curvature=float(particles.meta.omega_curvature),
        time_gyr=float(particles.meta.time_gyr),
        critical_density_msun_kpc3=float(particles.meta.critical_density_msun_kpc3),
        ptypes=tuple(str(ptype) for ptype in particles.ptypes),
        units=dict(particles.meta.units),
        fields_by_ptype={
            str(ptype): tuple(tables[str(ptype)].field_names())
            for ptype in tables.keys()
        },
        particle_counts=particle_counts,
    )
    return ParticleShard(meta=submeta, tables=tables), local_maps


def _build_node_substore(
    state: AHFSubhaloDirectState,
    *,
    node_indexes: Sequence[int],
    local_maps: Mapping[str, Mapping[int, int]],
) -> AHFNodeTable:
    node_indexes = [int(v) for v in node_indexes]
    halo_id = np.asarray([int(state.nodes.halo_id[idx]) for idx in node_indexes], dtype=np.int64)
    parent_halo_id = np.asarray([int(state.nodes.parent_halo_id[idx]) for idx in node_indexes], dtype=np.int64)
    top_halo_id = np.asarray([int(state.nodes.top_halo_id[idx]) for idx in node_indexes], dtype=np.int64)
    depth = np.asarray([int(state.nodes.depth[idx]) for idx in node_indexes], dtype=np.int64)
    star_count = np.asarray([int(state.nodes.star_count[idx]) for idx in node_indexes], dtype=np.int64)
    dm_count = np.asarray([int(state.nodes.dm_count[idx]) for idx in node_indexes], dtype=np.int64)
    fof_candidates = np.asarray([int(state.nodes.fof_candidates[idx]) for idx in node_indexes], dtype=np.int64)
    ancestor_rows = [np.asarray(state.nodes.ancestors_for(idx), dtype=np.int64) for idx in node_indexes]

    members_by_type: Dict[str, RaggedIndexBlock] = {}
    for ptype in state.particles.ptypes:
        ptype_map = local_maps.get(str(ptype), {})
        member_rows = []
        for idx in node_indexes:
            global_rows = np.asarray(state.nodes.members_for(idx, str(ptype), dtype=np.int64), dtype=np.int64)
            local_rows = np.asarray([int(ptype_map[int(row)]) for row in global_rows.tolist()], dtype=np.int64) if global_rows.size > 0 else np.empty(0, dtype=np.int64)
            member_rows.append(local_rows)
        members_by_type[str(ptype)] = RaggedIndexBlock.from_sequences(member_rows)

    return AHFNodeTable(
        halo_id=halo_id,
        parent_halo_id=parent_halo_id,
        top_halo_id=top_halo_id,
        depth=depth,
        star_count=star_count,
        dm_count=dm_count,
        fof_candidates=fof_candidates,
        ancestors=RaggedIndexBlock.from_sequences(ancestor_rows),
        members_by_type=members_by_type,
    )


def build_store_for_roots(
    state: AHFSubhaloDirectState,
    *,
    root_ids: Sequence[int],
) -> tuple[ParticleShard, AHFNodeTable, Dict[str, object]]:
    root_ids_sorted = tuple(sorted({int(root_id) for root_id in root_ids}))
    node_indexes = [
        int(node_index)
        for node_index in range(len(state.nodes))
        if int(state.nodes.top_halo_id[node_index]) in root_ids_sorted
    ]
    rows_by_ptype = _local_member_row_maps(state, node_indexes=node_indexes)
    particle_store, local_maps = _build_particle_substore(state.particles, rows_by_ptype=rows_by_ptype)
    node_store = _build_node_substore(state, node_indexes=node_indexes, local_maps=local_maps)
    return particle_store, node_store, {
        "root_ids": root_ids_sorted,
        "node_ids": tuple(int(node_store.halo_id[idx]) for idx in range(len(node_store))),
        "particle_rows_by_ptype": {
            str(ptype): int(np.asarray(rows, dtype=np.int64).size)
            for ptype, rows in rows_by_ptype.items()
        },
    }


def write_particle_store(path: str | Path, particles: ParticleShard) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = particles.meta
    with h5py.File(target, "w") as handle:
        meta_group = handle.create_group("meta")
        _snapshot_meta_to_attrs(meta_group, meta)

        particles_group = handle.create_group("particles")
        for ptype, table in particles.tables.items():
            ptype_group = particles_group.create_group(str(ptype))
            for field_name, values in table.fields.items():
                ptype_group.create_dataset(str(field_name), data=np.asarray(values))
    return target


def load_particle_store_meta(path: str | Path) -> SnapshotMeta:
    source = Path(path)
    with h5py.File(source, "r") as handle:
        return _snapshot_meta_from_attrs(handle["meta"])


def load_particle_store(path: str | Path) -> ParticleShard:
    source = Path(path)
    with h5py.File(source, "r") as handle:
        meta = _snapshot_meta_from_attrs(handle["meta"])
        particles_group = handle["particles"]
        tables: Dict[str, ParticleTable] = {}
        for ptype in particles_group.keys():
            group = particles_group[str(ptype)]
            tables[str(ptype)] = ParticleTable(
                fields={str(field_name): np.asarray(dataset[:]) for field_name, dataset in group.items()}
            )
    return ParticleShard(meta=meta, tables=tables)


def write_node_store(
    path: str | Path,
    node_store: AHFNodeTable,
    *,
    root_ids: Sequence[int],
    particle_rows_by_ptype: Optional[Mapping[str, int]] = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "root_ids": tuple(sorted(int(v) for v in root_ids)),
        "particle_rows_by_ptype": {
            str(k): int(v)
            for k, v in dict(particle_rows_by_ptype or {}).items()
        },
        "nodes": node_store.to_payload(),
    }
    with target.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return target


def load_node_store(path: str | Path) -> Dict[str, object]:
    source = Path(path)
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    return {
        "root_ids": tuple(int(v) for v in payload.get("root_ids", ())),
        "particle_rows_by_ptype": {
            str(k): int(v)
            for k, v in dict(payload.get("particle_rows_by_ptype", {})).items()
        },
        "nodes": AHFNodeTable.from_payload(dict(payload["nodes"])),
    }


def load_selected_particle_payload_from_store(
    path: str | Path | ParticleShard,
    *,
    gas_ids: np.ndarray,
    star_ids: np.ndarray,
    dm_ids: np.ndarray,
    bh_ids: np.ndarray,
    dust_ids: Optional[np.ndarray] = None,
) -> tuple[Dict[str, object], list[str], SnapshotMeta]:
    from caesar.property_manager import ptype_ints

    source = str(path)
    if isinstance(path, ParticleShard):
        particles = path
    else:
        source = str(Path(path))
        particles = load_particle_store(Path(path))
    snapshot_meta = particles.meta
    gas_ids = np.asarray(gas_ids, dtype=np.int64)
    star_ids = np.asarray(star_ids, dtype=np.int64)
    dm_ids = np.asarray(dm_ids, dtype=np.int64)
    bh_ids = np.asarray(bh_ids, dtype=np.int64)
    dust_ids = np.asarray(dust_ids if dust_ids is not None else np.empty(0, dtype=np.int64), dtype=np.int64)

    payload_dm: Dict[str, object] = {}
    pos_blocks: list[np.ndarray] = []
    vel_blocks: list[np.ndarray] = []
    mass_blocks: list[np.ndarray] = []
    pot_blocks: list[np.ndarray] = []
    ptype_blocks: list[np.ndarray] = []
    ptypes_local: list[str] = []

    selectors = (
        ("gas", gas_ids, ("gnh", "gsfr", "gZ", "gT", "gfHI", "gfH2", "dustmass")),
        ("dm", dm_ids, ()),
        ("star", star_ids, ("sZ", "age")),
        ("bh", bh_ids, ("bhmass", "bhmdot")),
        ("dust", dust_ids, ()),
    )
    offset = 0
    for ptype, ids, optional_fields in selectors:
        if ids.size == 0 or not particles.has_ptype(str(ptype)):
            continue
        table = particles.table(str(ptype))
        store_global = np.asarray(table.get("global_index", np.arange(len(table), dtype=np.int64)), dtype=np.int64)
        order = np.argsort(store_global)
        sorted_global = store_global[order]
        pos = np.searchsorted(sorted_global, ids)
        valid = (pos >= 0) & (pos < sorted_global.size) & (sorted_global[pos] == ids)
        if not np.all(valid):
            missing = ids[~valid]
            raise RuntimeError(
                f"Particle store {source} is missing {missing.size} requested {ptype} global indexes"
            )
        store_rows = order[pos].astype(np.int64, copy=False)
        count = int(store_rows.size)

        pos_arr = np.asarray(table.get("pos")[store_rows], dtype=np.float32)
        vel_arr = np.asarray(table.get("vel")[store_rows], dtype=np.float32)
        mass_arr = np.asarray(table.get("mass")[store_rows], dtype=np.float32)
        pot_field = table.get("pot")
        if pot_field is not None:
            pot_arr = np.asarray(pot_field[store_rows], dtype=np.float32)
        else:
            pot_arr = np.zeros(count, dtype=np.float32)

        pos_blocks.append(pos_arr)
        vel_blocks.append(vel_arr)
        mass_blocks.append(mass_arr)
        pot_blocks.append(pot_arr)
        ptype_blocks.append(np.full(count, ptype_ints[str(ptype)], dtype=np.int32))
        local_ids = np.arange(count, dtype=np.int64) + int(offset)

        if ptype == "gas":
            payload_dm["glist"] = local_ids
        elif ptype == "dm":
            payload_dm["dmlist"] = local_ids
        elif ptype == "star":
            payload_dm["slist"] = local_ids
        elif ptype == "bh":
            payload_dm["bhlist"] = local_ids
        elif ptype == "dust":
            payload_dm["dlist"] = local_ids

        for field_name in optional_fields:
            field = table.get(str(field_name))
            if field is not None:
                payload_dm[str(field_name)] = np.asarray(field[store_rows])

        ptypes_local.append(str(ptype))
        offset += count

    payload_dm["pos"] = np.concatenate(pos_blocks, axis=0) if pos_blocks else np.empty((0, 3), dtype=np.float32)
    payload_dm["vel"] = np.concatenate(vel_blocks, axis=0) if vel_blocks else np.empty((0, 3), dtype=np.float32)
    payload_dm["mass"] = np.concatenate(mass_blocks, axis=0) if mass_blocks else np.empty(0, dtype=np.float32)
    payload_dm["pot"] = np.concatenate(pot_blocks, axis=0) if pot_blocks else np.empty(0, dtype=np.float32)
    payload_dm["ptype"] = np.concatenate(ptype_blocks, axis=0) if ptype_blocks else np.empty(0, dtype=np.int32)
    return payload_dm, ptypes_local, snapshot_meta


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
    gas_local = np.asarray(membership["gas_sel"], dtype=np.int64)
    star_local = np.asarray(membership["star_sel"], dtype=np.int64)
    bh_local = np.asarray(membership["bh_sel"], dtype=np.int64)
    dm_local = np.asarray(
        np.unique(
            np.concatenate(
                [
                    np.asarray(membership["dm_sel"], dtype=np.int64),
                    np.asarray(membership["dm2_sel"], dtype=np.int64),
                    np.asarray(membership["dm3_sel"], dtype=np.int64),
                ]
            )
        ),
        dtype=np.int64,
    )
    gas_sel = (
        np.asarray(state.particles.table("gas").get("global_index")[gas_local], dtype=np.int64)
        if gas_local.size > 0 and state.particles.has_ptype("gas")
        else np.empty(0, dtype=np.int64)
    )
    star_sel = (
        np.asarray(state.particles.table("star").get("global_index")[star_local], dtype=np.int64)
        if star_local.size > 0 and state.particles.has_ptype("star")
        else np.empty(0, dtype=np.int64)
    )
    bh_sel = (
        np.asarray(state.particles.table("bh").get("global_index")[bh_local], dtype=np.int64)
        if bh_local.size > 0 and state.particles.has_ptype("bh")
        else np.empty(0, dtype=np.int64)
    )
    dm_sel_parts = []
    for ptype_name in ("dm", "dm2", "dm3"):
        local_rows = np.asarray(membership.get(f"{ptype_name}_sel", np.empty(0, dtype=np.int64)), dtype=np.int64)
        if local_rows.size > 0 and state.particles.has_ptype(ptype_name):
            dm_sel_parts.append(
                np.asarray(state.particles.table(ptype_name).get("global_index")[local_rows], dtype=np.int64)
            )
    dm_sel = np.asarray(np.unique(np.concatenate(dm_sel_parts)) if dm_sel_parts else np.empty(0, dtype=np.int64), dtype=np.int64)

    gas_pos = (
        np.asarray(state.particles.table("gas").get("pos")[gas_local], dtype=np.float32)
        if gas_local.size > 0 and state.particles.has_ptype("gas")
        else np.empty((0, 3), dtype=np.float32)
    )
    gas_vel = (
        np.asarray(state.particles.table("gas").get("vel")[gas_local], dtype=np.float32)
        if gas_local.size > 0 and state.particles.has_ptype("gas")
        else np.empty((0, 3), dtype=np.float32)
    )
    star_pos = (
        np.asarray(state.particles.table("star").get("pos")[star_local], dtype=np.float32)
        if star_local.size > 0 and state.particles.has_ptype("star")
        else np.empty((0, 3), dtype=np.float32)
    )
    star_vel = (
        np.asarray(state.particles.table("star").get("vel")[star_local], dtype=np.float32)
        if star_local.size > 0 and state.particles.has_ptype("star")
        else np.empty((0, 3), dtype=np.float32)
    )
    bh_pos = (
        np.asarray(state.particles.table("bh").get("pos")[bh_local], dtype=np.float32)
        if bh_local.size > 0 and state.particles.has_ptype("bh")
        else np.empty((0, 3), dtype=np.float32)
    )
    bh_vel = (
        np.asarray(state.particles.table("bh").get("vel")[bh_local], dtype=np.float32)
        if bh_local.size > 0 and state.particles.has_ptype("bh")
        else np.empty((0, 3), dtype=np.float32)
    )

    eligible_pos = np.concatenate([gas_pos, star_pos, bh_pos], axis=0)
    eligible_vel = np.concatenate([gas_vel, star_vel, bh_vel], axis=0)
    if eligible_pos.shape[0] == 0:
        return None

    return {
        "gas_sel": np.asarray(gas_sel, dtype=np.int32),
        "star_sel": np.asarray(star_sel, dtype=np.int32),
        "bh_sel": np.asarray(bh_sel, dtype=np.int32),
        "dm_sel": np.asarray(dm_sel, dtype=np.int32),
        "ng": int(gas_sel.size),
        "ns": int(star_sel.size),
        "nb": int(bh_sel.size),
        "eligible_pos": eligible_pos,
        "eligible_vel": eligible_vel,
    }


def build_halo_record(state: AHFSubhaloDirectState, *, node_id: int) -> Dict[str, object]:
    idx = state.nodes.index_of(int(node_id))
    def _global_members(ptype: str) -> np.ndarray:
        local_rows = np.asarray(state.nodes.members_for(idx, ptype, dtype=np.int64), dtype=np.int64)
        if local_rows.size == 0 or not state.particles.has_ptype(ptype):
            return np.empty(0, dtype=np.int64)
        return np.asarray(state.particles.table(ptype).get("global_index")[local_rows], dtype=np.int64)

    dm_parts = [_global_members(name) for name in ("dm", "dm2", "dm3")]
    dm_parts = [part for part in dm_parts if part.size > 0]
    return {
        "AHF_haloID": int(state.nodes.halo_id[idx]),
        "AHF_parent_haloID": int(state.nodes.parent_halo_id[idx]),
        "AHF_top_haloID": int(state.nodes.top_halo_id[idx]),
        "AHF_depth": int(state.nodes.depth[idx]),
        "AHF_ancestor_haloIDs": state.nodes.ancestors_for(idx).astype(np.int64, copy=False),
        "glist": _global_members("gas"),
        "slist": _global_members("star"),
        "dmlist": np.unique(np.concatenate(dm_parts)).astype(np.int64, copy=False) if dm_parts else np.empty(0, dtype=np.int64),
        "bhlist": _global_members("bh"),
        "dlist": _global_members("dust"),
    }
