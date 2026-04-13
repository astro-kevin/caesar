from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RaggedIndexBlock:
    offsets: np.ndarray
    data: np.ndarray

    @classmethod
    def from_sequences(cls, sequences: Iterable[Sequence[int] | np.ndarray]) -> "RaggedIndexBlock":
        rows = [np.asarray(row, dtype=np.int64).reshape(-1) for row in sequences]
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        if rows:
            sizes = np.asarray([int(row.size) for row in rows], dtype=np.int64)
            offsets[1:] = np.cumsum(sizes, dtype=np.int64)
            if int(offsets[-1]) > 0:
                data = np.concatenate(rows).astype(np.int64, copy=False)
            else:
                data = np.empty(0, dtype=np.int64)
        else:
            data = np.empty(0, dtype=np.int64)
        return cls(offsets=offsets, data=data)

    def __len__(self) -> int:
        return max(0, int(self.offsets.size) - 1)

    def get(self, index: int, *, dtype=np.int64) -> np.ndarray:
        idx = int(index)
        if idx < 0 or idx + 1 >= int(self.offsets.size):
            raise IndexError(idx)
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        return np.asarray(self.data[start:end], dtype=dtype)

    def to_payload(self) -> Dict[str, np.ndarray]:
        return {
            "offsets": np.asarray(self.offsets, dtype=np.int64),
            "data": np.asarray(self.data, dtype=np.int64),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, np.ndarray]) -> "RaggedIndexBlock":
        return cls(
            offsets=np.asarray(payload["offsets"], dtype=np.int64),
            data=np.asarray(payload["data"], dtype=np.int64),
        )


@dataclass(frozen=True)
class SnapshotMeta:
    snapshot_file: str
    boxsize: float
    redshift: float
    scale_factor: float
    hubble_constant: float
    omega_matter: float
    omega_lambda: float
    omega_curvature: float
    time_gyr: float
    critical_density_msun_kpc3: float
    ptypes: tuple[str, ...]
    units: Dict[str, str]
    fields_by_ptype: Dict[str, tuple[str, ...]]
    particle_counts: Dict[str, int]

    def to_payload(self) -> Dict[str, object]:
        return {
            "snapshot_file": str(self.snapshot_file),
            "boxsize": float(self.boxsize),
            "redshift": float(self.redshift),
            "scale_factor": float(self.scale_factor),
            "hubble_constant": float(self.hubble_constant),
            "omega_matter": float(self.omega_matter),
            "omega_lambda": float(self.omega_lambda),
            "omega_curvature": float(self.omega_curvature),
            "time_gyr": float(self.time_gyr),
            "critical_density_msun_kpc3": float(self.critical_density_msun_kpc3),
            "ptypes": tuple(self.ptypes),
            "units": dict(self.units),
            "fields_by_ptype": {str(k): tuple(v) for k, v in self.fields_by_ptype.items()},
            "particle_counts": {str(k): int(v) for k, v in self.particle_counts.items()},
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "SnapshotMeta":
        return cls(
            snapshot_file=str(payload["snapshot_file"]),
            boxsize=float(payload["boxsize"]),
            redshift=float(payload["redshift"]),
            scale_factor=float(payload["scale_factor"]),
            hubble_constant=float(payload["hubble_constant"]),
            omega_matter=float(payload["omega_matter"]),
            omega_lambda=float(payload["omega_lambda"]),
            omega_curvature=float(payload["omega_curvature"]),
            time_gyr=float(payload["time_gyr"]),
            critical_density_msun_kpc3=float(payload["critical_density_msun_kpc3"]),
            ptypes=tuple(str(v) for v in payload.get("ptypes", ())),
            units={str(k): str(v) for k, v in dict(payload.get("units", {})).items()},
            fields_by_ptype={
                str(k): tuple(str(x) for x in v)
                for k, v in dict(payload.get("fields_by_ptype", {})).items()
            },
            particle_counts={str(k): int(v) for k, v in dict(payload.get("particle_counts", {})).items()},
        )


@dataclass
class ParticleTable:
    fields: Dict[str, np.ndarray]

    def __len__(self) -> int:
        ids = self.fields.get("particle_id")
        return int(np.asarray(ids).size) if ids is not None else 0

    def has(self, name: str) -> bool:
        return str(name) in self.fields

    def get(self, name: str, default=None):
        return self.fields.get(str(name), default)

    def field_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.fields.keys()))

    def to_payload(self) -> Dict[str, np.ndarray]:
        return {str(k): np.asarray(v) for k, v in self.fields.items()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, np.ndarray]) -> "ParticleTable":
        return cls(fields={str(k): np.asarray(v) for k, v in payload.items()})


@dataclass
class ParticleShard:
    meta: SnapshotMeta
    tables: Dict[str, ParticleTable]

    @property
    def ptypes(self) -> tuple[str, ...]:
        return tuple(sorted(self.tables.keys()))

    def table(self, ptype: str) -> ParticleTable:
        return self.tables[str(ptype)]

    def has_ptype(self, ptype: str) -> bool:
        return str(ptype) in self.tables

    def to_payload(self) -> Dict[str, object]:
        return {
            "meta": self.meta.to_payload(),
            "tables": {str(k): table.to_payload() for k, table in self.tables.items()},
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ParticleShard":
        return cls(
            meta=SnapshotMeta.from_payload(dict(payload["meta"])),
            tables={
                str(k): ParticleTable.from_payload(dict(v))
                for k, v in dict(payload.get("tables", {})).items()
            },
        )


@dataclass
class AHFNodeTable:
    halo_id: np.ndarray
    parent_halo_id: np.ndarray
    top_halo_id: np.ndarray
    depth: np.ndarray
    star_count: np.ndarray
    dm_count: np.ndarray
    fof_candidates: np.ndarray
    ancestors: RaggedIndexBlock
    members_by_type: Dict[str, RaggedIndexBlock]

    def __len__(self) -> int:
        return int(self.halo_id.size)

    def index_of(self, halo_id: int) -> int:
        hid = int(halo_id)
        matches = np.where(self.halo_id == hid)[0]
        if matches.size == 0:
            raise KeyError(hid)
        return int(matches[0])

    def ancestors_for(self, index: int) -> np.ndarray:
        return self.ancestors.get(index, dtype=np.int64)

    def members_for(self, index: int, ptype: str, *, dtype=np.int64) -> np.ndarray:
        block = self.members_by_type.get(str(ptype))
        if block is None:
            return np.empty(0, dtype=dtype)
        return block.get(index, dtype=dtype)

    def to_payload(self) -> Dict[str, object]:
        return {
            "halo_id": np.asarray(self.halo_id, dtype=np.int64),
            "parent_halo_id": np.asarray(self.parent_halo_id, dtype=np.int64),
            "top_halo_id": np.asarray(self.top_halo_id, dtype=np.int64),
            "depth": np.asarray(self.depth, dtype=np.int64),
            "star_count": np.asarray(self.star_count, dtype=np.int64),
            "dm_count": np.asarray(self.dm_count, dtype=np.int64),
            "fof_candidates": np.asarray(self.fof_candidates, dtype=np.int64),
            "ancestors": self.ancestors.to_payload(),
            "members_by_type": {
                str(k): block.to_payload()
                for k, block in self.members_by_type.items()
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "AHFNodeTable":
        return cls(
            halo_id=np.asarray(payload["halo_id"], dtype=np.int64),
            parent_halo_id=np.asarray(payload["parent_halo_id"], dtype=np.int64),
            top_halo_id=np.asarray(payload["top_halo_id"], dtype=np.int64),
            depth=np.asarray(payload["depth"], dtype=np.int64),
            star_count=np.asarray(payload["star_count"], dtype=np.int64),
            dm_count=np.asarray(payload["dm_count"], dtype=np.int64),
            fof_candidates=np.asarray(payload["fof_candidates"], dtype=np.int64),
            ancestors=RaggedIndexBlock.from_payload(dict(payload["ancestors"])),
            members_by_type={
                str(k): RaggedIndexBlock.from_payload(dict(v))
                for k, v in dict(payload.get("members_by_type", {})).items()
            },
        )


@dataclass
class CandidateGalaxyTable:
    ahf_halo_id: np.ndarray
    ahf_parent_halo_id: np.ndarray
    ahf_top_halo_id: np.ndarray
    ahf_depth: np.ndarray
    ahf_ancestors: RaggedIndexBlock
    members_by_type: Dict[str, RaggedIndexBlock]

    def __len__(self) -> int:
        return int(self.ahf_halo_id.size)

    def to_payload(self) -> Dict[str, object]:
        return {
            "ahf_halo_id": np.asarray(self.ahf_halo_id, dtype=np.int64),
            "ahf_parent_halo_id": np.asarray(self.ahf_parent_halo_id, dtype=np.int64),
            "ahf_top_halo_id": np.asarray(self.ahf_top_halo_id, dtype=np.int64),
            "ahf_depth": np.asarray(self.ahf_depth, dtype=np.int64),
            "ahf_ancestors": self.ahf_ancestors.to_payload(),
            "members_by_type": {
                str(k): block.to_payload()
                for k, block in self.members_by_type.items()
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CandidateGalaxyTable":
        return cls(
            ahf_halo_id=np.asarray(payload["ahf_halo_id"], dtype=np.int64),
            ahf_parent_halo_id=np.asarray(payload["ahf_parent_halo_id"], dtype=np.int64),
            ahf_top_halo_id=np.asarray(payload["ahf_top_halo_id"], dtype=np.int64),
            ahf_depth=np.asarray(payload["ahf_depth"], dtype=np.int64),
            ahf_ancestors=RaggedIndexBlock.from_payload(dict(payload["ahf_ancestors"])),
            members_by_type={
                str(k): RaggedIndexBlock.from_payload(dict(v))
                for k, v in dict(payload.get("members_by_type", {})).items()
            },
        )


def candidate_records_to_table(records: Sequence[Mapping[str, object]]) -> CandidateGalaxyTable:
    rows = list(records)
    return CandidateGalaxyTable(
        ahf_halo_id=np.asarray([int(row.get("AHF_haloID", -1)) for row in rows], dtype=np.int64),
        ahf_parent_halo_id=np.asarray([int(row.get("AHF_parent_haloID", -1)) for row in rows], dtype=np.int64),
        ahf_top_halo_id=np.asarray([int(row.get("AHF_top_haloID", -1)) for row in rows], dtype=np.int64),
        ahf_depth=np.asarray([int(row.get("AHF_depth", 0)) for row in rows], dtype=np.int64),
        ahf_ancestors=RaggedIndexBlock.from_sequences(
            np.asarray(row.get("AHF_ancestor_haloIDs", []), dtype=np.int64)
            for row in rows
        ),
        members_by_type={
            "gas": RaggedIndexBlock.from_sequences(
                np.asarray(row.get("glist", []), dtype=np.int64) for row in rows
            ),
            "star": RaggedIndexBlock.from_sequences(
                np.asarray(row.get("slist", []), dtype=np.int64) for row in rows
            ),
            "dm": RaggedIndexBlock.from_sequences(
                np.asarray(row.get("dmlist", []), dtype=np.int64) for row in rows
            ),
            "bh": RaggedIndexBlock.from_sequences(
                np.asarray(row.get("bhlist", []), dtype=np.int64) for row in rows
            ),
            "dust": RaggedIndexBlock.from_sequences(
                np.asarray(row.get("dlist", []), dtype=np.int64) for row in rows
            ),
        },
    )


def candidate_records_from_table_payload(payload: Mapping[str, object]) -> list[Dict[str, np.ndarray | int]]:
    table = CandidateGalaxyTable.from_payload(payload)
    out: list[Dict[str, np.ndarray | int]] = []
    for idx in range(len(table)):
        out.append(
            {
                "AHF_haloID": int(table.ahf_halo_id[idx]),
                "AHF_parent_haloID": int(table.ahf_parent_halo_id[idx]),
                "AHF_top_haloID": int(table.ahf_top_halo_id[idx]),
                "AHF_depth": int(table.ahf_depth[idx]),
                "AHF_ancestor_haloIDs": table.ahf_ancestors.get(idx, dtype=np.int64),
                "glist": table.members_by_type["gas"].get(idx, dtype=np.int32),
                "slist": table.members_by_type["star"].get(idx, dtype=np.int32),
                "dmlist": table.members_by_type["dm"].get(idx, dtype=np.int32),
                "bhlist": table.members_by_type["bh"].get(idx, dtype=np.int32),
                "dlist": table.members_by_type["dust"].get(idx, dtype=np.int32),
            }
        )
    return out
