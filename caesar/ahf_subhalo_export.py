from __future__ import annotations

import os
import pickle
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Mapping, Optional, Sequence

import h5py
import numpy as np

from caesar.AHF_subhalo import (
    _ShardYTUnitHelper,
    _effective_resolution_from_ndm,
    _mean_interparticle_separation_from_boxsize,
)
from caesar.saver import serialize_global_attribs
from caesar.simulation_attributes import SimulationAttributes

_LEGACY_BLACKLIST = {
    "G",
    "initial_mass",
    "valid",
    "vel_conversion",
    "unbound_particles",
    "_units",
    "unit_registry_json",
    "unbound_indexes",
    "lists",
    "dicts",
}


@dataclass
class _ValueSpec:
    shape: tuple[int, ...]
    dtype: np.dtype
    unit: Optional[str] = None


@dataclass
class _GroupSchema:
    size: int
    list_attrs: tuple[str, ...]
    attr_specs: Dict[str, _ValueSpec] = field(default_factory=dict)
    dict_specs: Dict[str, Dict[str, _ValueSpec]] = field(default_factory=dict)
    list_lengths: Dict[str, np.ndarray] = field(default_factory=dict)
    list_dtypes: Dict[str, np.dtype] = field(default_factory=dict)


def _load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def _raw_value(value):
    return getattr(value, "d", getattr(value, "value", value))


def _unit_str(value) -> Optional[str]:
    units = getattr(value, "units", None)
    if units is None:
        return None
    return str(units)


def _infer_value_spec(value) -> Optional[_ValueSpec]:
    raw = _raw_value(value)
    arr = np.asarray(raw)
    if arr.dtype.kind not in {"i", "u", "f", "b"}:
        return None
    return _ValueSpec(
        shape=tuple(int(v) for v in arr.shape),
        dtype=np.asarray(arr).dtype,
        unit=_unit_str(value),
    )


def _infer_legacy_attr_spec(name: str, value) -> Optional[_ValueSpec]:
    if name in _LEGACY_BLACKLIST:
        return None

    unit = _unit_str(value)
    if unit is not None:
        return _infer_value_spec(value)

    if isinstance(value, np.ndarray):
        shape = np.shape(value)
        if len(shape) > 0 and int(shape[0]) == 3 and "list" not in str(name):
            arr = np.asarray(value)
            if arr.dtype.kind in {"i", "u", "f", "b"}:
                return _ValueSpec(
                    shape=tuple(int(v) for v in arr.shape),
                    dtype=arr.dtype,
                    unit=None,
                )
        return None

    if isinstance(value, (int, float, bool, np.number)):
        arr = np.asarray(value)
        return _ValueSpec(shape=tuple(int(v) for v in arr.shape), dtype=arr.dtype, unit=None)

    return None


def _legacy_schema_from_state(
    state: Optional[Mapping[str, object]],
    *,
    list_attrs: Sequence[str],
    skip_attrs: set[str],
) -> tuple[Dict[str, _ValueSpec], Dict[str, Dict[str, _ValueSpec]]]:
    if not state:
        return {}, {}

    list_attr_set = {str(v) for v in list_attrs}
    attr_specs: Dict[str, _ValueSpec] = {}
    dict_specs: Dict[str, Dict[str, _ValueSpec]] = {}
    for key, value in state.items():
        key = str(key)
        if key in skip_attrs or key in list_attr_set or key in _LEGACY_BLACKLIST:
            continue
        if isinstance(value, dict):
            bucket: Dict[str, _ValueSpec] = {}
            for subkey, subvalue in value.items():
                spec = _infer_value_spec(subvalue)
                if spec is not None:
                    bucket[str(subkey)] = spec
            if bucket:
                dict_specs[key] = bucket
            continue

        spec = _infer_legacy_attr_spec(key, value)
        if spec is not None:
            attr_specs[key] = spec
    return attr_specs, dict_specs


def _zero_for_spec(spec: _ValueSpec):
    if not spec.shape:
        return np.asarray(0, dtype=spec.dtype)
    return np.zeros(spec.shape, dtype=spec.dtype)


def _convert_for_spec(value, spec: _ValueSpec):
    if value is None:
        return _zero_for_spec(spec)
    raw = np.asarray(_raw_value(value), dtype=spec.dtype)
    if tuple(int(v) for v in raw.shape) != spec.shape:
        if raw.ndim == 0 and spec.shape:
            return np.zeros(spec.shape, dtype=spec.dtype)
        return np.reshape(raw, spec.shape)
    return raw


def _value_spec_to_payload(spec: _ValueSpec) -> Dict[str, object]:
    return {
        "shape": tuple(int(v) for v in spec.shape),
        "dtype": np.dtype(spec.dtype).str,
        "unit": None if spec.unit is None else str(spec.unit),
    }


def _value_spec_from_payload(payload: Mapping[str, object]) -> _ValueSpec:
    return _ValueSpec(
        shape=tuple(int(v) for v in payload.get("shape", ())),
        dtype=np.dtype(str(payload.get("dtype", np.dtype(np.float64).str))),
        unit=None if payload.get("unit", None) in (None, "") else str(payload.get("unit")),
    )


def _merge_value_spec(existing: Optional[_ValueSpec], incoming: _ValueSpec, *, label: str) -> _ValueSpec:
    if existing is None:
        return incoming
    if (
        tuple(int(v) for v in existing.shape) != tuple(int(v) for v in incoming.shape)
        or np.dtype(existing.dtype) != np.dtype(incoming.dtype)
        or (None if existing.unit in (None, "") else str(existing.unit))
        != (None if incoming.unit in (None, "") else str(incoming.unit))
    ):
        raise RuntimeError(
            f"Incompatible shard schema for {label}: "
            f"existing(shape={existing.shape}, dtype={np.dtype(existing.dtype).str}, unit={existing.unit}) "
            f"!= incoming(shape={incoming.shape}, dtype={np.dtype(incoming.dtype).str}, unit={incoming.unit})"
        )
    return existing


def _build_column_array(states: Sequence[Mapping[str, object]], *, getter, spec: _ValueSpec) -> np.ndarray:
    out = np.empty((len(states),) + tuple(int(v) for v in spec.shape), dtype=spec.dtype)
    for idx, state in enumerate(states):
        out[idx] = _convert_for_spec(getter(state), spec)
    return out


def build_group_column_payload(
    states: Sequence[Mapping[str, object]],
    *,
    list_attrs: Sequence[str],
    skip_attrs: set[str],
) -> Dict[str, object]:
    schema = _GroupSchema(
        size=len(states),
        list_attrs=tuple(str(v) for v in list_attrs),
        list_lengths={str(name): np.zeros(len(states), dtype=np.int64) for name in list_attrs},
    )
    for idx, state in enumerate(states):
        _scan_state_schema(state, schema=schema, skip_attrs=skip_attrs, list_lengths_index=int(idx))

    ids = np.asarray([int(state["id"]) for state in states], dtype=np.int64)
    attrs = {
        str(name): _build_column_array(states, getter=lambda state, key=name: state.get(str(key)), spec=spec)
        for name, spec in schema.attr_specs.items()
    }
    dicts = {
        str(dict_name): {
            str(subkey): _build_column_array(
                states,
                getter=lambda state, dn=dict_name, sk=subkey: state.get(str(dn), {}).get(str(sk)),
                spec=spec,
            )
            for subkey, spec in submap.items()
        }
        for dict_name, submap in schema.dict_specs.items()
    }
    return {
        "id": ids,
        "attr_specs": {str(name): _value_spec_to_payload(spec) for name, spec in schema.attr_specs.items()},
        "dict_specs": {
            str(dict_name): {str(subkey): _value_spec_to_payload(spec) for subkey, spec in submap.items()}
            for dict_name, submap in schema.dict_specs.items()
        },
        "list_lengths": {str(name): np.asarray(lengths, dtype=np.int64) for name, lengths in schema.list_lengths.items()},
        "list_dtypes": {str(name): np.dtype(dtype).str for name, dtype in schema.list_dtypes.items()},
        "attrs": attrs,
        "dicts": dicts,
    }


def _merge_group_schema_from_summary(
    schema: _GroupSchema,
    summary_entry: Mapping[str, object],
    *,
    label: str,
) -> None:
    for name, payload in dict(summary_entry.get("attr_specs", {})).items():
        incoming = _value_spec_from_payload(payload)
        schema.attr_specs[str(name)] = _merge_value_spec(schema.attr_specs.get(str(name)), incoming, label=f"{label}.{name}")
    for dict_name, submap in dict(summary_entry.get("dict_specs", {})).items():
        bucket = schema.dict_specs.setdefault(str(dict_name), {})
        for subkey, payload in dict(submap).items():
            incoming = _value_spec_from_payload(payload)
            bucket[str(subkey)] = _merge_value_spec(
                bucket.get(str(subkey)),
                incoming,
                label=f"{label}.{dict_name}.{subkey}",
            )


def _scan_state_schema(
    state: Mapping[str, object],
    *,
    schema: _GroupSchema,
    skip_attrs: set[str],
    list_lengths_index: int,
) -> None:
    for attr in schema.list_attrs:
        arr = np.asarray(_state_list_data(state, attr))
        schema.list_lengths[attr][int(list_lengths_index)] = int(arr.size)
        if attr not in schema.list_dtypes and arr.size > 0:
            schema.list_dtypes[attr] = arr.dtype

    for key, value in state.items():
        if key in skip_attrs or key in schema.list_attrs:
            continue
        if isinstance(value, dict):
            bucket = schema.dict_specs.setdefault(str(key), {})
            for subkey, subvalue in value.items():
                if str(subkey) in bucket:
                    continue
                spec = _infer_value_spec(subvalue)
                if spec is not None:
                    bucket[str(subkey)] = spec
        else:
            if str(key) in schema.attr_specs:
                continue
            spec = _infer_legacy_attr_spec(str(key), value)
            if spec is not None:
                schema.attr_specs[str(key)] = spec


def _state_list_data(state: Mapping[str, object], name: str):
    if name in state:
        return state[name]
    private = f"_{name}"
    return state.get(private, [])


def _compute_starts_ends(lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.zeros(lengths.size, dtype=np.int64)
    if lengths.size > 1:
        starts[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
    ends = starts + lengths
    return starts, ends


def _progress_style() -> str:
    style = os.environ.get("CAESAR_AHF_SUBHALO_PROGRESS_STYLE", "text").strip().lower()
    if style in {"bar", "bars", "progress", "progressbar"}:
        return "bar"
    return "text"


def _progress_bar(completed: int, total: int) -> str:
    try:
        width = max(10, int(os.environ.get("CAESAR_AHF_SUBHALO_PROGRESS_BAR_WIDTH", "28")))
    except Exception:
        width = 28
    total_int = max(0, int(total))
    done_int = max(0, int(completed))
    if total_int <= 0:
        filled = 0
    else:
        filled = int(round(min(1.0, max(0.0, float(done_int) / float(total_int))) * float(width)))
    filled = max(0, min(int(width), int(filled)))
    return "[" + "#" * filled + "-" * (int(width) - filled) + "]"


def _global_property_workers() -> int:
    try:
        workers = int(os.environ.get("CAESAR_AHF_SUBHALO_GLOBAL_PROPERTY_WORKERS", "-1"))
    except Exception:
        workers = -1
    if workers == 0:
        return 1
    return workers


def _find_state_by_id(
    shard_paths: Sequence[Path],
    *,
    group_key: str,
    target_id: Optional[int],
):
    if target_id is None:
        return None
    wanted = int(target_id)
    for shard_path in shard_paths:
        shard = _load_pickle(shard_path)
        for state in shard.get(group_key, []):
            if int(state["id"]) == wanted:
                return state
    return None


def _stream_reverse_map_dataset(
    hd: h5py.Group,
    *,
    dataset_name: str,
    size: int,
    entries: Iterable[tuple[np.ndarray, int]],
    temp_dir: str,
) -> None:
    size = int(size)
    if size <= 0:
        return
    fd, tmp_path = tempfile.mkstemp(prefix=f"{dataset_name}_", suffix=".tmp", dir=temp_dir)
    os.close(fd)
    try:
        arr = np.memmap(tmp_path, dtype=np.int32, mode="w+", shape=(size,))
        arr[:] = -1
        for members, gid in entries:
            idx = np.asarray(members, dtype=np.int64)
            if idx.size == 0:
                continue
            arr[idx] = int(gid)
        arr.flush()
        hd.create_dataset(dataset_name, data=arr)
        del arr
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _compute_local_density_dicts(
    *,
    pos: np.ndarray,
    mass: np.ndarray,
    boxsize: float,
    search_radii: Sequence[float],
    workers: Optional[int] = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if pos.shape[0] == 0:
        return {}, {}

    try:
        from scipy.spatial import cKDTree
    except Exception:
        return {}, {}

    query_workers = _global_property_workers() if workers is None else int(workers)
    tree = cKDTree(np.mod(pos, boxsize), boxsize=boxsize if boxsize > 0 else None)
    mass_out: Dict[str, np.ndarray] = {}
    number_out: Dict[str, np.ndarray] = {}
    for radius in search_radii:
        r = float(radius)
        if r <= 0.0:
            continue
        volume = (4.0 / 3.0) * np.pi * r**3
        neighbours = tree.query_ball_point(pos, r, workers=query_workers)
        key = str(int(r))
        local_mass = np.empty(pos.shape[0], dtype=np.float64)
        local_num = np.empty(pos.shape[0], dtype=np.float64)
        for i, ids in enumerate(neighbours):
            idx = np.asarray(ids, dtype=np.int64)
            local_mass[i] = float(np.sum(mass[idx])) / volume
            local_num[i] = float(idx.size) / volume
        mass_out[key] = local_mass
        number_out[key] = local_num
    return mass_out, number_out


def compute_global_properties_from_property_shards(
    *,
    snapshot_meta,
    property_results: Sequence[Mapping[str, object]],
    log_fn=None,
    stage_label: str = "global properties",
) -> Dict[str, object]:
    shard_paths = [Path(str(result["shard_path"])) for result in property_results]
    nhalos = int(sum(int(result.get("count_halos", 0)) for result in property_results))
    ngalaxies = int(sum(int(result.get("count_galaxies", 0)) for result in property_results))

    halo_total_mass = np.zeros(nhalos, dtype=np.float64)
    halo_pos = np.zeros((nhalos, 3), dtype=np.float64)
    galaxy_stellar_mass = np.zeros(ngalaxies, dtype=np.float64)
    galaxy_total_mass = np.zeros(ngalaxies, dtype=np.float64)
    galaxy_pos = np.zeros((ngalaxies, 3), dtype=np.float64)

    summary_ready = all(
        isinstance(result.get("summary"), Mapping)
        and isinstance(dict(result.get("summary", {})).get("halos", {}), Mapping)
        and isinstance(dict(result.get("summary", {})).get("galaxies", {}), Mapping)
        for result in property_results
    )
    if log_fn is not None:
        log_fn(f"{stage_label}: gathering neighbour-search inputs halos={nhalos} galaxies={ngalaxies}")

    if not summary_ready:
        raise RuntimeError("Property shard summaries are required for global property calculation")

    for result in property_results:
        summary = result.get("summary", {})
        halo_summary = summary.get("halos", {})
        halo_ids = np.asarray(halo_summary.get("id", []), dtype=np.int64)
        if halo_ids.size > 0:
            halo_total_mass[halo_ids] = np.asarray(halo_summary.get("total_mass", []), dtype=np.float64)
            halo_pos[halo_ids] = np.asarray(halo_summary.get("pos", []), dtype=np.float64)

        galaxy_summary = summary.get("galaxies", {})
        galaxy_ids = np.asarray(galaxy_summary.get("id", []), dtype=np.int64)
        if galaxy_ids.size > 0:
            galaxy_stellar_mass[galaxy_ids] = np.asarray(galaxy_summary.get("stellar_mass", []), dtype=np.float64)
            galaxy_total_mass[galaxy_ids] = np.asarray(galaxy_summary.get("total_mass", []), dtype=np.float64)
            galaxy_pos[galaxy_ids] = np.asarray(galaxy_summary.get("pos", []), dtype=np.float64)

    halo_order = np.argsort(-halo_total_mass, kind="stable")
    galaxy_order = np.argsort(-galaxy_stellar_mass, kind="stable")
    halo_pos_sorted = halo_pos[halo_order]
    halo_mass_sorted = halo_total_mass[halo_order]
    galaxy_pos_sorted = galaxy_pos[galaxy_order]
    galaxy_mass_sorted = galaxy_total_mass[galaxy_order]
    search_radii = np.asarray([300.0, 1000.0, 3000.0], dtype=np.float64)
    workers = _global_property_workers()
    if log_fn is not None:
        worker_label = "all available cores" if int(workers) < 0 else str(int(workers))
        log_fn(f"{stage_label}: computing neighbour-based densities with workers={worker_label}")
    halo_local_mass, halo_local_num = _compute_local_density_dicts(
        pos=halo_pos_sorted,
        mass=halo_mass_sorted,
        boxsize=float(snapshot_meta.boxsize),
        search_radii=search_radii,
        workers=workers,
    )
    galaxy_local_mass, galaxy_local_num = _compute_local_density_dicts(
        pos=galaxy_pos_sorted,
        mass=galaxy_mass_sorted,
        boxsize=float(snapshot_meta.boxsize),
        search_radii=search_radii,
        workers=workers,
    )
    if log_fn is not None:
        log_fn(f"{stage_label}: neighbour-based densities complete")
    return {
        "halo_order": np.asarray(halo_order, dtype=np.int64),
        "galaxy_order": np.asarray(galaxy_order, dtype=np.int64),
        "halo_local_mass": halo_local_mass,
        "halo_local_num": halo_local_num,
        "galaxy_local_mass": galaxy_local_mass,
        "galaxy_local_num": galaxy_local_num,
    }


def _build_save_stub(snapshot_meta, *, nhalos: int, ngalaxies: int):
    helper = _ShardYTUnitHelper(redshift=float(snapshot_meta.redshift))
    obj = SimpleNamespace()
    obj.yt_dataset = helper
    obj.units = dict(snapshot_meta.units)
    obj.nhalos = int(nhalos)
    obj.ngalaxies = int(ngalaxies)
    obj.nclouds = 0
    obj._ahf_matched = True
    obj._include_dm_in_galaxies = True
    obj.load_pot = True
    obj.load_haloid = False
    obj.skip_hash_check = True
    obj.nproc = 1

    sim = SimulationAttributes()
    z = float(snapshot_meta.redshift)
    a = float(snapshot_meta.scale_factor)
    om0 = float(snapshot_meta.omega_matter)
    ol0 = float(snapshot_meta.omega_lambda)
    ok0 = float(snapshot_meta.omega_curvature)
    ez = float(np.sqrt(ol0 + ok0 * (1.0 + z) ** 2 + om0 * (1.0 + z) ** 3))
    omz = float(om0 * (1.0 + z) ** 3 / (ez * ez)) if ez > 0.0 else om0
    hz_s = float(snapshot_meta.hubble_constant) * 100.0 * 3.24077929e-20 * ez

    sim.ds_type = "GadgetHDF5Dataset"
    sim.cosmological_simulation = True
    sim.XH = 0.76
    sim.redshift = z
    sim.scale_factor = a
    sim.time = helper.quan(float(snapshot_meta.time_gyr), "Gyr")
    sim.omega_baryon = 0.0
    sim.omega_matter = om0
    sim.omega_lambda = ol0
    sim.fullpath = os.path.dirname(str(snapshot_meta.snapshot_file))
    sim.basename = os.path.basename(str(snapshot_meta.snapshot_file))
    sim.parameters = {}
    sim.boxsize = helper.quan(float(snapshot_meta.boxsize), snapshot_meta.units["length"])
    sim.boxsize_units = str(sim.boxsize.units)
    sim.search_radius = helper.arr([300.0, 1000.0, 3000.0], snapshot_meta.units["length"])
    sim.E_z = ez
    sim.Om_z = omz
    sim.H_z = helper.quan(hz_s, "1/s")
    sim.G = helper.quan(4.51691362044e-39, "kpc**3/(Msun * s**2)")
    sim.critical_density = helper.quan(float(snapshot_meta.critical_density_msun_kpc3), "Msun/kpc**3")
    sim.Densities = helper.arr(
        np.asarray(
            [
                200.0 * float(snapshot_meta.critical_density_msun_kpc3),
                500.0 * float(snapshot_meta.critical_density_msun_kpc3),
                2500.0 * float(snapshot_meta.critical_density_msun_kpc3),
            ],
            dtype=np.float64,
        ),
        "Msun/kpc**3",
    )
    sim.ngas = int(snapshot_meta.particle_counts.get("gas", 0))
    sim.nstar = int(snapshot_meta.particle_counts.get("star", 0))
    sim.nbh = int(snapshot_meta.particle_counts.get("bh", 0))
    sim.ndust = int(snapshot_meta.particle_counts.get("dust", 0))
    sim.ndm = int(snapshot_meta.particle_counts.get("dm", 0))
    sim.ndm2 = int(snapshot_meta.particle_counts.get("dm2", 0))
    sim.ndm3 = int(snapshot_meta.particle_counts.get("dm3", 0))
    sim.ntot = int(sum(int(v) for v in snapshot_meta.particle_counts.values()))
    sim.hubble_constant = float(snapshot_meta.hubble_constant)
    sim.baryons_present = bool(sim.ngas > 0 or sim.nstar > 0)
    sim.unbind_halos = False
    sim.effective_resolution = int(_effective_resolution_from_ndm(sim.ndm))
    sim.mean_interparticle_separation = helper.quan(
        _mean_interparticle_separation_from_boxsize(float(snapshot_meta.boxsize), sim.ndm),
        snapshot_meta.units["length"],
    )
    obj.simulation = sim
    return obj


def write_catalogue_from_property_shards(
    *,
    snapshot_meta,
    property_results: Sequence[Mapping[str, object]],
    output_file: str,
    log_fn=None,
    stage_label: str = "writing",
    global_properties: Optional[Mapping[str, object]] = None,
) -> None:
    shard_paths = [Path(str(result["shard_path"])) for result in property_results]
    nhalos = int(sum(int(result.get("count_halos", 0)) for result in property_results))
    ngalaxies = int(sum(int(result.get("count_galaxies", 0)) for result in property_results))
    ptypes_present = set(str(v) for v in snapshot_meta.ptypes)

    halo_list_attrs = ["dmlist"]
    if "gas" in ptypes_present:
        halo_list_attrs.append("glist")
    if "star" in ptypes_present:
        halo_list_attrs.append("slist")
    if "bh" in ptypes_present:
        halo_list_attrs.append("bhlist")
    if "dust" in ptypes_present:
        halo_list_attrs.append("dlist")

    galaxy_list_attrs = []
    if "gas" in ptypes_present:
        galaxy_list_attrs.append("glist")
    if "star" in ptypes_present:
        galaxy_list_attrs.append("slist")
    if "bh" in ptypes_present:
        galaxy_list_attrs.append("bhlist")
    if "dust" in ptypes_present:
        galaxy_list_attrs.append("dlist")
    galaxy_list_attrs.extend(["cloud_index_list", "AHF_ancestor_haloIDs"])

    halo_schema = _GroupSchema(
        size=nhalos,
        list_attrs=tuple(halo_list_attrs),
        list_lengths={name: np.zeros(nhalos, dtype=np.int64) for name in halo_list_attrs},
    )
    galaxy_schema = _GroupSchema(
        size=ngalaxies,
        list_attrs=tuple(galaxy_list_attrs),
        list_lengths={name: np.zeros(ngalaxies, dtype=np.int64) for name in galaxy_list_attrs},
    )

    halo_skip = {
        "id",
        "_merge_id",
        "obj",
        "halo",
        "galaxies",
        "satellite_galaxies",
        "clouds",
        "galaxy",
        "central_galaxy",
        "galaxy_index_list",
        "glist",
        "slist",
        "dmlist",
        "bhlist",
        "dlist",
        "cloud_index_list",
        "AHF_ancestor_haloIDs",
        "_glist",
        "_slist",
        "_dmlist",
        "_bhlist",
        "_dlist",
    }
    galaxy_skip = {
        "id",
        "_merge_id",
        "obj",
        "halo",
        "galaxies",
        "satellite_galaxies",
        "clouds",
        "galaxy",
        "central_galaxy",
        "parent_halo_index",
        "_ahf_host_halo_index",
        "glist",
        "slist",
        "dmlist",
        "bhlist",
        "dlist",
        "cloud_index_list",
        "AHF_ancestor_haloIDs",
        "_glist",
        "_slist",
        "_dmlist",
        "_bhlist",
        "_dlist",
    }

    halo_total_mass = np.zeros(nhalos, dtype=np.float64)
    halo_pos = np.zeros((nhalos, 3), dtype=np.float64)
    halo_ahf_id = np.full(nhalos, -1, dtype=np.int64)

    galaxy_stellar_mass = np.zeros(ngalaxies, dtype=np.float64)
    galaxy_total_mass = np.zeros(ngalaxies, dtype=np.float64)
    galaxy_pos = np.zeros((ngalaxies, 3), dtype=np.float64)
    galaxy_top_halo_id = np.full(ngalaxies, -1, dtype=np.int64)
    summary_ready = all(
        isinstance(result.get("summary"), Mapping)
        and isinstance(dict(result.get("summary", {})).get("halos", {}), Mapping)
        and isinstance(dict(result.get("summary", {})).get("galaxies", {}), Mapping)
        for result in property_results
    )
    if not summary_ready:
        raise RuntimeError("Property shard summaries are required for AHF-subhalo export")

    for result in property_results:
        summary = dict(result.get("summary", {}))
        halo_summary = dict(summary.get("halos", {}))
        halo_ids = np.asarray(halo_summary.get("id", []), dtype=np.int64)
        if halo_ids.size > 0:
            halo_total_mass[halo_ids] = np.asarray(halo_summary.get("total_mass", []), dtype=np.float64)
            halo_pos[halo_ids] = np.asarray(halo_summary.get("pos", []), dtype=np.float64)
            halo_ahf_id[halo_ids] = np.asarray(halo_summary.get("ahf_halo_id", []), dtype=np.int64)
            for name, lengths in dict(halo_summary.get("list_lengths", {})).items():
                if name in halo_schema.list_lengths:
                    halo_schema.list_lengths[name][halo_ids] = np.asarray(lengths, dtype=np.int64)
            for name, dtype_str in dict(halo_summary.get("list_dtypes", {})).items():
                if name in halo_schema.list_lengths and name not in halo_schema.list_dtypes:
                    halo_schema.list_dtypes[name] = np.dtype(dtype_str)
        _merge_group_schema_from_summary(halo_schema, halo_summary, label="halo")

        galaxy_summary = dict(summary.get("galaxies", {}))
        galaxy_ids = np.asarray(galaxy_summary.get("id", []), dtype=np.int64)
        if galaxy_ids.size > 0:
            galaxy_stellar_mass[galaxy_ids] = np.asarray(galaxy_summary.get("stellar_mass", []), dtype=np.float64)
            galaxy_total_mass[galaxy_ids] = np.asarray(galaxy_summary.get("total_mass", []), dtype=np.float64)
            galaxy_pos[galaxy_ids] = np.asarray(galaxy_summary.get("pos", []), dtype=np.float64)
            galaxy_top_halo_id[galaxy_ids] = np.asarray(galaxy_summary.get("top_halo_id", []), dtype=np.int64)
            for name, lengths in dict(galaxy_summary.get("list_lengths", {})).items():
                if name in galaxy_schema.list_lengths:
                    galaxy_schema.list_lengths[name][galaxy_ids] = np.asarray(lengths, dtype=np.int64)
            for name, dtype_str in dict(galaxy_summary.get("list_dtypes", {})).items():
                if name in galaxy_schema.list_lengths and name not in galaxy_schema.list_dtypes:
                    galaxy_schema.list_dtypes[name] = np.dtype(dtype_str)
        _merge_group_schema_from_summary(galaxy_schema, galaxy_summary, label="galaxy")

    halo_order = np.argsort(-halo_total_mass, kind="stable")
    halo_new_index = np.empty(nhalos, dtype=np.int64)
    halo_new_index[halo_order] = np.arange(nhalos, dtype=np.int64)
    halo_ahf_to_new = {int(halo_ahf_id[mid]): int(halo_new_index[mid]) for mid in range(nhalos) if int(halo_ahf_id[mid]) >= 0}

    galaxy_order = np.argsort(-galaxy_stellar_mass, kind="stable")
    galaxy_new_index = np.empty(ngalaxies, dtype=np.int64)
    galaxy_new_index[galaxy_order] = np.arange(ngalaxies, dtype=np.int64)
    galaxy_parent_halo_new = np.asarray([int(halo_ahf_to_new.get(int(hid), -1)) for hid in galaxy_top_halo_id], dtype=np.int64)

    halo_galaxy_lengths = np.zeros(nhalos, dtype=np.int64)
    for merge_id in galaxy_order.tolist():
        host = int(galaxy_parent_halo_new[int(merge_id)])
        if host >= 0:
            halo_galaxy_lengths[host] += 1
    halo_galaxy_starts, halo_galaxy_ends = _compute_starts_ends(halo_galaxy_lengths)
    halo_galaxy_data = np.empty(int(halo_galaxy_lengths.sum()), dtype=np.int64)
    cursor = halo_galaxy_starts.copy()
    galaxy_central = np.zeros(ngalaxies, dtype=np.bool_)
    for merge_id in galaxy_order.tolist():
        host = int(galaxy_parent_halo_new[int(merge_id)])
        if host < 0:
            continue
        new_gid = int(galaxy_new_index[int(merge_id)])
        pos = int(cursor[host])
        halo_galaxy_data[pos] = new_gid
        if pos == int(halo_galaxy_starts[host]):
            galaxy_central[new_gid] = True
        cursor[host] += 1

    if global_properties is not None:
        expected_halo_order = np.asarray(global_properties.get("halo_order", []), dtype=np.int64)
        expected_galaxy_order = np.asarray(global_properties.get("galaxy_order", []), dtype=np.int64)
        if expected_halo_order.size > 0 and not np.array_equal(expected_halo_order, halo_order):
            raise RuntimeError("Precomputed halo global properties do not match writer halo ordering")
        if expected_galaxy_order.size > 0 and not np.array_equal(expected_galaxy_order, galaxy_order):
            raise RuntimeError("Precomputed galaxy global properties do not match writer galaxy ordering")
        halo_local_mass = {
            str(key): np.asarray(val, dtype=np.float64)
            for key, val in dict(global_properties.get("halo_local_mass", {})).items()
        }
        halo_local_num = {
            str(key): np.asarray(val, dtype=np.float64)
            for key, val in dict(global_properties.get("halo_local_num", {})).items()
        }
        galaxy_local_mass = {
            str(key): np.asarray(val, dtype=np.float64)
            for key, val in dict(global_properties.get("galaxy_local_mass", {})).items()
        }
        galaxy_local_num = {
            str(key): np.asarray(val, dtype=np.float64)
            for key, val in dict(global_properties.get("galaxy_local_num", {})).items()
        }
    else:
        halo_pos_sorted = halo_pos[halo_order]
        halo_mass_sorted = halo_total_mass[halo_order]
        galaxy_pos_sorted = galaxy_pos[galaxy_order]
        galaxy_mass_sorted = galaxy_total_mass[galaxy_order]
        search_radii = np.asarray([300.0, 1000.0, 3000.0], dtype=np.float64)
        halo_local_mass, halo_local_num = _compute_local_density_dicts(
            pos=halo_pos_sorted,
            mass=halo_mass_sorted,
            boxsize=float(snapshot_meta.boxsize),
            search_radii=search_radii,
        )
        galaxy_local_mass, galaxy_local_num = _compute_local_density_dicts(
            pos=galaxy_pos_sorted,
            mass=galaxy_mass_sorted,
            boxsize=float(snapshot_meta.boxsize),
            search_radii=search_radii,
        )

    halo_schema.attr_specs.setdefault("GroupID", _ValueSpec(shape=(), dtype=np.int64))
    halo_schema.attr_specs.setdefault("central_galaxy", _ValueSpec(shape=(), dtype=np.int64))
    halo_schema.attr_specs.setdefault("caesar_parent_halo_index", _ValueSpec(shape=(), dtype=np.int64))
    halo_schema.attr_specs.setdefault("caesar_top_halo_index", _ValueSpec(shape=(), dtype=np.int64))
    galaxy_schema.attr_specs.setdefault("GroupID", _ValueSpec(shape=(), dtype=np.int64))
    galaxy_schema.attr_specs.setdefault("parent_halo_index", _ValueSpec(shape=(), dtype=np.int64))
    galaxy_schema.attr_specs.setdefault("_ahf_host_halo_index", _ValueSpec(shape=(), dtype=np.int64))
    galaxy_schema.attr_specs.setdefault("caesar_parent_halo_index", _ValueSpec(shape=(), dtype=np.int64))
    galaxy_schema.attr_specs.setdefault("caesar_top_halo_index", _ValueSpec(shape=(), dtype=np.int64))
    galaxy_schema.attr_specs.setdefault("central", _ValueSpec(shape=(), dtype=np.bool_))
    halo_schema.list_lengths["galaxy_index_list"] = halo_galaxy_lengths

    halo_schema.dict_specs.setdefault("local_mass_density", {})
    halo_schema.dict_specs.setdefault("local_number_density", {})
    galaxy_schema.dict_specs.setdefault("local_mass_density", {})
    galaxy_schema.dict_specs.setdefault("local_number_density", {})
    _unit_probe = _ShardYTUnitHelper(redshift=float(snapshot_meta.redshift))
    density_unit = str(
        (
            _unit_probe.quan(1.0, snapshot_meta.units["mass"])
            / _unit_probe.quan(1.0, snapshot_meta.units["length"]) ** 3
        ).units
    )
    number_density_unit = str(
        (
            1.0
            / _unit_probe.quan(1.0, snapshot_meta.units["length"]) ** 3
        ).units
    )
    for key in halo_local_mass.keys():
        halo_schema.dict_specs["local_mass_density"][key] = _ValueSpec(shape=(), dtype=np.float64, unit=density_unit)
        halo_schema.dict_specs["local_number_density"][key] = _ValueSpec(shape=(), dtype=np.float64, unit=number_density_unit)
    for key in galaxy_local_mass.keys():
        galaxy_schema.dict_specs["local_mass_density"][key] = _ValueSpec(shape=(), dtype=np.float64, unit=density_unit)
        galaxy_schema.dict_specs["local_number_density"][key] = _ValueSpec(shape=(), dtype=np.float64, unit=number_density_unit)

    halo_list_names = tuple(name for name in ("dmlist", "glist", "slist", "bhlist", "dlist", "galaxy_index_list") if name in halo_schema.list_lengths)
    galaxy_list_names = tuple(name for name in ("glist", "slist", "bhlist", "dlist", "cloud_index_list", "AHF_ancestor_haloIDs") if name in galaxy_schema.list_lengths)

    halo_ordered_lengths = {}
    for name in halo_list_names:
        lengths = np.asarray(halo_schema.list_lengths[name], dtype=np.int64)
        if name == "galaxy_index_list":
            halo_ordered_lengths[name] = lengths
        else:
            halo_ordered_lengths[name] = lengths[halo_order]

    galaxy_ordered_lengths = {
        name: np.asarray(galaxy_schema.list_lengths[name], dtype=np.int64)[galaxy_order]
        for name in galaxy_list_names
    }

    halo_starts = {name: _compute_starts_ends(halo_ordered_lengths[name])[0] for name in halo_list_names}
    halo_ends = {name: _compute_starts_ends(halo_ordered_lengths[name])[1] for name in halo_list_names}
    galaxy_starts = {name: _compute_starts_ends(galaxy_ordered_lengths[name])[0] for name in galaxy_list_names}
    galaxy_ends = {name: _compute_starts_ends(galaxy_ordered_lengths[name])[1] for name in galaxy_list_names}

    units_seen: set[str] = set()
    for spec in halo_schema.attr_specs.values():
        if spec.unit:
            units_seen.add(str(spec.unit))
    for submap in halo_schema.dict_specs.values():
        for spec in submap.values():
            if spec.unit:
                units_seen.add(str(spec.unit))
    for spec in galaxy_schema.attr_specs.values():
        if spec.unit:
            units_seen.add(str(spec.unit))
    for submap in galaxy_schema.dict_specs.values():
        for spec in submap.values():
            if spec.unit:
                units_seen.add(str(spec.unit))

    helper_obj = _build_save_stub(snapshot_meta, nhalos=nhalos, ngalaxies=ngalaxies)
    for unit in sorted(units_seen):
        if unit in ("", "dimensionless", None):
            continue
        try:
            helper_obj.yt_dataset.quan(1.0, unit)
        except Exception:
            pass
    unit_registry_json = helper_obj.yt_dataset.unit_registry.to_json()
    temp_dir = os.environ.get("CAESAR_STREAM_SAVE_TMPDIR") or (os.path.dirname(os.path.abspath(output_file)) or ".")
    os.makedirs(temp_dir, exist_ok=True)

    if log_fn is not None:
        log_fn(
            f"{stage_label}: starting final export halos={nhalos} galaxies={ngalaxies} "
            f"property_shards={len(shard_paths)}"
        )

    if os.path.isfile(output_file):
        os.remove(output_file)

    with h5py.File(output_file, "w") as outfile:
        outfile.attrs.create("caesar", 315)
        outfile.attrs.create("unit_registry_json", unit_registry_json.encode("utf8"))
        serialize_global_attribs(helper_obj, outfile)
        helper_obj.simulation._serialize(helper_obj, outfile)

        halo_group = outfile.create_group("halo_data")
        halo_lists_group = halo_group.create_group("lists")
        halo_dicts_group = halo_group.create_group("dicts")
        galaxy_group = outfile.create_group("galaxy_data")
        galaxy_lists_group = galaxy_group.create_group("lists")
        galaxy_dicts_group = galaxy_group.create_group("dicts")

        halo_list_dsets = {}
        for name in halo_list_names:
            total_len = int(halo_ends[name][-1]) if nhalos > 0 else 0
            list_dtype = schema_dtype = halo_schema.list_dtypes.get(name)
            if schema_dtype is None:
                list_dtype = np.float64 if total_len == 0 else np.int64
            halo_list_dsets[name] = halo_lists_group.create_dataset(name, shape=(total_len,), dtype=list_dtype)
            halo_group.create_dataset(f"{name}_start", data=halo_starts[name])
            halo_group.create_dataset(f"{name}_end", data=halo_ends[name])

        galaxy_list_dsets = {}
        for name in galaxy_list_names:
            total_len = int(galaxy_ends[name][-1]) if ngalaxies > 0 else 0
            list_dtype = schema_dtype = galaxy_schema.list_dtypes.get(name)
            if schema_dtype is None:
                list_dtype = np.float64 if total_len == 0 else np.int64
            galaxy_list_dsets[name] = galaxy_lists_group.create_dataset(name, shape=(total_len,), dtype=list_dtype)
            galaxy_group.create_dataset(f"{name}_start", data=galaxy_starts[name])
            galaxy_group.create_dataset(f"{name}_end", data=galaxy_ends[name])

        halo_attr_dsets = {}
        for name, spec in halo_schema.attr_specs.items():
            shape = (nhalos,) + spec.shape
            ds = halo_group.create_dataset(name, shape=shape, dtype=spec.dtype)
            if spec.unit is not None:
                ds.attrs.create("unit", str(spec.unit).encode("utf8"))
            halo_attr_dsets[name] = ds

        galaxy_attr_dsets = {}
        for name, spec in galaxy_schema.attr_specs.items():
            shape = (ngalaxies,) + spec.shape
            ds = galaxy_group.create_dataset(name, shape=shape, dtype=spec.dtype)
            if spec.unit is not None:
                ds.attrs.create("unit", str(spec.unit).encode("utf8"))
            galaxy_attr_dsets[name] = ds

        halo_dict_dsets = {}
        for dict_name, submap in halo_schema.dict_specs.items():
            halo_dict_dsets[dict_name] = {}
            for subkey, spec in submap.items():
                shape = (nhalos,) + spec.shape
                ds = halo_dicts_group.create_dataset(f"{dict_name}.{subkey}", shape=shape, dtype=spec.dtype)
                if spec.unit is not None:
                    ds.attrs.create("unit", str(spec.unit).encode("utf8"))
                halo_dict_dsets[dict_name][subkey] = ds

        galaxy_dict_dsets = {}
        for dict_name, submap in galaxy_schema.dict_specs.items():
            galaxy_dict_dsets[dict_name] = {}
            for subkey, spec in submap.items():
                shape = (ngalaxies,) + spec.shape
                ds = galaxy_dicts_group.create_dataset(f"{dict_name}.{subkey}", shape=shape, dtype=spec.dtype)
                if spec.unit is not None:
                    ds.attrs.create("unit", str(spec.unit).encode("utf8"))
                galaxy_dict_dsets[dict_name][subkey] = ds

        global_group = outfile.create_group("global_lists")
        reverse_specs = [
            ("halo_dmlist", int(getattr(helper_obj.simulation, "ndm", 0))),
            ("halo_glist", int(getattr(helper_obj.simulation, "ngas", 0))),
            ("halo_slist", int(getattr(helper_obj.simulation, "nstar", 0))),
            ("galaxy_glist", int(getattr(helper_obj.simulation, "ngas", 0))),
            ("galaxy_slist", int(getattr(helper_obj.simulation, "nstar", 0))),
        ]
        if "bhlist" in halo_list_names or "bhlist" in galaxy_list_names:
            reverse_specs.extend(
                [
                    ("halo_bhlist", int(getattr(helper_obj.simulation, "nbh", 0))),
                    ("galaxy_bhlist", int(getattr(helper_obj.simulation, "nbh", 0))),
                ]
            )
        if "dlist" in halo_list_names or "dlist" in galaxy_list_names:
            reverse_specs.extend(
                [
                    ("halo_dlist", int(getattr(helper_obj.simulation, "ndust", 0))),
                    ("galaxy_dlist", int(getattr(helper_obj.simulation, "ndust", 0))),
                ]
            )

        halo_final_lists = {
            name: np.empty(int(halo_ends[name][-1]) if nhalos > 0 else 0, dtype=halo_list_dsets[name].dtype)
            for name in halo_list_dsets
        }
        galaxy_final_lists = {
            name: np.empty(int(galaxy_ends[name][-1]) if ngalaxies > 0 else 0, dtype=galaxy_list_dsets[name].dtype)
            for name in galaxy_list_dsets
        }
        reverse_arrays = {
            name: np.full(int(size), -1, dtype=np.int32)
            for name, size in reverse_specs
            if int(size) > 0
        }

        def _apply_list_block(
            *,
            block: Mapping[str, object],
            name: str,
            owner_new_index: np.ndarray,
            starts: Mapping[str, np.ndarray],
            ends: Mapping[str, np.ndarray],
            final_arrays: Mapping[str, np.ndarray],
            reverse_name_prefix: str,
        ) -> None:
            if not isinstance(block, Mapping):
                raise RuntimeError(f"Missing required ragged block for {reverse_name_prefix}_{name}")
            arr = final_arrays[name]
            reverse_name = f"{reverse_name_prefix}_{name}"
            reverse_arr = reverse_arrays.get(reverse_name)
            owner_ids = np.asarray(block.get("owner_id", []), dtype=np.int64)
            lengths = np.asarray(block.get("lengths", []), dtype=np.int64)
            data = np.asarray(block.get("data", []), dtype=arr.dtype)
            if owner_ids.size != lengths.size:
                raise RuntimeError(f"Invalid ragged block for {reverse_name}: owner_ids and lengths size mismatch")
            if owner_ids.size == 0:
                return
            if np.any(owner_ids < 0) or np.any(owner_ids >= owner_new_index.size):
                raise RuntimeError(f"Invalid ragged block for {reverse_name}: owner_id out of bounds")
            if np.unique(owner_ids).size != owner_ids.size:
                raise RuntimeError(f"Invalid ragged block for {reverse_name}: duplicate owner_id entries")

            src_starts, src_ends = _compute_starts_ends(lengths)
            owner_new_idxs = np.asarray(owner_new_index[owner_ids], dtype=np.int64)
            for idx in np.argsort(owner_new_idxs, kind="stable").tolist():
                new_idx = int(owner_new_idxs[idx])
                dst_start = int(starts[name][new_idx])
                dst_end = int(ends[name][new_idx])
                if dst_end <= dst_start:
                    continue
                src_start = int(src_starts[idx])
                src_end = int(src_ends[idx])
                segment = data[src_start:src_end]
                expected_len = int(dst_end - dst_start)
                if int(segment.size) != expected_len:
                    raise RuntimeError(
                        f"Invalid ragged block for {reverse_name}: owner_id={int(owner_ids[idx])} "
                        f"segment length {int(segment.size)} != expected {expected_len}"
                    )
                arr[dst_start:dst_end] = segment
                if reverse_arr is not None and segment.size > 0:
                    reverse_arr[np.asarray(segment, dtype=np.int64)] = int(new_idx)

        write_progress_every = max(1, int(os.environ.get("CAESAR_AHF_SUBHALO_EXPORT_PROGRESS_EVERY", "4")))
        write_progress_seconds = max(5.0, float(os.environ.get("CAESAR_AHF_SUBHALO_EXPORT_PROGRESS_SECONDS", "60.0")))
        last_write_status = time.monotonic()
        written_halos = 0
        written_galaxies = 0

        for shard_idx, shard_path in enumerate(shard_paths, start=1):
            shard = _load_pickle(shard_path)
            halo_columns = dict(shard.get("halo_columns", {}))
            galaxy_columns = dict(shard.get("galaxy_columns", {}))
            halo_list_blocks = dict(shard.get("halo_lists", {})) if isinstance(shard.get("halo_lists", {}), Mapping) else {}
            galaxy_list_blocks = dict(shard.get("galaxy_lists", {})) if isinstance(shard.get("galaxy_lists", {}), Mapping) else {}
            halo_attrs = dict(halo_columns.get("attrs", {}))
            halo_dicts = {str(name): dict(submap) for name, submap in dict(halo_columns.get("dicts", {})).items()}
            galaxy_attrs = dict(galaxy_columns.get("attrs", {}))
            galaxy_dicts = {str(name): dict(submap) for name, submap in dict(galaxy_columns.get("dicts", {})).items()}

            halo_merge_ids = np.asarray(halo_columns.get("id", []), dtype=np.int64)
            if halo_merge_ids.size > 0:
                halo_new_idxs = np.asarray(halo_new_index[halo_merge_ids], dtype=np.int64)
                halo_order_local = np.argsort(halo_new_idxs)
                halo_merge_ids = halo_merge_ids[halo_order_local]
                halo_new_idxs = halo_new_idxs[halo_order_local]
                halo_parent_ids = np.asarray(
                    halo_attrs.get("AHF_parent_haloID", np.full(halo_merge_ids.size, -1, dtype=np.int64)),
                    dtype=np.int64,
                )[halo_order_local]
                halo_top_ids = np.asarray(
                    halo_attrs.get("AHF_top_haloID", np.full(halo_merge_ids.size, -1, dtype=np.int64)),
                    dtype=np.int64,
                )[halo_order_local]

                for name, ds in halo_attr_dsets.items():
                    if name == "GroupID":
                        ds[halo_new_idxs] = np.asarray(halo_new_idxs, dtype=ds.dtype)
                    elif name == "central_galaxy":
                        values = np.asarray(
                            [
                                int(halo_galaxy_data[int(halo_galaxy_starts[int(new_idx)])])
                                if int(halo_galaxy_ends[int(new_idx)]) > int(halo_galaxy_starts[int(new_idx)])
                                else -1
                                for new_idx in halo_new_idxs.tolist()
                            ],
                            dtype=ds.dtype,
                        )
                        ds[halo_new_idxs] = values
                    elif name == "caesar_parent_halo_index":
                        ds[halo_new_idxs] = np.asarray(
                            [int(halo_ahf_to_new.get(int(v), -1)) for v in halo_parent_ids.tolist()],
                            dtype=ds.dtype,
                        )
                    elif name == "caesar_top_halo_index":
                        ds[halo_new_idxs] = np.asarray(
                            [int(halo_ahf_to_new.get(int(v), -1)) for v in halo_top_ids.tolist()],
                            dtype=ds.dtype,
                        )
                    elif name in halo_attrs:
                        ds[halo_new_idxs] = np.asarray(halo_attrs[name], dtype=ds.dtype)[halo_order_local]

                for name in ("dmlist", "glist", "slist", "bhlist", "dlist"):
                    if name not in halo_list_dsets:
                        continue
                    _apply_list_block(
                        block=halo_list_blocks.get(name),
                        name=name,
                        owner_new_index=halo_new_index,
                        starts=halo_starts,
                        ends=halo_ends,
                        final_arrays=halo_final_lists,
                        reverse_name_prefix="halo",
                    )

                for dict_name, submap in halo_dict_dsets.items():
                    for subkey, ds in submap.items():
                        if dict_name == "local_mass_density":
                            ds[halo_new_idxs] = np.asarray(halo_local_mass[subkey], dtype=ds.dtype)[halo_new_idxs]
                        elif dict_name == "local_number_density":
                            ds[halo_new_idxs] = np.asarray(halo_local_num[subkey], dtype=ds.dtype)[halo_new_idxs]
                        elif subkey in halo_dicts.get(dict_name, {}):
                            ds[halo_new_idxs] = np.asarray(halo_dicts[dict_name][subkey], dtype=ds.dtype)[halo_order_local]

            galaxy_merge_ids = np.asarray(galaxy_columns.get("id", []), dtype=np.int64)
            if galaxy_merge_ids.size > 0:
                galaxy_new_idxs = np.asarray(galaxy_new_index[galaxy_merge_ids], dtype=np.int64)
                galaxy_order_local = np.argsort(galaxy_new_idxs)
                galaxy_merge_ids = galaxy_merge_ids[galaxy_order_local]
                galaxy_new_idxs = galaxy_new_idxs[galaxy_order_local]
                galaxy_parent_ids = np.asarray(
                    galaxy_attrs.get("AHF_parent_haloID", np.full(galaxy_merge_ids.size, -1, dtype=np.int64)),
                    dtype=np.int64,
                )[galaxy_order_local]
                galaxy_top_ids = np.asarray(
                    galaxy_attrs.get("AHF_top_haloID", np.full(galaxy_merge_ids.size, -1, dtype=np.int64)),
                    dtype=np.int64,
                )[galaxy_order_local]

                for name, ds in galaxy_attr_dsets.items():
                    if name == "GroupID":
                        ds[galaxy_new_idxs] = np.asarray(galaxy_new_idxs, dtype=ds.dtype)
                    elif name in {"parent_halo_index", "_ahf_host_halo_index"}:
                        ds[galaxy_new_idxs] = np.asarray(galaxy_parent_halo_new[galaxy_merge_ids], dtype=ds.dtype)
                    elif name == "caesar_parent_halo_index":
                        ds[galaxy_new_idxs] = np.asarray(
                            [int(halo_ahf_to_new.get(int(v), -1)) for v in galaxy_parent_ids.tolist()],
                            dtype=ds.dtype,
                        )
                    elif name == "caesar_top_halo_index":
                        ds[galaxy_new_idxs] = np.asarray(
                            [int(halo_ahf_to_new.get(int(v), -1)) for v in galaxy_top_ids.tolist()],
                            dtype=ds.dtype,
                        )
                    elif name == "central":
                        ds[galaxy_new_idxs] = np.asarray(galaxy_central[galaxy_new_idxs], dtype=ds.dtype)
                    elif name in galaxy_attrs:
                        ds[galaxy_new_idxs] = np.asarray(galaxy_attrs[name], dtype=ds.dtype)[galaxy_order_local]

                for name in ("glist", "slist", "bhlist", "dlist", "AHF_ancestor_haloIDs", "cloud_index_list"):
                    if name not in galaxy_list_dsets:
                        continue
                    _apply_list_block(
                        block=galaxy_list_blocks.get(name),
                        name=name,
                        owner_new_index=galaxy_new_index,
                        starts=galaxy_starts,
                        ends=galaxy_ends,
                        final_arrays=galaxy_final_lists,
                        reverse_name_prefix="galaxy",
                    )

                for dict_name, submap in galaxy_dict_dsets.items():
                    for subkey, ds in submap.items():
                        if dict_name == "local_mass_density":
                            ds[galaxy_new_idxs] = np.asarray(galaxy_local_mass[subkey], dtype=ds.dtype)[galaxy_new_idxs]
                        elif dict_name == "local_number_density":
                            ds[galaxy_new_idxs] = np.asarray(galaxy_local_num[subkey], dtype=ds.dtype)[galaxy_new_idxs]
                        elif subkey in galaxy_dicts.get(dict_name, {}):
                            ds[galaxy_new_idxs] = np.asarray(galaxy_dicts[dict_name][subkey], dtype=ds.dtype)[galaxy_order_local]

            written_halos += int(halo_merge_ids.size)
            written_galaxies += int(galaxy_merge_ids.size)
            now = time.monotonic()
            if log_fn is not None and (
                shard_idx == len(shard_paths)
                or shard_idx % write_progress_every == 0
                or now - last_write_status >= write_progress_seconds
            ):
                if _progress_style() == "bar":
                    log_fn(
                        f"{stage_label}: write shards {_progress_bar(shard_idx, len(shard_paths))} "
                        f"{shard_idx}/{len(shard_paths)} | halos={written_halos}/{nhalos} "
                        f"| galaxies={written_galaxies}/{ngalaxies}"
                    )
                else:
                    log_fn(
                        f"{stage_label}: writing shards {shard_idx}/{len(shard_paths)}; "
                        f"halos={written_halos}/{nhalos}, galaxies={written_galaxies}/{ngalaxies}"
                    )
                last_write_status = now

        for name, ds in halo_list_dsets.items():
            if name == "galaxy_index_list":
                ds[:] = halo_galaxy_data
            else:
                ds[:] = halo_final_lists[name]
        for name, ds in galaxy_list_dsets.items():
            ds[:] = galaxy_final_lists[name]

        reverse_total = len(reverse_specs)
        for reverse_idx, (name, size) in enumerate(reverse_specs, start=1):
            if log_fn is not None:
                if _progress_style() == "bar":
                    log_fn(
                        f"{stage_label}: global_lists "
                        f"{_progress_bar(reverse_idx, reverse_total)} {reverse_idx}/{reverse_total}"
                    )
                else:
                    log_fn(f"{stage_label}: streaming global_lists/{name}")
            if int(size) > 0:
                global_group.create_dataset(name, data=reverse_arrays[name])

    if log_fn is not None:
        log_fn(f"{stage_label}: saved catalogue to {output_file}")
