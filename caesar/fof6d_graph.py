"""Graph-based 6D-FOF utilities for AHF-subhalo workflows.

This module provides a backend-selectable graph 6D-FOF implementation
based on:
1) spatial edge construction
2) optional velocity-space edge filtering
3) connected-components on the retained graph

The velocity gate follows a symmetric formulation. For the AHF-subhalo
pipeline we use CAESAR's kernel-table weighted local velocity RMS to
construct per-particle sigma and then apply ``vel_LL * max(sigma_i,
sigma_j)`` as the pair criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import numpy as np

_CUPY_CACHED = None
_CUPY_CHECKED = False
_CUGRAPH_CACHED = None
_CUGRAPH_CHECKED = False


def _try_import_cupy():
    global _CUPY_CACHED, _CUPY_CHECKED
    if _CUPY_CHECKED:
        return _CUPY_CACHED
    _CUPY_CHECKED = True
    try:
        import cupy as cp  # type: ignore

        try:
            _ = cp.cuda.runtime.getDeviceCount()
        except Exception:
            _CUPY_CACHED = None
            return None
        _CUPY_CACHED = cp
        return cp
    except Exception:
        _CUPY_CACHED = None
        return None


def _try_import_cugraph():
    global _CUGRAPH_CACHED, _CUGRAPH_CHECKED
    if _CUGRAPH_CHECKED:
        return _CUGRAPH_CACHED
    _CUGRAPH_CHECKED = True
    try:
        import cudf  # type: ignore
        import cugraph  # type: ignore

        _CUGRAPH_CACHED = (cudf, cugraph)
        return _CUGRAPH_CACHED
    except Exception:
        _CUGRAPH_CACHED = None
        return None


def _is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "outofmemory" in exc.__class__.__name__.lower() or "out of memory" in msg


def _as_xp_array(x, xp):
    if xp is np:
        return np.asarray(x)
    return xp.asarray(x)


def _normalize_boxsize(boxsize: Optional[Sequence[float]]):
    if boxsize is None:
        return None
    if isinstance(boxsize, (int, float)):
        return (float(boxsize), float(boxsize), float(boxsize))
    if len(boxsize) != 3:
        raise ValueError("boxsize must be a scalar or length-3 sequence")
    return (float(boxsize[0]), float(boxsize[1]), float(boxsize[2]))


def _neighbor_offsets_half() -> np.ndarray:
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    offsets.append((0, 0, 0))
                    continue
                if dx > 0 or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz > 0):
                    offsets.append((dx, dy, dz))
    return np.asarray(offsets, dtype=np.int64)


OFFSETS_HALF = _neighbor_offsets_half()


@dataclass(frozen=True)
class RadiusPairs:
    u: "np.ndarray"
    v: "np.ndarray"
    r2: "np.ndarray"


def radius_pairs_grid(
    pos: np.ndarray,
    radius: float,
    *,
    boxsize: Optional[Sequence[float]] = None,
    xp=np,
    offsets: Optional[np.ndarray] = None,
    max_pairs_per_batch: int = 5_000_000,
) -> RadiusPairs:
    if radius <= 0:
        raise ValueError("radius must be > 0")

    pos = _as_xp_array(pos, xp)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must be shape (N,3)")

    n = int(pos.shape[0])
    if n < 2:
        empty_i = xp.empty(0, dtype=xp.int64)
        empty_f = xp.empty(0, dtype=pos.dtype)
        return RadiusPairs(u=empty_i, v=empty_i, r2=empty_f)

    box = _normalize_boxsize(boxsize)
    if box is not None:
        box_arr = xp.asarray(box, dtype=pos.dtype)
        pos = xp.mod(pos, box_arr)
    else:
        box_arr = None

    if offsets is None:
        offsets = OFFSETS_HALF
    offsets_xp = xp.asarray(offsets, dtype=xp.int64)

    if box_arr is not None:
        dims = xp.ceil(box_arr / float(radius)).astype(xp.int64)
        dims = xp.maximum(dims, 1)
        cell_s = xp.floor(pos / float(radius)).astype(xp.int64)
        cell_s = xp.mod(cell_s, dims)
    else:
        cell = xp.floor(pos / float(radius)).astype(xp.int64)
        cell_min = xp.min(cell, axis=0)
        cell_max = xp.max(cell, axis=0)
        dims = (cell_max - cell_min + 1).astype(xp.int64)
        cell_s = cell - cell_min

    key = (cell_s[:, 0] * dims[1] + cell_s[:, 1]) * dims[2] + cell_s[:, 2]
    order = xp.argsort(key)
    key_sorted = key[order]
    pos_sorted = pos[order]
    cell_sorted = cell_s[order]

    unique_keys, start_idx, counts = xp.unique(
        key_sorted, return_index=True, return_counts=True
    )
    cell_coords = cell_sorted[start_idx]
    n_cells = int(unique_keys.shape[0])

    r2_thresh = float(radius) ** 2
    out_u = []
    out_v = []
    out_r2 = []

    def _min_image(dxyz):
        if box_arr is None:
            return dxyz
        return dxyz - box_arr * xp.rint(dxyz / box_arr)

    for off in offsets_xp:
        if box_arr is not None:
            neigh_coords_in = xp.mod(cell_coords + off, dims)
            inb = None
        else:
            neigh_coords = cell_coords + off
            inb = xp.all(neigh_coords >= 0, axis=1) & xp.all(neigh_coords < dims, axis=1)
            if not bool(xp.any(inb)):
                continue
            neigh_coords_in = neigh_coords[inb]
        neigh_key = (neigh_coords_in[:, 0] * dims[1] + neigh_coords_in[:, 1]) * dims[2] + neigh_coords_in[:, 2]
        idx = xp.searchsorted(unique_keys, neigh_key)
        ok0 = idx < n_cells
        idx_ok = idx[ok0]
        key_ok = neigh_key[ok0]
        hit = unique_keys[idx_ok] == key_ok
        if not bool(xp.any(hit)):
            continue
        src_cells = xp.where(ok0)[0][hit]
        dst_cells = idx_ok[hit]
        if inb is not None:
            src_cells = xp.where(inb)[0][src_cells]

        for sc, dc in zip(src_cells.tolist(), dst_cells.tolist()):
            s0 = int(start_idx[sc])
            s1 = s0 + int(counts[sc])
            d0 = int(start_idx[dc])
            d1 = d0 + int(counts[dc])

            ns = max(1, s1 - s0)
            nd = max(1, d1 - d0)
            total_pairs = ns * nd
            if sc == dc:
                total_pairs = ns * max(0, ns - 1) // 2
                if total_pairs == 0:
                    continue

            batch = max(1, int(max_pairs_per_batch))
            src_idx = xp.arange(s0, s1, dtype=xp.int64)
            dst_idx = xp.arange(d0, d1, dtype=xp.int64)

            if sc == dc:
                if ns * ns <= batch:
                    ii, jj = xp.triu_indices(ns, k=1)
                    us = src_idx[ii]
                    vs = src_idx[jj]
                    dxyz = _min_image(pos_sorted[us] - pos_sorted[vs])
                    r2 = xp.sum(dxyz * dxyz, axis=1)
                    keep = r2 <= r2_thresh
                    if bool(xp.any(keep)):
                        out_u.append(order[us[keep]])
                        out_v.append(order[vs[keep]])
                        out_r2.append(r2[keep])
                else:
                    for i0 in range(0, ns, max(1, batch // max(1, ns))):
                        i1 = min(ns, i0 + max(1, batch // max(1, ns)))
                        local = src_idx[i0:i1]
                        ii, jj = xp.meshgrid(local, src_idx, indexing="ij")
                        mask = ii < jj
                        us = ii[mask]
                        vs = jj[mask]
                        if int(us.size) == 0:
                            continue
                        dxyz = _min_image(pos_sorted[us] - pos_sorted[vs])
                        r2 = xp.sum(dxyz * dxyz, axis=1)
                        keep = r2 <= r2_thresh
                        if bool(xp.any(keep)):
                            out_u.append(order[us[keep]])
                            out_v.append(order[vs[keep]])
                            out_r2.append(r2[keep])
            else:
                rows_per_chunk = max(1, batch // nd)
                for i0 in range(0, ns, rows_per_chunk):
                    i1 = min(ns, i0 + rows_per_chunk)
                    local = src_idx[i0:i1]
                    ii, jj = xp.meshgrid(local, dst_idx, indexing="ij")
                    us = ii.reshape(-1)
                    vs = jj.reshape(-1)
                    dxyz = _min_image(pos_sorted[us] - pos_sorted[vs])
                    r2 = xp.sum(dxyz * dxyz, axis=1)
                    keep = r2 <= r2_thresh
                    if bool(xp.any(keep)):
                        uu = order[us[keep]]
                        vv = order[vs[keep]]
                        lo = xp.minimum(uu, vv)
                        hi = xp.maximum(uu, vv)
                        out_u.append(lo)
                        out_v.append(hi)
                        out_r2.append(r2[keep])

    if not out_u:
        empty_i = xp.empty(0, dtype=xp.int64)
        empty_f = xp.empty(0, dtype=pos.dtype)
        return RadiusPairs(u=empty_i, v=empty_i, r2=empty_f)

    u = xp.concatenate(out_u).astype(xp.int64, copy=False)
    v = xp.concatenate(out_v).astype(xp.int64, copy=False)
    r2 = xp.concatenate(out_r2).astype(pos.dtype, copy=False)

    if box_arr is None:
        pair_key = u * n + v
        keep = xp.ones(u.shape[0], dtype=bool)
        order_unique = xp.argsort(pair_key)
        key_sorted = pair_key[order_unique]
        dup = xp.zeros(key_sorted.shape[0], dtype=bool)
        dup[1:] = key_sorted[1:] == key_sorted[:-1]
        keep_unique = ~dup
        chosen = order_unique[keep_unique]
        u = u[chosen]
        v = v[chosen]
        r2 = r2[chosen]

    return RadiusPairs(u=u, v=v, r2=r2)


def _kernel_table_caesar(fof_ll: float, ntab: int = 1000) -> np.ndarray:
    kerneltab = np.zeros(ntab + 1)
    hinv = 1.0 / fof_ll
    norm = 0.31832 * hinv ** 3
    for i in range(ntab):
        r = 1.0 * i / ntab
        q = 2 * r * hinv
        if q > 2:
            kerneltab[i] = 0.0
        elif q > 1:
            kerneltab[i] = 0.25 * norm * (2 - q) ** 3
        else:
            kerneltab[i] = norm * (1 - 1.5 * q * q * (1 - 0.5 * q))
    return kerneltab


def _kernel_lookup_caesar(r_over_h, ktab, *, xp):
    ntab = int(ktab.shape[0] - 1)
    rtab = ntab * r_over_h + 0.5
    itab = xp.asarray(rtab, dtype=xp.int64)
    itab = xp.clip(itab, 0, ntab)
    return ktab[itab]


def _connected_components_from_edges(u, v, n_nodes: int, *, xp):
    if xp is np:
        from scipy.sparse import coo_matrix  # type: ignore
        from scipy.sparse.csgraph import connected_components  # type: ignore
    else:
        from cupyx.scipy.sparse import coo_matrix  # type: ignore
        from cupyx.scipy.sparse.csgraph import connected_components  # type: ignore

    if int(u.size) == 0:
        labels = xp.arange(n_nodes, dtype=xp.int64)
        return int(n_nodes), labels

    row = xp.concatenate((u, v))
    col = xp.concatenate((v, u))
    data = xp.ones(row.shape[0], dtype=xp.int8)
    graph = coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    n_comp, labels = connected_components(graph, directed=False, return_labels=True)
    return int(n_comp), labels


def _connected_components_from_edges_cugraph(u, v, n_nodes: int, *, xp):
    if xp is np:
        raise ValueError("cugraph connected components requires a CuPy/CUDA backend")
    mods = _try_import_cugraph()
    if mods is None:
        raise RuntimeError("cc_backend='cugraph' requested but cudf/cugraph are not installed")
    cudf, cugraph = mods
    idx_dtype = xp.int32 if n_nodes <= int(np.iinfo(np.int32).max) else xp.int64
    edges = cudf.DataFrame(
        {
            "src": xp.asnumpy(u.astype(idx_dtype, copy=False)),
            "dst": xp.asnumpy(v.astype(idx_dtype, copy=False)),
        }
    )
    graph = cugraph.Graph(directed=False)
    graph.from_cudf_edgelist(edges, source="src", destination="dst", renumber=False)
    labels_df = cugraph.connected_components(graph, connection="weak")
    labels_np = labels_df.sort_values("vertex")["labels"].to_numpy()
    labels = xp.asarray(labels_np, dtype=xp.int64)
    return int(labels_df["labels"].nunique()), labels


def fof6d_on_pool_graph(
    pos: np.ndarray,
    vel: np.ndarray,
    *,
    fof_ll: float,
    vel_ll: Optional[float],
    mingrp: int,
    periodic: bool = False,
    Lbox: Optional[float] = None,
    kernel: Literal["caesar_table", "unity"] = "caesar_table",
    backend: Literal["auto", "numpy", "cupy"] = "auto",
    cc_backend: Literal["auto", "gpu", "cpu", "cugraph"] = "auto",
    return_backend_array: bool = False,
    max_pairs_per_batch: int = 5_000_000,
) -> Tuple[np.ndarray, int]:
    if backend == "cupy":
        cp = _try_import_cupy()
        if cp is None:
            raise RuntimeError("CuPy backend requested but CUDA/CuPy not available")
        xp = cp
    elif backend == "auto":
        cp = _try_import_cupy()
        xp = cp if cp is not None else np
    else:
        xp = np

    requested_cc_backend = str(cc_backend)
    if requested_cc_backend == "auto":
        resolved_cc_backend = "gpu" if xp is not np else "cpu"
    else:
        resolved_cc_backend = requested_cc_backend
    cc_on_cpu = bool(xp is not np and resolved_cc_backend == "cpu")
    if resolved_cc_backend in {"gpu", "cugraph"} and xp is np:
        raise ValueError(f"cc_backend='{resolved_cc_backend}' requires backend to select CuPy")

    pos_xp = _as_xp_array(pos, xp).astype(xp.float64, copy=False)
    vel_xp = _as_xp_array(vel, xp).astype(xp.float64, copy=False)
    if pos_xp.ndim != 2 or int(pos_xp.shape[1]) != 3 or vel_xp.shape != pos_xp.shape:
        raise ValueError("pos and vel must both be shape (N,3)")

    n = int(pos_xp.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64) - 1, 0

    boxsize = None
    if periodic:
        if Lbox is None:
            raise ValueError("periodic=True requires Lbox")
        boxsize = float(Lbox)

    pairs = radius_pairs_grid(
        pos_xp,
        float(fof_ll),
        boxsize=boxsize,
        xp=xp,
        max_pairs_per_batch=int(max_pairs_per_batch),
    )
    idx_dtype = xp.int32 if n <= int(np.iinfo(np.int32).max) else xp.int64
    u = pairs.u.astype(idx_dtype, copy=False)
    v = pairs.v.astype(idx_dtype, copy=False)
    if int(u.size) == 0:
        return np.zeros(n, dtype=np.int64) - 1, 0

    if vel_ll is None:
        u_keep, v_keep = u, v
    else:
        if kernel == "unity":
            w0 = xp.asarray(1.0, dtype=vel_xp.dtype)
        else:
            ktab_np = _kernel_table_caesar(float(fof_ll), ntab=1000)
            ktab = _as_xp_array(ktab_np, xp).astype(vel_xp.dtype, copy=False)
            w0 = ktab[0]

        edge_count = int(u.shape[0])
        edge_batch = max(1, int(max_pairs_per_batch))
        num = xp.zeros(n, dtype=vel_xp.dtype)
        den = xp.full(n, w0, dtype=vel_xp.dtype)
        r2_edges = pairs.r2.astype(vel_xp.dtype, copy=False)

        for e0 in range(0, edge_count, edge_batch):
            e1 = min(edge_count, e0 + edge_batch)
            uc = u[e0:e1]
            vc = v[e0:e1]
            dvel = vel_xp[uc] - vel_xp[vc]
            dv2 = xp.sum(dvel * dvel, axis=1)
            if kernel == "unity":
                w = xp.ones(uc.shape[0], dtype=vel_xp.dtype)
            else:
                r_over_h = xp.sqrt(r2_edges[e0:e1]) / float(fof_ll)
                w = _kernel_lookup_caesar(r_over_h, ktab, xp=xp)
            wdv2 = w * dv2
            num = num + xp.bincount(uc, weights=wdv2, minlength=n).astype(vel_xp.dtype, copy=False)
            num = num + xp.bincount(vc, weights=wdv2, minlength=n).astype(vel_xp.dtype, copy=False)
            den = den + xp.bincount(uc, weights=w, minlength=n).astype(vel_xp.dtype, copy=False)
            den = den + xp.bincount(vc, weights=w, minlength=n).astype(vel_xp.dtype, copy=False)

        sigma = xp.sqrt(num / den)

        kept_u = []
        kept_v = []
        for e0 in range(0, edge_count, edge_batch):
            e1 = min(edge_count, e0 + edge_batch)
            uc = u[e0:e1]
            vc = v[e0:e1]
            dvel = vel_xp[uc] - vel_xp[vc]
            dv2 = xp.sum(dvel * dvel, axis=1)
            sig = xp.maximum(sigma[uc], sigma[vc])
            keep = dv2 <= (float(vel_ll) * sig) ** 2
            if bool(xp.any(keep)):
                if cc_on_cpu:
                    kept_u.append(xp.asnumpy(uc[keep]).astype(np.int64, copy=False))
                    kept_v.append(xp.asnumpy(vc[keep]).astype(np.int64, copy=False))
                else:
                    kept_u.append(uc[keep])
                    kept_v.append(vc[keep])
        if kept_u:
            if cc_on_cpu:
                u_keep = np.concatenate(kept_u).astype(np.int64, copy=False)
                v_keep = np.concatenate(kept_v).astype(np.int64, copy=False)
            else:
                u_keep = xp.concatenate(kept_u).astype(idx_dtype, copy=False)
                v_keep = xp.concatenate(kept_v).astype(idx_dtype, copy=False)
        else:
            if cc_on_cpu:
                u_keep = np.empty(0, dtype=np.int64)
                v_keep = np.empty(0, dtype=np.int64)
            else:
                u_keep = xp.empty(0, dtype=idx_dtype)
                v_keep = xp.empty(0, dtype=idx_dtype)

    if resolved_cc_backend == "cugraph":
        n_comp, labels = _connected_components_from_edges_cugraph(u_keep, v_keep, n, xp=xp)
        cc_xp = xp
    else:
        cc_xp = np if (xp is np or cc_on_cpu) else xp
        try:
            n_comp, labels = _connected_components_from_edges(u_keep, v_keep, n, xp=cc_xp)
        except Exception as exc:
            if (
                requested_cc_backend == "auto"
                and xp is not np
                and cc_xp is xp
                and _is_cuda_oom(exc)
                and _try_import_cugraph() is not None
            ):
                try:
                    xp.cuda.runtime.deviceSynchronize()
                    xp.get_default_memory_pool().free_all_blocks()
                    xp.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass
                n_comp, labels = _connected_components_from_edges_cugraph(u_keep, v_keep, n, xp=xp)
                cc_xp = xp
            else:
                raise

    if mingrp > 1:
        sizes = cc_xp.bincount(labels, minlength=n_comp)
        valid = sizes >= int(mingrp)
        comp_to_gid = cc_xp.full(n_comp, -1, dtype=cc_xp.int64)
        valid_ids = cc_xp.nonzero(valid)[0]
        comp_to_gid[valid_ids] = cc_xp.arange(int(valid_ids.size), dtype=cc_xp.int64)
        tags = comp_to_gid[labels]
        ngroups = int(valid_ids.size)
    else:
        tags = labels.astype(cc_xp.int64, copy=False)
        ngroups = int(n_comp)

    if xp is not np and cc_on_cpu and return_backend_array:
        tags = xp.asarray(tags)
    elif xp is not np and not return_backend_array:
        tags = xp.asnumpy(tags)
    else:
        tags = np.asarray(tags, dtype=np.int64)

    return tags.astype(np.int64, copy=False), int(ngroups)
