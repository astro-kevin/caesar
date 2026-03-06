from pathlib import Path
import importlib.util
import sys

import numpy as np


module_path = Path(__file__).resolve().parents[1] / "caesar" / "ahf_fast_match.py"
spec = importlib.util.spec_from_file_location("ahf_fast_match", module_path)
ahf_fast_match = importlib.util.module_from_spec(spec)
sys.modules["ahf_fast_match"] = ahf_fast_match
spec.loader.exec_module(ahf_fast_match)

HostSchedState = ahf_fast_match.HostSchedState


_GIB = int(2**30)


def _mk_hosts(predicted_bytes, fof_candidates=None, tiny_idx=None):
    tiny_idx = set(tiny_idx or [])
    if fof_candidates is None:
        fof_candidates = [max(1, int(v // (256 * 1024**2))) for v in predicted_bytes]
    hosts = []
    for i, pred in enumerate(predicted_bytes):
        hosts.append(
            HostSchedState(
                order_idx=i,
                root_id=i,
                nodes=set(),
                fof_candidates=int(fof_candidates[i]),
                star_count=0,
                predicted_bytes=int(pred),
                is_tiny_proxy=(i in tiny_idx),
            )
        )
    return hosts


def test_refit_cadence_is_powers_of_two_starting_at_8():
    target = int(2**3)
    seq = [target]
    for _ in range(4):
        target = ahf_fast_match._ahf_fast_next_refit_target(current_target=target, exp_max=16)
        seq.append(target)
    assert seq == [8, 16, 32, 64, 128]


def test_monotone_model_prediction_is_non_decreasing_with_size():
    x = np.log1p(np.asarray([16, 64, 256, 1024, 4096], dtype=np.float64))
    y = np.asarray([0.5, 1.0, 2.5, 5.0, 10.0], dtype=np.float64) * _GIB
    w = np.ones_like(y, dtype=np.float64)
    model = ahf_fast_match._ahf_fast_fit_mem_model(
        sample_x=x.tolist(),
        sample_y=y.tolist(),
        sample_w=w.tolist(),
        tau=0.90,
        knots=5,
        neighbors=5,
    )
    assert model is not None
    preds = [
        ahf_fast_match._ahf_fast_predict_reservation_bytes(
            fof_candidates=v,
            model=model,
            seed_bytes=1 * _GIB,
            pred_min_bytes=int(0.25 * _GIB),
            pred_max_bytes=int(32 * _GIB),
        )
        for v in [8, 32, 128, 512, 2048, 8192]
    ]
    assert all(a <= b for a, b in zip(preds, preds[1:]))


def test_seed_anchor_samples_build_monotone_initial_spline():
    anchor_x, anchor_y, anchor_w = ahf_fast_match._ahf_fast_seed_anchor_samples(
        fof_candidates=[64, 256, 1024, 4096, 16384, 65536],
        work_unit=128,
        base_unit_bytes=2 * _GIB,
        pred_min_bytes=int(0.25 * _GIB),
        pred_max_bytes=int(32 * _GIB),
        anchor_count=6,
        anchor_weight=0.25,
    )
    assert len(anchor_x) >= 2
    assert len(anchor_x) == len(anchor_y) == len(anchor_w)
    assert all(a < b for a, b in zip(anchor_x, anchor_x[1:]))
    assert all(a <= b for a, b in zip(anchor_y, anchor_y[1:]))

    model = ahf_fast_match._ahf_fast_fit_mem_model(
        sample_x=anchor_x,
        sample_y=anchor_y,
        sample_w=anchor_w,
        tau=0.90,
        knots=6,
        neighbors=6,
    )
    assert model is not None

    preds = [
        ahf_fast_match._ahf_fast_predict_reservation_bytes(
            fof_candidates=v,
            model=model,
            seed_bytes=1 * _GIB,
            pred_min_bytes=int(0.25 * _GIB),
            pred_max_bytes=int(32 * _GIB),
        )
        for v in [64, 512, 4096, 32768]
    ]
    assert all(a <= b for a, b in zip(preds, preds[1:]))


def test_real_samples_can_replace_seed_anchors_after_threshold():
    anchor_x, anchor_y, anchor_w = ahf_fast_match._ahf_fast_seed_anchor_samples(
        fof_candidates=[64, 256, 1024, 4096, 16384, 65536],
        work_unit=128,
        base_unit_bytes=2 * _GIB,
        pred_min_bytes=int(0.25 * _GIB),
        pred_max_bytes=int(32 * _GIB),
        anchor_count=6,
        anchor_weight=0.25,
    )
    real_x = np.log1p(np.asarray([128, 512, 2048, 8192, 32768, 131072], dtype=np.float64))
    real_y = np.asarray([1.0, 1.4, 2.1, 3.8, 6.2, 11.0], dtype=np.float64) * _GIB
    real_w = np.ones_like(real_y, dtype=np.float64)
    fit_x_small, fit_y_small, fit_w_small = ahf_fast_match._ahf_fast_model_fit_samples(
        seed_anchor_x=anchor_x,
        seed_anchor_y=anchor_y,
        seed_anchor_w=anchor_w,
        sample_x=real_x[:3].tolist(),
        sample_y=real_y[:3].tolist(),
        sample_w=real_w[:3].tolist(),
        seed_drop_samples=8,
    )
    assert len(fit_x_small) == len(anchor_x) + 3
    assert fit_x_small[: len(anchor_x)] == anchor_x

    fit_x_full, fit_y_full, fit_w_full = ahf_fast_match._ahf_fast_model_fit_samples(
        seed_anchor_x=anchor_x,
        seed_anchor_y=anchor_y,
        seed_anchor_w=anchor_w,
        sample_x=real_x.tolist(),
        sample_y=real_y.tolist(),
        sample_w=real_w.tolist(),
        seed_drop_samples=4,
    )
    assert fit_x_full == real_x.tolist()
    assert fit_y_full == real_y.tolist()
    assert fit_w_full == real_w.tolist()


def test_phase_a_can_fill_all_worker_slots_when_capacity_permits():
    jobs = 6
    capacity = 1000
    pending = _mk_hosts([20, 25, 30, 15, 10, 18, 12, 9], tiny_idx=[])
    launched = 0
    reserved = 0
    while pending and launched < jobs:
        idx = ahf_fast_match._ahf_fast_select_best_fit_index(
            pending_hosts=pending,
            scan_window=len(pending),
            remaining_bytes=int(capacity - reserved),
        )
        if idx < 0:
            break
        host = pending.pop(idx)
        reserved += int(host.predicted_bytes)
        launched += 1
    assert launched == jobs


def test_phase_a_global_best_fit_not_front_window_limited():
    pending = _mk_hosts([120, 110, 95, 30, 20], tiny_idx=[])
    idx_front = ahf_fast_match._ahf_fast_select_best_fit_index(
        pending_hosts=pending,
        scan_window=3,
        remaining_bytes=35,
    )
    idx_full = ahf_fast_match._ahf_fast_select_best_fit_index(
        pending_hosts=pending,
        scan_window=len(pending),
        remaining_bytes=35,
    )
    assert idx_front == -1
    assert idx_full >= 0
    assert int(pending[idx_full].predicted_bytes) == 30


def test_phase_a_choice_is_invariant_to_host_order_under_global_scan():
    predicted = [70, 55, 42, 35, 28]
    remaining = 50
    base = _mk_hosts(predicted, tiny_idx=[])
    idx = ahf_fast_match._ahf_fast_select_best_fit_index(
        pending_hosts=base,
        scan_window=len(base),
        remaining_bytes=remaining,
    )
    assert idx >= 0
    expected = int(base[idx].predicted_bytes)

    perm = _mk_hosts([35, 70, 28, 42, 55], tiny_idx=[])
    idx_perm = ahf_fast_match._ahf_fast_select_best_fit_index(
        pending_hosts=perm,
        scan_window=len(perm),
        remaining_bytes=remaining,
    )
    assert idx_perm >= 0
    assert int(perm[idx_perm].predicted_bytes) == expected


def test_phase_b_backfills_with_tiny_when_phase_a_cannot_fit():
    pending = _mk_hosts(
        predicted_bytes=[60, 55, 4, 3],
        fof_candidates=[6000, 5500, 20, 16],
        tiny_idx=[2, 3],
    )
    phase_a = ahf_fast_match._ahf_fast_select_best_fit_index(
        pending_hosts=pending,
        scan_window=len(pending),
        remaining_bytes=10,
    )
    assert phase_a == -1

    jobs = 5
    inflight = 2
    free_slots = jobs - inflight
    launched_tiny = 0
    while launched_tiny < free_slots:
        phase_b = ahf_fast_match._ahf_fast_select_tiny_backfill_index(
            pending_hosts=pending,
            scan_window=len(pending),
        )
        if phase_b < 0:
            break
        assert pending[phase_b].is_tiny_proxy
        pending.pop(phase_b)
        launched_tiny += 1

    assert launched_tiny == 2


def test_observed_bytes_and_sample_weight_helpers():
    obs1 = ahf_fast_match._ahf_fast_observed_bytes_estimate(
        submit_inflight=1,
        rss_submit_bytes=10 * _GIB,
        rss_done_bytes=11 * _GIB,
    )
    obs2 = ahf_fast_match._ahf_fast_observed_bytes_estimate(
        submit_inflight=2,
        rss_submit_bytes=10 * _GIB,
        rss_done_bytes=11 * _GIB,
    )
    obs3 = ahf_fast_match._ahf_fast_observed_bytes_estimate(
        submit_inflight=3,
        rss_submit_bytes=10 * _GIB,
        rss_done_bytes=11 * _GIB,
    )
    w1 = ahf_fast_match._ahf_fast_sample_weight(
        submit_inflight=1,
        anchor_max_inflight=2,
        sample_max_inflight=8,
    )
    w2 = ahf_fast_match._ahf_fast_sample_weight(
        submit_inflight=2,
        anchor_max_inflight=2,
        sample_max_inflight=8,
    )
    w3 = ahf_fast_match._ahf_fast_sample_weight(
        submit_inflight=3,
        anchor_max_inflight=2,
        sample_max_inflight=8,
    )
    w8 = ahf_fast_match._ahf_fast_sample_weight(
        submit_inflight=8,
        anchor_max_inflight=2,
        sample_max_inflight=8,
    )
    w9 = ahf_fast_match._ahf_fast_sample_weight(
        submit_inflight=9,
        anchor_max_inflight=2,
        sample_max_inflight=8,
    )

    assert obs1 == 1 * _GIB
    assert obs2 == int(0.5 * _GIB)
    assert obs3 == int(round((1 * _GIB) / 3.0))
    assert w1 == 1.0
    assert w2 == 0.5
    assert abs(w3 - (1.0 / 3.0)) < 1.0e-12
    assert w8 == 0.125
    assert w9 == 0.0
