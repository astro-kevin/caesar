from pathlib import Path
import importlib.util
import sys


module_path = Path(__file__).resolve().parents[1] / "caesar" / "ahf_fast_match.py"
spec = importlib.util.spec_from_file_location("ahf_fast_match", module_path)
ahf_fast_match = importlib.util.module_from_spec(spec)
sys.modules["ahf_fast_match"] = ahf_fast_match
spec.loader.exec_module(ahf_fast_match)

HostSchedState = ahf_fast_match.HostSchedState


_GIB = int(2**30)


def _mk_hosts(predicted_bytes):
    hosts = []
    for i, pred in enumerate(predicted_bytes):
        hosts.append(
            HostSchedState(
                order_idx=i,
                root_id=i,
                nodes=set(),
                fof_candidates=0,
                star_count=0,
                predicted_bytes=int(pred),
            )
        )
    return hosts


def _pack_best_fit(predictions, capacity_bytes, scan_window=256):
    pending = _mk_hosts(predictions)
    used = 0
    chosen = []
    while pending:
        idx = ahf_fast_match._ahf_fast_select_best_fit_index(
            pending_hosts=pending,
            scan_window=scan_window,
            remaining_bytes=max(0, int(capacity_bytes) - int(used)),
        )
        if idx < 0:
            break
        host = pending.pop(idx)
        used += int(host.predicted_bytes)
        chosen.append(int(host.predicted_bytes))
    return used, chosen


def _pack_fifo(predictions, capacity_bytes):
    used = 0
    for pred in predictions:
        pred_i = int(pred)
        if used + pred_i > int(capacity_bytes):
            break
        used += pred_i
    return used


def test_best_fit_admission_packs_better_than_fifo():
    preds = [120, 60, 40, 20]
    best_fit_used, _ = _pack_best_fit(preds, 100)
    fifo_used = _pack_fifo(preds, 100)
    assert best_fit_used == 100
    assert best_fit_used > fifo_used


def test_no_large_cap_when_capacity_allows_multiple_large_hosts():
    preds = [45, 44, 12]
    used, chosen = _pack_best_fit(preds, 89)
    assert used == 89
    assert sum(1 for v in chosen if v >= 40) == 2


def test_borrow_mode_gate_requires_high_mode_and_non_negative_slope():
    assert ahf_fast_match._ahf_fast_can_borrow(
        mode="high", slope_gbps=0.0, inflight_count=1, target_workers=4
    )
    assert not ahf_fast_match._ahf_fast_can_borrow(
        mode="nominal", slope_gbps=0.0, inflight_count=1, target_workers=4
    )
    assert not ahf_fast_match._ahf_fast_can_borrow(
        mode="high", slope_gbps=-0.1, inflight_count=1, target_workers=4
    )
    assert not ahf_fast_match._ahf_fast_can_borrow(
        mode="high", slope_gbps=0.1, inflight_count=4, target_workers=4
    )


def test_anchor_update_policy_is_opportunistic_at_low_contention():
    should_update, is_anchor, alpha = ahf_fast_match._ahf_fast_update_policy_for_sample(
        submit_inflight=2,
        anchor_max_inflight=2,
        anchor_found=False,
        alpha_anchor=0.35,
        alpha_online=0.08,
    )
    assert should_update
    assert is_anchor
    assert alpha == 0.35

    should_update, is_anchor, alpha = ahf_fast_match._ahf_fast_update_policy_for_sample(
        submit_inflight=2,
        anchor_max_inflight=2,
        anchor_found=True,
        alpha_anchor=0.35,
        alpha_online=0.08,
    )
    assert should_update
    assert not is_anchor
    assert alpha == 0.08

    should_update, is_anchor, alpha = ahf_fast_match._ahf_fast_update_policy_for_sample(
        submit_inflight=3,
        anchor_max_inflight=2,
        anchor_found=False,
        alpha_anchor=0.35,
        alpha_online=0.08,
    )
    assert not should_update
    assert not is_anchor
    assert alpha == 0.08


def test_online_multiplier_update_clamps_ratio_and_moves_directionally():
    up, up_ratio = ahf_fast_match._ahf_fast_update_bin_multiplier(
        current=1.0,
        observed_bytes=400,
        predicted_bytes=100,
        alpha=0.5,
    )
    down, down_ratio = ahf_fast_match._ahf_fast_update_bin_multiplier(
        current=1.0,
        observed_bytes=10,
        predicted_bytes=100,
        alpha=0.5,
    )

    assert up_ratio == 2.0
    assert down_ratio == 0.5
    assert up > 1.0
    assert down < 1.0


def test_observed_bytes_and_smallest_fallback_helpers():
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

    assert obs1 == 1 * _GIB
    assert obs2 == int(0.5 * _GIB)
    assert obs3 is None

    pending = _mk_hosts([7 * _GIB, 2 * _GIB, 5 * _GIB])
    idx = ahf_fast_match._ahf_fast_select_smallest_index(pending_hosts=pending, scan_window=3)
    assert idx == 1
