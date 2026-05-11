from __future__ import annotations

from bench.select_qwen35_graph_route_v1 import INDUCTOR, RAW, select_route
from tests.test_qwen35_best_optimized_hf_v1 import row, use_raw_graph


PAIR = "rwkv-0.4b__qwen3.5-0.8b"


def probe(route: str, batch: int, prompt: int, decode: int, rate: float) -> dict:
    item = row(PAIR, batch, prompt, decode)
    if route == RAW:
        use_raw_graph(item)
    item.update(
        {
            "device": "NVIDIA GeForce RTX 4080",
            "model_id_or_path": "/models/qwen-0.8b",
            "decode_tokps_total_raw": rate,
            "decode_sec_median_raw": batch * decode / rate,
            "decode_sec_median": batch * decode / rate,
            "decode_sec_samples": [batch * decode / rate] * 7,
        }
    )
    return item


def rows(
    route: str, short_rates: tuple[float, float], *, boundary: bool = False
) -> list[dict]:
    shapes = (
        ((1, 2048, 512), (8, 2048, 512))
        if boundary
        else (
            (1, 128, 128),
            (8, 128, 128),
        )
    )
    return [
        probe(route, batch, prompt, decode, rate)
        for (batch, prompt, decode), rate in zip(shapes, short_rates, strict=True)
    ]


def test_unanimous_short_margin_selects_one_whole_model_route() -> None:
    summary = select_route(
        rows(INDUCTOR, (120.0, 1200.0)),
        rows(RAW, (100.0, 1000.0)),
        expected_device="NVIDIA GeForce RTX 4080",
    )
    assert summary["status"] == "pass"
    assert summary["selected_route"] == INDUCTOR
    assert summary["selection_policy"]["per_cell_route_mixing"] is False


def test_close_or_split_short_probe_requires_boundaries() -> None:
    summary = select_route(
        rows(INDUCTOR, (103.0, 980.0)),
        rows(RAW, (100.0, 1000.0)),
        expected_device="NVIDIA GeForce RTX 4080",
    )
    assert summary["status"] == "needs_boundary"
    assert summary["selected_route"] is None


def test_four_cell_geometric_mean_selects_route() -> None:
    summary = select_route(
        rows(INDUCTOR, (103.0, 1030.0)),
        rows(RAW, (100.0, 1000.0)),
        expected_device="NVIDIA GeForce RTX 4080",
        inductor_boundary=rows(INDUCTOR, (104.0, 1040.0), boundary=True),
        raw_boundary=rows(RAW, (100.0, 1000.0), boundary=True),
    )
    assert summary["status"] == "pass"
    assert summary["selected_route"] == INDUCTOR


def test_route_or_runtime_drift_fails_closed() -> None:
    raw = rows(RAW, (100.0, 1000.0))
    raw[0]["torch_version"] = "different"
    summary = select_route(
        rows(INDUCTOR, (120.0, 1200.0)),
        raw,
        expected_device="NVIDIA GeForce RTX 4080",
    )
    assert summary["status"] == "fail"
    assert any("runtime signature" in error for error in summary["errors"])


def test_correctness_failure_disqualifies_only_that_route() -> None:
    inductor = rows(INDUCTOR, (120.0, 1200.0))
    inductor[0]["status"] = "fail"
    inductor[0]["qwen_graph_parity_verified"] = False
    summary = select_route(
        inductor,
        rows(RAW, (100.0, 1000.0)),
        expected_device="NVIDIA GeForce RTX 4080",
    )
    assert summary["status"] == "pass"
    assert summary["selected_route"] == RAW
    assert summary["decision"] == "selected_alternate_after_correctness_failure"
    assert summary["route_errors"][INDUCTOR]


def test_explicit_informational_cross_cache_policy_is_preserved() -> None:
    inductor = rows(INDUCTOR, (120.0, 1200.0))
    raw = rows(RAW, (100.0, 1000.0))
    for item in [*inductor, *raw]:
        item["qwen_cross_cache_full_greedy_policy_requested"] = "informational"
        item["qwen_cross_cache_full_greedy_policy_effective"] = "informational"
        item["qwen_cross_cache_full_greedy_required"] = False
    raw[0]["qwen_graph_greedy_match"] = False
    raw[0]["qwen_static_cache_eager_greedy_match"] = False
    raw[0]["qwen_dynamic_static_full_greedy_mismatch_count"] = 1
    raw[0]["qwen_dynamic_static_full_greedy_first_mismatch_index"] = 100
    raw[0]["qwen_dynamic_candidate_full_greedy_mismatch_count"] = 1
    raw[0]["qwen_dynamic_candidate_full_greedy_first_mismatch_index"] = 100
    summary = select_route(
        inductor,
        raw,
        expected_device="NVIDIA GeForce RTX 4080",
        cross_cache_full_greedy_policy="informational",
    )
    assert summary["status"] == "pass"
    assert summary["selection_policy"]["cross_cache_full_greedy_policy"] == (
        "informational"
    )
