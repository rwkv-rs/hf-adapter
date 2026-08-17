from __future__ import annotations

import json
import re
from statistics import median
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = {
    "0.4B / 0.8B": 0,
    "1.5B / 2B": 1,
    "2.9B / 4B": 2,
    "7.2B / 9B": 3,
}
GPU_ORDER = {
    "V100 32GB": 0,
    "RTX 3090": 1,
    "RTX 4080": 2,
    "RTX 4090": 3,
    "RTX 5070 Laptop": 4,
    "RTX 5090": 5,
}
PAIR_LABELS = {
    "rwkv-0.4b__qwen3.5-0.8b": "0.4B / 0.8B",
    "rwkv-1.5b__qwen3.5-2b": "1.5B / 2B",
    "rwkv-2.9b__qwen3.5-4b": "2.9B / 4B",
    "rwkv-7.2b__qwen3.5-9b": "7.2B / 9B",
}


def assert_throughput_format(cell: str) -> None:
    values = cell.replace("**", "").split("/")
    assert len(values) == 2
    for value in values:
        rendered = value.strip()
        numeric = float(rendered.replace(",", ""))
        if numeric >= 100:
            assert "." not in rendered, rendered
        else:
            assert re.fullmatch(r"\d{1,2}\.\d", rendered), rendered


def latest_tokps_rows(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines.index(
        "| 模型对（RWKV / Qwen） | 显卡 | Batch | RWKV P / D tok/s | Qwen P / D tok/s | 参数调整 P / D | 严格门 | 证据 |"
    )
    rows: list[list[str]] = []
    for line in lines[header + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def read_jsonl(relative: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (ROOT / relative).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def display_rate(value: float) -> str:
    return f"{value:,.0f}" if value >= 100 else f"{value:.1f}"


def displayed_pd(prefill: float, decode: float) -> str:
    return f"**{display_rate(prefill)} / {display_rate(decode)}**"


def displayed_ratio_pd(prefill: float, decode: float) -> str:
    return f"{prefill:.3f}x / {decode:.3f}x"


def test_latest_tokps_doc_is_model_gpu_batch_sorted_and_scoped() -> None:
    path = ROOT / "docs/QWEN35_LATEST_P_D_TOKPS.md"
    rows = latest_tokps_rows(path)
    assert len(rows) == 38

    keys = [
        (MODEL_ORDER[row[0]], GPU_ORDER[row[1]], int(row[2].removeprefix("B")))
        for row in rows
    ]
    assert keys == sorted(keys)
    assert keys == sorted(set(keys))

    for row in rows:
        assert_throughput_format(row[3])
        assert_throughput_format(row[4])
        if row[1] == "RTX 5090":
            assert row[6] == "P+D 6/6 PASS*"
        else:
            assert row[6] == "P+D 6/6 PASS"

    text = path.read_text(encoding="utf-8")
    assert "原始 Prefill 48/48、参数调整 Prefill 48/48" in text
    assert "1.089713x/1.354606x/4.590900x" in text
    assert "RTX 4080 受 16 GiB 容量限制" in text


def test_latest_tokps_doc_matches_promoted_artifact_medians() -> None:
    expected: dict[tuple[str, str, int], tuple[str, str, str]] = {}
    paired_artifacts = {
        "V100 32GB": ("bench/v100_qwen35_paired_pd_v1_20260814/paired_pd_table.jsonl"),
        "RTX 3090": ("bench/3090_qwen35_paired_pd_v2_20260816/paired_pd_table.jsonl"),
        "RTX 4080": ("bench/4080_qwen35_paired_pd_v1_20260814/paired_pd_table.jsonl"),
        "RTX 4090": ("bench/4090_qwen35_paired_pd_v2_20260815/paired_pd_table.jsonl"),
    }
    for gpu, relative in paired_artifacts.items():
        artifact_rows = read_jsonl(relative)
        for model_pair, label in PAIR_LABELS.items():
            for batch_size in (1, 8):
                group = [
                    row
                    for row in artifact_rows
                    if row["model_pair"] == model_pair
                    and row["batch_size"] == batch_size
                ]
                if not group:
                    continue
                assert len(group) == 6
                expected[(label, gpu, batch_size)] = (
                    displayed_pd(
                        median(
                            row["candidate_prefill_tokps_total_raw"] for row in group
                        ),
                        median(
                            row["candidate_decode_tokps_total_raw"] for row in group
                        ),
                    ),
                    displayed_pd(
                        median(
                            row["reference_prefill_tokps_total_raw"] for row in group
                        ),
                        median(
                            row["reference_decode_tokps_total_raw"] for row in group
                        ),
                    ),
                    displayed_ratio_pd(
                        median(row["adjusted_prefill_ratio"] for row in group),
                        median(row["adjusted_decode_ratio"] for row in group),
                    ),
                )

    candidate_rows = read_jsonl(
        "bench/5090_qwen35_paired_decode_v1_20260813/rwkv_candidate.jsonl"
    )
    reference_rows = read_jsonl(
        "bench/5090_qwen35_best_optimized_hf_v1_20260813/qwen_reference.jsonl"
    )
    all_5090_adjusted_prefill: list[float] = []
    for model_pair, label in PAIR_LABELS.items():
        for batch_size in (1, 8):
            candidate_group = [
                row
                for row in candidate_rows
                if row["model_pair"] == model_pair and row["batch_size"] == batch_size
            ]
            reference_group = [
                row
                for row in reference_rows
                if row["model_pair"] == model_pair and row["batch_size"] == batch_size
            ]
            assert len(candidate_group) == len(reference_group) == 6
            candidate_by_shape = {
                (row["prompt_tokens"], row["decode_tokens"]): row
                for row in candidate_group
            }
            reference_by_shape = {
                (row["prompt_tokens"], row["decode_tokens"]): row
                for row in reference_group
            }
            assert candidate_by_shape.keys() == reference_by_shape.keys()
            adjusted_prefill: list[float] = []
            adjusted_decode: list[float] = []
            for shape, candidate in candidate_by_shape.items():
                reference = reference_by_shape[shape]
                active_ratio = (
                    candidate["active_parameter_count"]
                    / reference["active_parameter_count"]
                )
                adjusted_prefill.append(
                    candidate["prefill_tokps_total_raw"]
                    / reference["prefill_tokps_total_raw"]
                    * active_ratio
                )
                adjusted_decode.append(
                    candidate["decode_tokps_total_raw"]
                    / reference["decode_tokps_total_raw"]
                    * active_ratio
                )
            assert all(value > 1.0 for value in adjusted_prefill)
            assert all(value > 1.0 for value in adjusted_decode)
            all_5090_adjusted_prefill.extend(adjusted_prefill)
            expected[(label, "RTX 5090", batch_size)] = (
                displayed_pd(
                    median(row["prefill_tokps_total_raw"] for row in candidate_group),
                    median(row["decode_tokps_total_raw"] for row in candidate_group),
                ),
                displayed_pd(
                    median(row["prefill_tokps_total_raw"] for row in reference_group),
                    median(row["decode_tokps_total_raw"] for row in reference_group),
                ),
                displayed_ratio_pd(median(adjusted_prefill), median(adjusted_decode)),
            )

    assert len(all_5090_adjusted_prefill) == 48
    assert (
        f"{min(all_5090_adjusted_prefill):.6f}x/"
        f"{median(all_5090_adjusted_prefill):.6f}x/"
        f"{max(all_5090_adjusted_prefill):.6f}x" == "1.089713x/1.354606x/4.590900x"
    )

    document_rows = latest_tokps_rows(ROOT / "docs/QWEN35_LATEST_P_D_TOKPS.md")
    actual = {
        (row[0], row[1], int(row[2].removeprefix("B"))): (row[3], row[4], row[5])
        for row in document_rows
    }
    assert actual == expected
