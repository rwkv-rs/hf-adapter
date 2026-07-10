from __future__ import annotations

import json

from evaluation.run_lm_eval_matrix import (
    find_result_json,
    resolve_model,
    task_provenance,
)
from evaluation.validate_lm_eval_three_way import (
    compare_sample_outcomes,
    sample_outcomes,
)


def test_kernel_route_trace_is_not_mistaken_for_lm_eval_result(tmp_path):
    unit = tmp_path / "unit"
    nested = unit / "model"
    nested.mkdir(parents=True)
    result = nested / "results_2026-08-27T00-00-00.json"
    result.write_text(
        json.dumps({"configs": {"piqa": {}}, "samples": {"piqa": []}}),
        encoding="utf-8",
    )
    route = unit / "kernel-route.json"
    route.write_text(
        json.dumps({"actual_recurrent_calls": {"graph": 1}}), encoding="utf-8"
    )
    route.touch()

    assert find_result_json(unit) == result
    assert task_provenance(unit, "piqa")["result_json"] == str(result)


def test_local_model_provenance_hashes_rwkv_vocabulary(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "rwkv_vocab_v20230424.txt").write_text("vocabulary", encoding="utf-8")
    provenance = resolve_model(str(model), None)
    assert "rwkv_vocab_v20230424.txt" in provenance["files"]


def _write_samples(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_sample_outcomes_compare_selected_choices_not_only_accuracy(tmp_path):
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    base = {
        "doc_hash": "same-doc",
        "metrics": ["acc", "acc_norm"],
        "acc": 0.0,
        "acc_norm": 0.0,
        "arguments": {
            "gen_args_0": {"arg_1": " first"},
            "gen_args_1": {"arg_1": " much longer second"},
            "gen_args_2": {"arg_1": " third"},
        },
    }
    _write_samples(
        left,
        [
            {
                **base,
                "filtered_resps": [
                    ["-2", "False"],
                    ["-3", "False"],
                    ["-4", "False"],
                ],
            }
        ],
    )
    _write_samples(
        right,
        [
            {
                **base,
                "filtered_resps": [
                    ["-4", "False"],
                    ["-3", "False"],
                    ["-2", "False"],
                ],
            }
        ],
    )
    left_rows = sample_outcomes(left, "piqa")
    right_rows = sample_outcomes(right, "piqa")
    assert left_rows["same-doc"][1] == 0
    assert right_rows["same-doc"][1] == 2
    comparison = compare_sample_outcomes(left_rows, right_rows, "piqa")
    assert comparison["prediction_mismatches"] == 1


def test_wikitext_sample_nll_gate_is_point_one_percent(tmp_path):
    left = tmp_path / "left.jsonl"
    near = tmp_path / "near.jsonl"
    far = tmp_path / "far.jsonl"
    base = {
        "doc_hash": "wiki-doc",
        "word_perplexity": [-100.0, 20],
        "byte_perplexity": [-100.0, 80],
    }
    _write_samples(left, [{**base, "filtered_resps": ["-100.0"]}])
    _write_samples(near, [{**base, "filtered_resps": ["-100.05"]}])
    _write_samples(far, [{**base, "filtered_resps": ["-100.2"]}])
    reference = sample_outcomes(left, "wikitext")
    assert compare_sample_outcomes(
        reference, sample_outcomes(near, "wikitext"), "wikitext"
    )["continuous_mismatches"] == 0
    assert compare_sample_outcomes(
        reference, sample_outcomes(far, "wikitext"), "wikitext"
    )["continuous_mismatches"] == 1
