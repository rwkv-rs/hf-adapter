from __future__ import annotations

import json

from evaluation.run_lm_eval_matrix import find_result_json, task_provenance


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
