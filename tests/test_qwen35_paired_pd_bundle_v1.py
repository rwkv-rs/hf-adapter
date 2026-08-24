from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

from bench.analyzers.compare_rwkv_prefill_probe import compare
from bench.validators.validate_qwen35_paired_pd_bundle_v1 import main, validate_bundle
from tests.test_qwen35_paired_pd_v1 import COMMIT, PAIRS, candidates, references


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    return {"path": f"/formal/{path.name}", "sha256": _sha(path)}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _hash_manifest(path: Path, models: list[str]) -> None:
    lines: list[str] = []
    for model in models:
        lines.extend(
            [
                f"[{model}]",
                f"{'1' * 64}  config.json",
                f"{'2' * 64}  model.safetensors",
                f"{'3' * 64}  tokenizer.json",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _probe(path: Path, pair: str, size: str, model: str, batch: int) -> dict:
    input_ids = torch.arange(2048).repeat(batch, 1)
    if batch == 8:
        input_ids = input_ids + torch.arange(batch).unsqueeze(1)
    value = {
        "probe_schema_version": 2,
        "benchmark_repository_commit": COMMIT,
        "model_pair": pair,
        "model_size_label": size,
        "model_id_or_path": model,
        "probe_output": f"/formal/{path.name}",
        "input_ids": input_ids,
        "greedy_tokens": torch.arange(512).repeat(batch, 1).T.squeeze(-1),
        "prompt_logits": torch.ones(batch, 16),
        "final_logits": torch.ones(batch, 16) * 2,
        "decode_logits_all_finite": True,
        "decode_logits_finite_by_batch": torch.ones(batch, dtype=torch.bool),
    }
    torch.save(value, path)
    return value


def _bundle(tmp_path: Path) -> dict[str, object]:
    candidate_rows = candidates()
    reference_rows = references()
    models = {
        PAIRS[0]: ("0.4b", "/models/rwkv-0p4", "/models/qwen-0p8"),
        PAIRS[1]: ("1.5b", "/models/rwkv-1p5", "/models/qwen-2b"),
        PAIRS[2]: ("2.9b", "/models/rwkv-2p9", "/models/qwen-4b"),
    }
    for row in candidate_rows:
        row["model_id_or_path"] = models[row["model_pair"]][1]
    for row in reference_rows:
        row["model_id_or_path"] = models[row["model_pair"]][2]

    correctness_entries = []
    lanes = []
    for tag, pair in zip(("0p4", "1p5", "2p9"), PAIRS, strict=True):
        size, rwkv_model, _qwen_model = models[pair]
        for batch in (1, 8):
            lane_rows = [
                row
                for row in candidate_rows
                if row["model_pair"] == pair and row["batch_size"] == batch
            ]
            main = next(
                row
                for row in lane_rows
                if row["prompt_tokens"] == 2048 and row["decode_tokens"] == 512
            )
            native_probe_path = (
                tmp_path / f"decode_correctness_{tag}_b{batch}_native.pt"
            )
            fla_probe_path = tmp_path / f"decode_correctness_{tag}_b{batch}_fla.pt"
            native_probe = _probe(native_probe_path, pair, size, rwkv_model, batch)
            fla_probe = _probe(fla_probe_path, pair, size, rwkv_model, batch)
            main.update(
                {
                    "probe_output": f"/formal/{native_probe_path.name}",
                    "probe_tokens": 512,
                    "probe_batch_size": batch,
                    "probe_distinct_batch_prompts": batch == 8,
                    "probe_decode_logits_all_finite": True,
                    "probe_decode_logits_finite_by_batch": [True] * batch,
                    "probe_greedy_tokens": native_probe["greedy_tokens"].tolist(),
                }
            )
            lane_path = tmp_path / f"rwkv_{tag}_b{batch}.jsonl"
            _write_jsonl(lane_path, lane_rows)
            native_row_path = (
                tmp_path / f"decode_correctness_{tag}_b{batch}_native.jsonl"
            )
            _write_jsonl(native_row_path, [main])
            fla_row = dict(main)
            fla_row.update(
                {
                    "benchmark_matrix": "rwkv_native_graph_fla_correctness_4080_v1",
                    "optimization_lane": "fla_reference",
                    "warmup": 1,
                    "runs": 1,
                    "rwkv_implementation_requested": "wrapper_repo",
                    "rwkv_implementation_effective": "wrapper_repo",
                    "effective_backend": "fla",
                    "cache_type": "RWKV7StateCache",
                    "probe_output": f"/formal/{fla_probe_path.name}",
                }
            )
            fla_row_path = tmp_path / f"decode_correctness_{tag}_b{batch}_fla.jsonl"
            _write_jsonl(fla_row_path, [fla_row])
            comparison = compare(fla_probe, native_probe, 0.9999)
            comparison["contract_errors"] = []
            comparison_path = (
                tmp_path / f"decode_correctness_{tag}_b{batch}_compare.json"
            )
            _write_json(comparison_path, comparison)
            correctness_entries.append(
                {
                    "model_pair": pair,
                    "model_size_label": size,
                    "model_path": rwkv_model,
                    "batch_size": batch,
                    "prompt_tokens": 2048,
                    "decode_tokens": 512,
                    "probe_tokens": 512,
                    "fla_reference": {
                        "row": _artifact(fla_row_path),
                        "probe": _artifact(fla_probe_path),
                    },
                    "native_candidate": {
                        "row": _artifact(native_row_path),
                        "probe": _artifact(native_probe_path),
                        "source_lane": _artifact(lane_path),
                        "source_cell": {
                            "batch_size": batch,
                            "prompt_tokens": 2048,
                            "decode_tokens": 512,
                        },
                    },
                    "comparison": _artifact(comparison_path),
                }
            )
            lanes.append(
                {
                    "model_pair": pair,
                    "batch_size": batch,
                    "artifact": _artifact(lane_path),
                    "rows": 6,
                    "probe_cell": [batch, 2048, 512],
                    "ada_wagv_lora_require_extension": batch == 1,
                    "rkv_policy": (
                        "manual" if pair == PAIRS[2] and batch == 8 else "vkwr_auto"
                    ),
                }
            )

    candidate = tmp_path / "rwkv_candidate.jsonl"
    reference = tmp_path / "qwen_reference.jsonl"
    _write_jsonl(candidate, candidate_rows)
    _write_jsonl(reference, reference_rows)
    sidecar = tmp_path / "rwkv_candidate.sha256"
    sidecar.write_text(f"{_sha(candidate)}  {candidate.name}\n", encoding="utf-8")
    model_hashes = tmp_path / "model_hashes.sha256"
    model_hashes_after = tmp_path / "model_hashes.after.sha256"
    _hash_manifest(model_hashes, [models[pair][1] for pair in PAIRS])
    model_hashes_after.write_bytes(model_hashes.read_bytes())
    pip = tmp_path / "pip-freeze.txt"
    pip.write_text("torch==2.11.0\n", encoding="utf-8")
    runtime = tmp_path / "runtime-lock.json"
    _write_json(
        runtime,
        {
            "schema_version": 1,
            "protocol": "qwen35_paired_pd_v1",
            "repository_commit": COMMIT,
            "runtime": {
                "python": "3.12.2",
                "torch": "2.11.0+cu130",
                "torch_cuda": "13.0",
                "triton": "3.6.0",
                "transformers": "5.12.1",
                "fla": "0.5.1",
                "causal_conv1d": "1.6.2.post1",
            },
            "pip_freeze_sha256": _sha(pip),
            "torch_cuda_arch_list": "8.9",
        },
    )
    system = tmp_path / "system.csv"
    system.write_text(
        "name, uuid, pci.bus_id, compute_cap, driver_version, memory.total [MiB]\n"
        "NVIDIA GeForce RTX 4080, GPU-x, 00000000:01:00.0, 8.9, 595.71.05, 16376 MiB\n",
        encoding="utf-8",
    )
    correctness = tmp_path / "rwkv_native_graph_fla_correctness.json"
    _write_json(
        correctness,
        {
            "schema_version": 1,
            "protocol": "rwkv_native_graph_fla_correctness_4080_v1",
            "benchmark_repository_commit": COMMIT,
            "model_hashes_sha256": _sha(model_hashes),
            "runtime": _artifact(runtime),
            "coverage": {
                "models": 3,
                "batch_sizes": [1, 8],
                "entries": 6,
                "baseline_fresh_gpu_processes": 6,
                "candidate_additional_gpu_processes": 0,
                "candidate_formal_lane_processes": 6,
                "prompt_tokens": 2048,
                "decode_tokens": 512,
                "probe_tokens": 512,
            },
            "reference_contract": {
                "rwkv_implementation": "wrapper_repo",
                "RWKV7_FAST_TOKEN_BACKEND": "fla",
                "RWKV7_NATIVE_MODEL_BACKEND": "eager",
                "RWKV7_FAST_PREFILL": "0",
                "RWKV7_NATIVE_PREFILL_GRAPH": "0",
                "TORCHDYNAMO_DISABLE": "1",
                "TORCH_COMPILE_DISABLE": "1",
                "performance_role": False,
            },
            "gates": {
                "greedy_tokens": "exact_all_512",
                "prompt_logits_min_row_cosine": 0.9999,
                "final_logits_min_row_cosine": 0.9999,
                "decode_logits_all_finite": True,
                "b8_distinct_prompts": True,
            },
            "entries": correctness_entries,
        },
    )
    route = tmp_path / "rwkv_candidate_routes.json"
    _write_json(
        route,
        {
            "schema_version": 1,
            "protocol": "qwen35_paired_pd_v1",
            "benchmark_repository_commit": COMMIT,
            "repository_clean_pre_and_post": True,
            "candidate_rows": 36,
            "candidate_result": _artifact(candidate),
            "candidate_sha256_sidecar": _artifact(sidecar),
            "model_hash_contract": {
                "algorithm": "sha256",
                "scope": "every recursive regular file",
                "before": _artifact(model_hashes),
                "after": _artifact(model_hashes_after),
                "byte_identical": True,
            },
            "native_graph_fla_correctness_manifest": _artifact(correctness),
            "runtime_lock": _artifact(runtime),
            "pip_freeze": _artifact(pip),
            "system_identity": _artifact(system),
            "forced_environment": {
                "CUDA_VISIBLE_DEVICES": "0",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "PYTHONPATH": "/repo",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_CUDA_ARCH_LIST": "8.9",
                "CPATH": "/cuda-components/include",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "RWKV7_FAST_TOKEN_BACKEND": "native_graph",
                "RWKV7_NATIVE_MODEL_BACKEND": "native_graph",
                "RWKV7_NATIVE_PREFILL_GRAPH": "unset_exact_card_policy",
                "RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA_REQUIRE_EXTENSION": "1_for_B1_0_for_B8",
                "RWKV7_NATIVE_GRAPH_RKV_POLICY": "vkwr_auto_except_2p9_B8_manual",
                "RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM": "1",
                "RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G": "0",
                "RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN": "0",
                "CACHE_ROOT": "/cache/rwkv",
            },
            "lanes": lanes,
        },
    )

    qwen_manifests = []
    qwen_contracts = {
        PAIRS[0]: ("0.8b", "static_cache_inductor_cudagraph", "max-autotune"),
        PAIRS[1]: ("2b", "static_cache_inductor_cudagraph", "max-autotune"),
        PAIRS[2]: ("4b", "module_call_dynamic", None),
    }
    for index, pair in enumerate(PAIRS):
        size, decode_route, compile_mode = qwen_contracts[pair]
        qwen_result = tmp_path / f"qwen_{size}.jsonl"
        _write_jsonl(
            qwen_result, [row for row in reference_rows if row["model_pair"] == pair]
        )
        hashes = tmp_path / f"qwen_{size}_model_hashes.sha256"
        hashes_after = tmp_path / f"qwen_{size}_model_hashes.after.sha256"
        _hash_manifest(hashes, [models[pair][2]])
        hashes_after.write_bytes(hashes.read_bytes())
        manifest = tmp_path / f"qwen_{size}_route.json"
        _write_json(
            manifest,
            {
                "schema_version": 1,
                "protocol": "qwen35_best_optimized_hf_4080_v1",
                "benchmark_repository_commit": COMMIT,
                "repository_clean_pre_and_post": True,
                "model_pair": pair,
                "model_size_label": size,
                "model_path": models[pair][2],
                "result": _artifact(qwen_result),
                "model_hash_contract": {
                    "algorithm": "sha256",
                    "scope": "every recursive regular file",
                    "before": _artifact(hashes),
                    "after": _artifact(hashes_after),
                    "byte_identical": True,
                },
                "decode_route": decode_route,
                "compile_mode": compile_mode,
                "forced_environment": {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "PYTHONPATH": "/repo",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "TORCH_CUDA_ARCH_LIST": "8.9",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                    "CACHE_ROOT": f"/cache/qwen-{index}",
                },
            },
        )
        qwen_manifests.append(manifest)
    return {
        "candidate": candidate,
        "reference": reference,
        "route": route,
        "correctness": correctness,
        "runtime": runtime,
        "model_hashes": model_hashes,
        "qwen_manifests": qwen_manifests,
        "system": system,
    }


def _validate(paths: dict[str, object]) -> dict:
    return validate_bundle(
        candidate=paths["candidate"],
        reference=paths["reference"],
        candidate_route_manifest=paths["route"],
        correctness_manifest=paths["correctness"],
        runtime_lock=paths["runtime"],
        candidate_model_hashes=paths["model_hashes"],
        qwen_route_manifests=paths["qwen_manifests"],
        expected_candidate_commit=COMMIT,
    )


def test_complete_4080_bundle_passes(tmp_path: Path) -> None:
    summary = _validate(_bundle(tmp_path))
    assert summary["status"] == "pass", summary["errors"]
    assert summary["bundle_authenticated"] is True
    assert len(summary["evidence"]["correctness_entries"]) == 6


def test_system_tamper_fails_even_with_refreshed_manifest_sha(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    system = paths["system"]
    system.write_text(
        system.read_text(encoding="utf-8").replace("RTX 4080", "RTX 4090"),
        encoding="utf-8",
    )
    route = json.loads(paths["route"].read_text(encoding="utf-8"))
    route["system_identity"]["sha256"] = _sha(system)
    _write_json(paths["route"], route)
    summary = _validate(paths)
    assert summary["status"] == "fail"
    assert any("system.csv:name" in error for error in summary["errors"])


def test_probe_or_qwen_route_tamper_fails_closed(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    manifest = paths["qwen_manifests"][0]
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["decode_route"] = "static_cache_raw_cudagraph"
    _write_json(manifest, doc)
    summary = _validate(paths)
    assert summary["status"] == "fail"
    assert any("decode_route" in error for error in summary["errors"])


def test_cli_failure_removes_a_stale_paired_table(tmp_path: Path, monkeypatch) -> None:
    paths = _bundle(tmp_path)
    manifest = paths["qwen_manifests"][0]
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["decode_route"] = "static_cache_raw_cudagraph"
    _write_json(manifest, doc)
    summary = tmp_path / "validation.json"
    table = tmp_path / "paired.jsonl"
    markdown = tmp_path / "paired.md"
    table.write_text("stale pass\n", encoding="utf-8")
    args = [
        "validator",
        "--candidate",
        str(paths["candidate"]),
        "--reference",
        str(paths["reference"]),
        "--candidate-route-manifest",
        str(paths["route"]),
        "--correctness-manifest",
        str(paths["correctness"]),
        "--runtime-lock",
        str(paths["runtime"]),
        "--candidate-model-hashes",
        str(paths["model_hashes"]),
    ]
    for route in paths["qwen_manifests"]:
        args.extend(["--qwen-route-manifest", str(route)])
    args.extend(
        [
            "--expected-candidate-commit",
            COMMIT,
            "--summary",
            str(summary),
            "--paired-table",
            str(table),
            "--markdown",
            str(markdown),
        ]
    )
    monkeypatch.setattr(sys, "argv", args)
    assert main() == 1
    assert not table.exists()
    assert json.loads(summary.read_text(encoding="utf-8"))["status"] == "fail"
