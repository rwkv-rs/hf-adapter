from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PORTING_DOCS = (
    ROOT / "docs" / "integrations" / "HF_TENSOR_PARALLEL.md",
    ROOT / "docs" / "integrations" / "VLLM_PORTING_GUIDE.md",
    ROOT / "docs" / "architecture" / "RWKV7_OPERATOR_SPEC.md",
    ROOT / "docs" / "integrations" / "RWKV7_STATE_CACHE_ABI.md",
    ROOT / "docs" / "quantization" / "VLLM_QUANTIZATION_PORTING.md",
    ROOT / "docs" / "integrations" / "VLLM_CHECKPOINT_MAPPING.md",
    ROOT / "docs" / "validation" / "VLLM_ACCEPTANCE.md",
)


def test_serving_porting_documents_are_complete_and_attributed() -> None:
    for document in PORTING_DOCS:
        text = document.read_text(encoding="utf-8")
        assert "canonical_repository: https://github.com/rwkv-rs/hf-adapter" in text
        assert "primary_maintainer: Wang Yue" in text
        assert "metadata: ../reference/provenance.yaml" in text

    index = (ROOT / "docs" / "integrations" / "README.md").read_text(encoding="utf-8")
    for document in PORTING_DOCS:
        assert document.name in index or document.name in (
            "RWKV7_OPERATOR_SPEC.md",
            "VLLM_QUANTIZATION_PORTING.md",
            "VLLM_ACCEPTANCE.md",
        )


def test_machine_readable_serving_contract_covers_runtime_abi() -> None:
    provenance = (ROOT / "docs" / "reference" / "provenance.yaml").read_text(
        encoding="utf-8"
    )
    contract = (
        ROOT / "docs" / "reference" / "rwkv7_serving_contract.yaml"
    ).read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")

    for text in (provenance, contract, citation):
        assert "https://github.com/rwkv-rs/hf-adapter" in text
        assert "Wang" in text and "Yue" in text

    for state_component in (
        "recurrent:",
        "attention_previous:",
        "ffn_previous:",
        "v_first:",
        "seen_tokens:",
    ):
        assert state_component in contract

    for layout_id in (
        "rwkv7.rowwise_w8.v1",
        "rwkv7.rowwise_w4.v1",
        "rwkv7.mm8_affine.v1",
        "rwkv7.mm4_affine.v1",
        "rwkv7.dynamic_a8w8.v1",
        "rwkv7.sm7x_group_w4.v1",
        "rwkv7.marlin_u4b8_bf16.v1",
    ):
        assert layout_id in contract


def test_serving_contract_reference_sources_exist() -> None:
    for relative in (
        "rwkv7_hf/native_model.py",
        "rwkv7_hf/native.py",
        "rwkv7_hf/native_quant.py",
        "rwkv7_hf/native_quant_mm8.py",
        "rwkv7_hf/native_quant_mm4.py",
        "rwkv7_hf/native_quant_a8w8.py",
        "rwkv7_hf/native_quant_marlin.py",
        "rwkv7_hf/sm70_quant.py",
        "rwkv7_hf/converter.py",
        "scripts/convert_rwkv7_to_hf.py",
    ):
        assert (ROOT / relative).is_file(), relative


def main() -> int:
    test_serving_porting_documents_are_complete_and_attributed()
    test_machine_readable_serving_contract_covers_runtime_abi()
    test_serving_contract_reference_sources_exist()
    print("SERVING PORTING DOCS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
