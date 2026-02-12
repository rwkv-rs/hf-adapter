from __future__ import annotations

from scripts.prepare_rwkv7_g1_validation_models import CHECKPOINTS, selected_specs


def test_validation_checkpoint_manifest_is_complete_and_pinned() -> None:
    assert [spec.label for spec in CHECKPOINTS] == [
        "0.4b",
        "1.5b",
        "2.9b",
        "7.2b",
        "13.3b",
    ]
    assert len({spec.filename for spec in CHECKPOINTS}) == len(CHECKPOINTS)
    assert len({spec.output_name for spec in CHECKPOINTS}) == len(CHECKPOINTS)
    for spec in CHECKPOINTS:
        assert len(spec.sha256) == 64
        assert all(character in "0123456789abcdef" for character in spec.sha256)
        assert spec.size_bytes > 0


def test_validation_checkpoint_selection_keeps_manifest_order() -> None:
    assert [spec.label for spec in selected_specs(["7.2b", "0.4b"])] == [
        "0.4b",
        "7.2b",
    ]
    assert selected_specs(["all"]) == list(CHECKPOINTS)
