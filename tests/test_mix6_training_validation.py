from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


EVALUATION = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION))

import validate_mix6_training as validator  # noqa: E402


class FakeKernelAPI:
    def __init__(self, envelope):
        self.envelope = envelope
        self.calls = []

    def execute_optional_v4(self, kind, *args, **kwargs):
        self.calls.append((kind, args, kwargs))
        return self.envelope


def test_validator_executes_mix6_through_public_api_v4():
    outputs = tuple(torch.randn(2, 3, 4) for _ in range(6))
    api = FakeKernelAPI(
        {
            "kind": "mix6_training",
            "supported": True,
            "implementation": "test-mix6-v4",
            "reason": "executed",
            "result": outputs,
            "phase": "training",
        }
    )
    inputs = tuple(torch.randn(2, 3, 4) for _ in range(8))

    assert validator.public_mix6_training_v1(api, *inputs) == outputs
    assert api.calls == [
        ("mix6_training", inputs, {"program_id": None, "facts": {}})
    ]


@pytest.mark.parametrize(
    ("envelope", "error"),
    [
        (None, "non-mapping"),
        ({"kind": "recurrent", "supported": True}, "wrong API-v4 kind"),
        (
            {
                "kind": "mix6_training",
                "supported": False,
                "reason": "unsupported shape",
            },
            "unsupported shape",
        ),
        (
            {
                "kind": "mix6_training",
                "supported": True,
                "result": (torch.zeros(1),),
            },
            "six tensors",
        ),
    ],
)
def test_validator_rejects_invalid_public_mix6_envelopes(envelope, error):
    with pytest.raises((TypeError, ValueError, RuntimeError), match=error):
        validator.public_mix6_training_v1(FakeKernelAPI(envelope))
