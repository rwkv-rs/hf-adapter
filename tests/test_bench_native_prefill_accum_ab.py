from __future__ import annotations

import pytest

from bench.bench_native_prefill_accum_ab import (
    mode_flags,
    model_shape_spec,
    route_effective_matches,
    sweep_orders,
)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("off", ("0", "0")),
        ("global", ("1", "0")),
        ("block", ("0", "1")),
    ],
)
def test_mode_flags(mode: str, expected: tuple[str, str]) -> None:
    assert mode_flags(mode) == expected


def test_mode_flags_reject_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unsupported accumulation mode"):
        mode_flags("unknown")


def test_sweep_orders_support_forward_reverse_and_both() -> None:
    assert sweep_orders("forward") == (("off", "global", "block"),)
    assert sweep_orders("reverse") == (("block", "global", "off"),)
    assert sweep_orders("both") == (
        ("off", "global", "block"),
        ("block", "global", "off"),
    )


def test_route_effective_match_is_exact() -> None:
    assert route_effective_matches("off", False, False)
    assert route_effective_matches("global", True, False)
    assert route_effective_matches("block", False, True)
    assert not route_effective_matches("global", True, True)
    assert not route_effective_matches("block", False, False)


def test_model_shape_spec_is_deterministic() -> None:
    assert model_shape_spec(1024, 24, [1, 8], [128, 512]) == (
        "1024x24x1x128 1024x24x1x512 "
        "1024x24x8x128 1024x24x8x512"
    )
