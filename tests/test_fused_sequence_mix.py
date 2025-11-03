import pytest

torch = pytest.importorskip("torch")

from rwkv7_hf.fused_elementwise import fused_relu_square
from rwkv7_hf.fused_time_mix import fused_attn_sequence_shift_mix, fused_ffn_sequence_shift_mix


def _shifted(x: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
    return torch.cat([initial[:, None], x[:, :-1]], dim=1)


def test_attention_sequence_shift_mix_cpu_matches_reference_and_updates_state():
    torch.manual_seed(7)
    x = torch.randn(2, 5, 8, dtype=torch.float32)
    initial = torch.randn(2, 8, dtype=torch.float32)
    mixes = [torch.randn(1, 1, 8, dtype=torch.float32) for _ in range(6)]

    *outputs, next_state = fused_attn_sequence_shift_mix(x, initial, *mixes)
    previous = _shifted(x, initial)
    for output, mix in zip(outputs, mixes):
        expected = x + (previous - x) * mix
        torch.testing.assert_close(output, expected)
    torch.testing.assert_close(next_state, x[:, -1])


def test_ffn_sequence_shift_mix_cpu_matches_reference_and_updates_state():
    torch.manual_seed(11)
    x = torch.randn(3, 4, 6, dtype=torch.float32)
    initial = torch.randn(3, 6, dtype=torch.float32)
    mix = torch.randn(6, dtype=torch.float32)

    output, next_state = fused_ffn_sequence_shift_mix(x, initial, mix)
    expected = x + (_shifted(x, initial) - x) * mix.view(1, 1, -1)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(next_state, x[:, -1])


def test_fused_relu_square_cpu_fallback_matches_expression():
    x = torch.tensor([[-2.0, -0.0, 0.5, 3.0]], dtype=torch.float32)
    torch.testing.assert_close(fused_relu_square(x), torch.relu(x) ** 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/Triton kernel test")
def test_sequence_mix_and_relu_square_cuda_kernels_match_reference():
    torch.manual_seed(17)
    device = torch.device("cuda")
    x = torch.randn(2, 5, 64, device=device, dtype=torch.float16)
    initial = torch.randn(2, 64, device=device, dtype=torch.float16)
    mixes = [torch.randn(64, device=device, dtype=torch.float16) for _ in range(6)]
    previous = _shifted(x, initial)

    *outputs, next_state = fused_attn_sequence_shift_mix(x, initial, *mixes)
    for output, mix in zip(outputs, mixes):
        torch.testing.assert_close(output, x + (previous - x) * mix, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(next_state, x[:, -1], rtol=0, atol=0)

    ffn_output, ffn_state = fused_ffn_sequence_shift_mix(x, initial, mixes[0])
    torch.testing.assert_close(ffn_output, x + (previous - x) * mixes[0], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ffn_state, x[:, -1], rtol=0, atol=0)

    relu_input = torch.randn(17, 129, device=device, dtype=torch.float16)
    torch.testing.assert_close(
        fused_relu_square(relu_input),
        torch.relu(relu_input) ** 2,
        rtol=1e-3,
        atol=1e-3,
    )
