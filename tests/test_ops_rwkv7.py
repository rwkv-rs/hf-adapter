from __future__ import annotations

import torch

from rwkv7_hf.ops_rwkv7 import rwkv7_recurrent


def official_vk_reference(r, w, k, v, a, b, state_kv, mask=None):
    state = state_kv.transpose(-1, -2)
    outputs = []
    for token_idx in range(r.shape[1]):
        candidate = (
            state * w[:, token_idx].unsqueeze(-2)
            + state
            @ (
                a[:, token_idx].unsqueeze(-1)
                @ b[:, token_idx].unsqueeze(-2)
            )
            + v[:, token_idx].unsqueeze(-1)
            @ k[:, token_idx].unsqueeze(-2)
        )
        output = (candidate @ r[:, token_idx].unsqueeze(-1)).squeeze(-1)
        if mask is not None:
            active = mask[:, token_idx]
            state = torch.where(active[:, None, None, None], candidate, state)
            output = torch.where(active[:, None, None], output, torch.zeros_like(output))
        else:
            state = candidate
        outputs.append(output)
    return torch.stack(outputs, dim=1), state.transpose(-1, -2)


def test_operator_output_state_and_gradients_match_official_equation():
    torch.manual_seed(123)
    shape = (4, 17, 2, 4)
    tensors = [torch.randn(shape, dtype=torch.float64, requires_grad=True) for _ in range(6)]
    state = torch.randn(4, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(4, 17, dtype=torch.bool)
    mask[1, :3] = False
    mask[2, -4:] = False

    output, final_state = rwkv7_recurrent(*tensors, state, mask)
    expected_output, expected_state = official_vk_reference(*tensors, state, mask)
    torch.testing.assert_close(output, expected_output, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(final_state, expected_state, rtol=1e-12, atol=1e-12)

    loss = output.square().mean() + final_state.square().mean()
    expected_loss = expected_output.square().mean() + expected_state.square().mean()
    gradients = torch.autograd.grad(loss, tensors + [state], retain_graph=True)
    expected_gradients = torch.autograd.grad(expected_loss, tensors + [state])
    for actual, expected in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
