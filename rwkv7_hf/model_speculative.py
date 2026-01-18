# coding=utf-8
"""Greedy speculative decoding integration for native RWKV-7 CausalLM."""
from __future__ import annotations

import torch

from .model_cache import NativeRWKV7Cache


class _NativeSpeculativeGenerationMixin:
    @torch.inference_mode()
    def rwkv7_speculative_generate(
        self,
        input_ids: torch.LongTensor,
        draft_model: torch.nn.Module,
        max_new_tokens: int = 32,
        draft_tokens: int = 4,
        eos_token_id: int | list[int] | tuple[int, ...] | None = None,
        return_stats: bool = False,
        logits_to_keep: int = 1,
        **forward_kwargs,
    ):
        """Greedy batch-one speculative decoding through standard HF calls."""

        if self.training:
            raise RuntimeError("rwkv7_speculative_generate is inference-only; call model.eval() first")
        if draft_model is None:
            raise ValueError("rwkv7_speculative_generate requires a draft_model")
        if getattr(draft_model, "training", False):
            raise RuntimeError("draft_model must be in eval mode for speculative decoding")
        if input_ids.dim() != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("rwkv7_speculative_generate currently supports input_ids shaped [1, seq]")
        if int(input_ids.shape[1]) <= 0:
            raise ValueError("rwkv7_speculative_generate requires at least one prompt token")
        max_new_tokens = int(max_new_tokens)
        draft_tokens = int(draft_tokens)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if draft_tokens <= 0:
            raise ValueError("draft_tokens must be positive")

        stats = {
            "generated_tokens": 0,
            "proposed_tokens": 0,
            "accepted_tokens": 0,
            "corrected_tokens": 0,
            "resyncs": 0,
            "resync_tokens": 0,
            "full_resync_tokens": 0,
            "resync_saved_tokens": 0,
            "target_forward_calls": 0,
            "draft_forward_calls": 0,
            "acceptance_rate": None,
        }
        if max_new_tokens == 0:
            return {"sequences": input_ids, "stats": stats} if return_stats else input_ids

        eos_ids = (
            {int(eos_token_id)}
            if isinstance(eos_token_id, int)
            else ({int(value) for value in eos_token_id} if eos_token_id is not None else set())
        )
        prefill_kwargs = dict(forward_kwargs)
        step_kwargs = {
            key: value
            for key, value in forward_kwargs.items()
            if key
            not in {
                "attention_mask",
                "position_ids",
                "cache_position",
                "past_key_values",
                "use_cache",
                "return_dict",
                "logits_to_keep",
            }
        }

        def _forward(model, tokens, past=None, *, prefill: bool = False, keep: int | None = None):
            call_kwargs = dict(prefill_kwargs if prefill else step_kwargs)
            for key in ("past_key_values", "use_cache", "return_dict", "logits_to_keep"):
                call_kwargs.pop(key, None)
            return model(
                tokens,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
                logits_to_keep=logits_to_keep if keep is None else keep,
                **call_kwargs,
            )

        def _argmax_token(logits: torch.Tensor) -> torch.LongTensor:
            return torch.argmax(logits[:, -1, :], dim=-1).to(device=input_ids.device)

        def _append_token(sequence: torch.LongTensor, token: torch.LongTensor) -> torch.LongTensor:
            return torch.cat([sequence, token.reshape(1, 1).to(sequence.device)], dim=1)

        def _append_tokens(sequence: torch.LongTensor, tokens: list[torch.LongTensor]) -> torch.LongTensor:
            if not tokens:
                return sequence
            return torch.cat(
                [sequence] + [token.reshape(1, 1).to(sequence.device) for token in tokens],
                dim=1,
            )

        def _is_eos(token: torch.LongTensor) -> bool:
            return bool(eos_ids and int(token.reshape(-1)[0].detach().cpu()) in eos_ids)

        def _clone_past(past):
            if hasattr(past, "clone"):
                return past.clone()
            return NativeRWKV7Cache.from_legacy_cache(past).clone()

        generated = input_ids
        target_out = _forward(self, generated, prefill=True)
        stats["target_forward_calls"] += 1
        target_past = target_out.past_key_values
        target_next = _argmax_token(target_out.logits)

        draft_out = _forward(draft_model, generated, prefill=True)
        stats["draft_forward_calls"] += 1
        draft_past = draft_out.past_key_values
        draft_next = _argmax_token(draft_out.logits)

        while stats["generated_tokens"] < max_new_tokens:
            proposals: list[torch.LongTensor] = []
            draft_past_before_block = _clone_past(draft_past)
            for _ in range(min(draft_tokens, max_new_tokens - stats["generated_tokens"])):
                proposal = draft_next.reshape(1).to(input_ids.device)
                proposals.append(proposal)
                stats["proposed_tokens"] += 1
                draft_out = _forward(draft_model, proposal.reshape(1, 1), past=draft_past)
                stats["draft_forward_calls"] += 1
                draft_past = draft_out.past_key_values
                draft_next = _argmax_token(draft_out.logits)
            if not proposals:
                break

            proposal_ids = torch.cat(
                [proposal.reshape(1, 1).to(input_ids.device) for proposal in proposals],
                dim=1,
            )
            verify_out = _forward(
                self,
                proposal_ids,
                past=_clone_past(target_past),
                keep=len(proposals),
            )
            stats["target_forward_calls"] += 1
            verify_logits = verify_out.logits
            target_predictions = [target_next.reshape(1)]
            for position in range(max(0, len(proposals) - 1)):
                target_predictions.append(
                    torch.argmax(verify_logits[:, position, :], dim=-1).to(input_ids.device)
                )

            accepted_prefix: list[torch.LongTensor] = []
            mismatch = False
            stop_after_append = False
            for index, proposal in enumerate(proposals):
                expected = target_predictions[index].reshape(1)
                if int(proposal.reshape(-1)[0]) == int(expected.reshape(-1)[0]):
                    accepted_prefix.append(proposal)
                    stats["accepted_tokens"] += 1
                    stats["generated_tokens"] += 1
                    if _is_eos(proposal) or stats["generated_tokens"] >= max_new_tokens:
                        stop_after_append = True
                        break
                    continue

                generated = _append_tokens(generated, accepted_prefix)
                correction = expected
                generated = _append_token(generated, correction)
                stats["corrected_tokens"] += 1
                stats["generated_tokens"] += 1
                mismatch = True
                if not _is_eos(correction) and stats["generated_tokens"] < max_new_tokens:
                    repair_tokens = torch.cat(
                        [
                            token.reshape(1, 1).to(input_ids.device)
                            for token in [*accepted_prefix, correction]
                        ],
                        dim=1,
                    )
                    target_out = _forward(
                        self,
                        repair_tokens,
                        past=_clone_past(target_past),
                        keep=1,
                    )
                    stats["target_forward_calls"] += 1
                    target_past = target_out.past_key_values
                    target_next = _argmax_token(target_out.logits)
                    draft_out = _forward(
                        draft_model,
                        repair_tokens,
                        past=draft_past_before_block,
                        keep=1,
                    )
                    stats["draft_forward_calls"] += 1
                    draft_past = draft_out.past_key_values
                    draft_next = _argmax_token(draft_out.logits)
                    stats["resyncs"] += 1
                    stats["resync_tokens"] += int(repair_tokens.shape[1])
                    stats["full_resync_tokens"] += int(generated.shape[1])
                    stats["resync_saved_tokens"] = max(
                        0,
                        int(stats["full_resync_tokens"]) - int(stats["resync_tokens"]),
                    )
                stop_after_append = True
                break

            if not mismatch:
                generated = _append_tokens(generated, accepted_prefix)
                if len(accepted_prefix) == len(proposals):
                    target_past = verify_out.past_key_values
                    target_next = _argmax_token(verify_logits)
                elif not stop_after_append:
                    target_out = _forward(self, generated, prefill=True)
                    stats["target_forward_calls"] += 1
                    target_past = target_out.past_key_values
                    target_next = _argmax_token(target_out.logits)

            if _is_eos(generated[:, -1]) or stats["generated_tokens"] >= max_new_tokens:
                break

        if stats["proposed_tokens"]:
            stats["acceptance_rate"] = float(stats["accepted_tokens"]) / float(
                stats["proposed_tokens"]
            )
        return {"sequences": generated, "stats": stats} if return_stats else generated
