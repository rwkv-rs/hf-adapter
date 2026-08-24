"""Lazy Hugging Face loaders shared by benchmark runners."""

from __future__ import annotations

from typing import Any


def load_tokenizer(model: str, **kwargs: Any):
    from transformers import AutoTokenizer

    kwargs.setdefault("trust_remote_code", True)
    return AutoTokenizer.from_pretrained(model, **kwargs)


def load_causal_lm(model: str, *, evaluate: bool = True, **kwargs: Any):
    from transformers import AutoModelForCausalLM

    kwargs.setdefault("trust_remote_code", True)
    loaded = AutoModelForCausalLM.from_pretrained(model, **kwargs)
    return loaded.eval() if evaluate else loaded
