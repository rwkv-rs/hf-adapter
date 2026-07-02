#!/usr/bin/env python3
# coding=utf-8
"""Regression tests for the RWKV byte-level HF tokenizer fast trie."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable

from rwkv7_hf.tokenization_rwkv7 import RWKV7Tokenizer


def _token_bytes(token: bytes | str) -> bytes:
    return token.encode("utf-8") if isinstance(token, str) else token


def _write_vocab(path: Path, tokens: Iterable[tuple[int, bytes | str]]) -> dict[bytes, int]:
    token2idx: dict[bytes, int] = {}
    with path.open("w", encoding="utf-8") as f:
        for idx, token in tokens:
            raw = _token_bytes(token)
            token2idx[raw] = idx
            f.write(f"{idx} {token!r} {len(raw)}\n")
    return token2idx


def _toy_tokens() -> tuple[list[str], list[tuple[int, bytes | str]], dict[bytes, int]]:
    samples = [
        "",
        "hello world",
        "hello RWKV 10 99",
        "你好，RWKV🙂",
        "起業家イーロン・マスク",
        "tabs\tand\nnewlines",
    ]
    byte_set = sorted({b for sample in samples for b in sample.encode("utf-8")})
    tokens: list[tuple[int, bytes | str]] = []
    next_id = 1
    for b in byte_set:
        tokens.append((next_id, bytes([b])))
        next_id += 1
    for token in ["hello", " world", "你好", "🙂", b" 10", b"RWKV", "起業"]:
        tokens.append((next_id, token))
        next_id += 1
    token2idx = {_token_bytes(token): idx for idx, token in tokens}
    return samples, tokens, token2idx


def test_fast_trie_greedy_roundtrip() -> None:
    samples, tokens, token2idx = _toy_tokens()
    with tempfile.TemporaryDirectory() as td:
        vocab = Path(td) / "rwkv_vocab_v20230424.txt"
        _write_vocab(vocab, tokens)
        tok = RWKV7Tokenizer(str(vocab), model_vocab_size=128)

        assert tok.trie.encode("hello world") == [token2idx[b"hello"], token2idx[b" world"]]
        assert tok.trie.encode("你好🙂") == [token2idx["你好".encode()], token2idx["🙂".encode()]]
        assert tok.trie.encode("RWKV") == [token2idx[b"RWKV"]]

        for sample in samples:
            ids = tok(sample, add_special_tokens=False)["input_ids"]
            assert tok.decode(ids) == sample
            assert tok.trie.decode(tok.trie.encode(sample)) == sample


def test_fast_trie_save_vocab_and_missing_ids() -> None:
    _, tokens, token2idx = _toy_tokens()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        vocab = root / "rwkv_vocab_v20230424.txt"
        _write_vocab(vocab, tokens)
        tok = RWKV7Tokenizer(str(vocab), model_vocab_size=128)
        saved = tok.save_vocabulary(str(root / "saved"))
        reloaded = RWKV7Tokenizer(saved[0], model_vocab_size=128)

        ids = [token2idx[b"hello"], -1, 9999, token2idx[b" world"]]
        assert tok.trie.decode(ids) == "hello world"
        assert reloaded.trie.encode("hello world") == [token2idx[b"hello"], token2idx[b" world"]]


def test_fast_trie_uses_literal_eval_for_vocab_lines() -> None:
    with tempfile.TemporaryDirectory() as td:
        vocab = Path(td) / "rwkv_vocab_v20230424.txt"
        vocab.write_text("1 __import__('os').system('false') 1\n", encoding="utf-8")
        try:
            RWKV7Tokenizer(str(vocab), model_vocab_size=8)
        except (ValueError, SyntaxError):
            pass
        else:
            raise AssertionError("malformed vocab literal should not be accepted")


def main() -> int:
    test_fast_trie_greedy_roundtrip()
    test_fast_trie_save_vocab_and_missing_ids()
    test_fast_trie_uses_literal_eval_for_vocab_lines()
    print("TOKENIZER FAST TRIE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
