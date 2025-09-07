# coding=utf-8
"""Slow Hugging Face tokenizer for RWKV vocab v20230424.

The RWKV vocab is a byte-level trie vocabulary stored as lines:
    <id> <python repr token> <byte length>
IDs are kept identical to the official RWKV tokenizer.
"""
from __future__ import annotations

import os
import shutil
from typing import Dict, Iterable, List, Optional, Tuple

from transformers import PreTrainedTokenizer

VOCAB_FILES_NAMES = {"vocab_file": "rwkv_vocab_v20230424.txt"}


class _TrieNode:
    __slots__ = ("children", "value")

    def __init__(self):
        self.children: Dict[int, "_TrieNode"] = {}
        self.value: Optional[Tuple[bytes, int]] = None

    def add(self, key: bytes, value: Tuple[bytes, int]) -> None:
        node = self
        for b in key:
            node = node.children.setdefault(b, _TrieNode())
        node.value = value

    def find_longest(self, src: bytes, start: int) -> Tuple[int, Tuple[bytes, int]]:
        node = self
        best_pos = start
        best = None
        pos = start
        while pos < len(src):
            nxt = node.children.get(src[pos])
            if nxt is None:
                break
            node = nxt
            pos += 1
            if node.value is not None:
                best_pos = pos
                best = node.value
        if best is None:
            raise ValueError(f"RWKV tokenizer cannot encode byte at offset {start}: {src[start:start+8]!r}")
        return best_pos, best


class _RWKVTrie:
    def __init__(self, vocab_file: str):
        self.vocab_file = vocab_file
        self.idx2token: Dict[int, bytes] = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                first = line.index(" ")
                last = line.rindex(" ")
                idx = int(line[:first])
                token_obj = eval(line[first + 1:last])  # official RWKV vocab format
                token = token_obj.encode("utf-8") if isinstance(token_obj, str) else token_obj
                if not isinstance(token, bytes):
                    raise TypeError(f"Invalid token type at id={idx}: {type(token)}")
                expected_len = int(line[last + 1:])
                if len(token) != expected_len:
                    raise ValueError(f"Length mismatch at id={idx}: got {len(token)} expected {expected_len}")
                self.idx2token[idx] = token
        self.token2idx = {v: k for k, v in self.idx2token.items()}
        self.root = _TrieNode()
        for token, idx in self.token2idx.items():
            self.root.add(token, (token, idx))
        self.max_id = max(self.idx2token) if self.idx2token else 0

    def encode(self, text: str) -> List[int]:
        src = text.encode("utf-8")
        pos = 0
        out: List[int] = []
        while pos < len(src):
            pos, (_, idx) = self.root.find_longest(src, pos)
            out.append(idx)
        return out

    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        chunks = []
        for i in ids:
            tok = self.idx2token.get(int(i))
            # RWKV-7 checkpoints have a few unused embedding rows. Treat them as empty specials.
            if tok is not None:
                chunks.append(tok)
        return b"".join(chunks).decode("utf-8", errors=errors)


class RWKV7Tokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        errors: str = "replace",
        model_vocab_size: int = 65536,
        pad_token: Optional[str] = "<|padding|>",
        eos_token: Optional[str] = "<|endoftext|>",
        bos_token: Optional[str] = None,
        unk_token: Optional[str] = None,
        **kwargs,
    ):
        self.vocab_file = vocab_file
        self.errors = errors
        self.trie = _RWKVTrie(vocab_file)
        self.model_vocab_size = max(int(model_vocab_size), self.trie.max_id + 1)
        # The official vocab starts at id 1; id 0 is unused by the tokenizer and is convenient for padding.
        self._special_ids = {}
        if pad_token is not None:
            self._special_ids[pad_token] = 0
        if eos_token is not None:
            self._special_ids[eos_token] = 0
        if bos_token is not None:
            self._special_ids[bos_token] = 1
        super().__init__(
            pad_token=pad_token,
            eos_token=eos_token,
            bos_token=bos_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return self.model_vocab_size

    def get_vocab(self) -> Dict[str, int]:
        vocab = {str(i): i for i in range(self.model_vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return [str(i) for i in self.trie.encode(text)]

    def _convert_token_to_id(self, token: str) -> int:
        if token in self._special_ids:
            return self._special_ids[token]
        try:
            idx = int(token)
        except (TypeError, ValueError):
            return 0 if self.unk_token is None else self.unk_token_id
        if 0 <= idx < self.model_vocab_size:
            return idx
        return 0 if self.unk_token is None else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        index = int(index)
        for tok, idx in self._special_ids.items():
            if idx == index:
                return tok
        return str(index)


    def convert_tokens_to_ids(self, tokens):
        if tokens is None:
            return None
        if isinstance(tokens, str):
            if tokens in self._special_ids:
                return self._special_ids[tokens]
            return super().convert_tokens_to_ids(tokens)
        return [self.convert_tokens_to_ids(t) for t in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens: bool = False):
        if isinstance(ids, int):
            if skip_special_tokens and ids in set(self._special_ids.values()):
                return None
            return self._convert_id_to_token(ids)
        out = []
        special_values = set(self._special_ids.values())
        for i in ids:
            i = int(i)
            if skip_special_tokens and i in special_values:
                continue
            out.append(self._convert_id_to_token(i))
        return out

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        ids = []
        for tok in tokens:
            if tok in self._special_ids:
                continue
            try:
                ids.append(int(tok))
            except (TypeError, ValueError):
                continue
        return self.trie.decode(ids, errors=self.errors)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return list(token_ids_0)
        return list(token_ids_0) + list(token_ids_1)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        os.makedirs(save_directory, exist_ok=True)
        out_name = (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        out_path = os.path.join(save_directory, out_name)
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_path):
            shutil.copyfile(self.vocab_file, out_path)
        return (out_path,)
