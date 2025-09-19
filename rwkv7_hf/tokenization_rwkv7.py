# coding=utf-8
"""Hugging Face tokenizer for RWKV vocab v20230424.

The RWKV vocab is a byte-level trie vocabulary stored as lines:
    <id> <python repr token> <byte length>
IDs are kept identical to the official RWKV tokenizer.

The internal trie uses a dense 256-way child table per node, following the fast
Python trie-tokenizer layout used by ChatRWKV while keeping the public
``PreTrainedTokenizer`` contract unchanged.
"""
from __future__ import annotations

import os
import shutil
from ast import literal_eval
from typing import Dict, Iterable, List, Optional, Tuple

from transformers import PreTrainedTokenizer

VOCAB_FILES_NAMES = {"vocab_file": "rwkv_vocab_v20230424.txt"}


class _TrieNode:
    __slots__ = ("children", "token_plus_one")

    def __init__(self):
        # Dense byte-indexed children are faster than a dict for RWKV's byte-level
        # tokenizer and keep encode() on the pure-Python HF slow-tokenizer path cheap.
        self.children: List[Optional["_TrieNode"]] = [None] * 256
        # Store id + 1 so token id 0 remains representable while 0 means no token.
        self.token_plus_one = 0

    def add(self, key: bytes, token_id: int) -> None:
        node = self
        for b in key:
            child = node.children[b]
            if child is None:
                child = _TrieNode()
                node.children[b] = child
            node = child
        node.token_plus_one = int(token_id) + 1


class _RWKVTrie:
    def __init__(self, vocab_file: str):
        self.vocab_file = vocab_file
        parsed: Dict[int, bytes] = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                idx, token = self._parse_vocab_line(line)
                parsed[idx] = token

        self.max_id = max(parsed) if parsed else 0
        # RWKV-7 checkpoints can have unused embedding rows. Missing ids decode to
        # empty bytes to preserve the previous "ignore unknown embedding rows" behavior.
        self.idx2token: List[bytes] = [b""] * (self.max_id + 1)
        for idx, token in parsed.items():
            self.idx2token[idx] = token
        self.token2idx = {token: idx for idx, token in parsed.items()}
        self.root = _TrieNode()
        for token, idx in self.token2idx.items():
            self.root.add(token, idx)

    @staticmethod
    def _parse_vocab_line(line: str) -> Tuple[int, bytes]:
        first = line.index(" ")
        last = line.rindex(" ")
        idx = int(line[:first])
        token_obj = literal_eval(line[first + 1 : last])
        token = token_obj.encode("utf-8") if isinstance(token_obj, str) else token_obj
        if not isinstance(token, bytes):
            raise TypeError(f"Invalid token type at id={idx}: {type(token)}")
        expected_len = int(line[last + 1 :])
        if len(token) != expected_len:
            raise ValueError(f"Length mismatch at id={idx}: got {len(token)} expected {expected_len}")
        return idx, token

    def encode_bytes(self, src: bytes) -> List[int]:
        idx = 0
        src_len = len(src)
        out: List[int] = []
        append = out.append
        root_children = self.root.children
        while idx < src_len:
            node = root_children[src[idx]]
            if node is None:
                raise ValueError(f"RWKV tokenizer cannot encode byte at offset {idx}: {src[idx:idx+8]!r}")

            pos = idx + 1
            token_plus_one = node.token_plus_one
            end = pos
            children = node.children
            while pos < src_len:
                node = children[src[pos]]
                if node is None:
                    break
                pos += 1
                if node.token_plus_one:
                    token_plus_one = node.token_plus_one
                    end = pos
                children = node.children

            if token_plus_one == 0:
                raise ValueError(f"RWKV tokenizer cannot encode byte at offset {idx}: {src[idx:idx+8]!r}")
            append(token_plus_one - 1)
            idx = end
        return out

    def encode(self, text: str) -> List[int]:
        return self.encode_bytes(text.encode("utf-8"))

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        idx2token = self.idx2token
        size = len(idx2token)
        if isinstance(ids, (list, tuple)):
            if not ids:
                return b""
            try:
                if min(ids) >= 0 and max(ids) < size:
                    return b"".join(map(idx2token.__getitem__, ids))
            except TypeError:
                # Fall back for tensor / numpy scalar ids that need int() conversion.
                pass

        chunks: List[bytes] = []
        append = chunks.append
        for token_id in ids:
            idx = int(token_id)
            if 0 <= idx < size:
                token = idx2token[idx]
                if token:
                    append(token)
        return b"".join(chunks)

    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        return self.decode_bytes(ids).decode("utf-8", errors=errors)


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
