#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "bench" / "musa_s70_shift_mix_20260728"


def test_musa_s70_shift_mix_evidence_is_complete_and_fail_closed() -> None:
    summary = json.loads((EVIDENCE / "summary.json").read_text(encoding="utf-8"))
    aggregate = summary["aggregate"]
    assert summary["scope"]["device"] == "MTT S70"
    assert summary["scope"]["fusion_default"] == "off"
    assert aggregate["cell_count"] == 16
    assert aggregate["prefill_ratio_min"] > 1.0
    assert aggregate["decode_ratio_min"] >= 0.99
    assert aggregate["peak_memory_equal"]
    assert aggregate["route_valid"]
    assert summary["correctness"]["generated_ids_equal"]
    assert summary["correctness"]["state_compare_passed"]
    assert summary["correctness"]["logits_equal"]
    assert summary["correctness"]["all_state_groups_equal"]


def test_musa_s70_shift_mix_evidence_hashes_match() -> None:
    for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(None, 1)
        path = EVIDENCE.joinpath(*relative.strip().split("/"))
        assert path.is_file(), relative
        payload = path.read_bytes()
        if os.name == "nt":
            payload = payload.replace(b"\r\n", b"\n")
        assert hashlib.sha256(payload).hexdigest() == expected
