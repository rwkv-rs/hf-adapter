from __future__ import annotations

import json

from rwkv7_hf.marlin_autotune import schedules_for_linear


class _FakeCuda:
    @staticmethod
    def current_device():
        return 0

    @staticmethod
    def get_device_name(_index):
        return "NVIDIA GeForce RTX 5090"

    @staticmethod
    def get_device_capability(_index):
        return (12, 0)


class _FakeDevice:
    index = 0


class _FakeTorch:
    __version__ = "2.11.0+cu128"
    cuda = _FakeCuda()
    version = type("Version", (), {"cuda": "12.8"})()

    @staticmethod
    def device(_device):
        return _FakeDevice()


def _profile():
    return {
        "schema_version": 1,
        "device": "NVIDIA GeForce RTX 5090",
        "compute_capability": [12, 0],
        "torch_version": "2.11.0+cu128",
        "cuda_version": "12.8",
        "entries": [
            {
                "k": 8192,
                "n": 2048,
                "group_size": 32,
                "rows": 128,
                "schedule": [128, 64, 128, -1, -1],
            }
        ],
    }


def test_exact_runtime_profile_selects_only_matching_linear(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_profile()), encoding="utf-8")
    selected = schedules_for_linear(
        device="cuda:0",
        in_features=8192,
        out_features=2048,
        group_size=32,
        torch_module=_FakeTorch,
        profile_path=path,
    )
    assert selected == {128: (128, 64, 128, -1, -1)}
    assert schedules_for_linear(
        device="cuda:0",
        in_features=8192,
        out_features=4096,
        group_size=32,
        torch_module=_FakeTorch,
        profile_path=path,
    ) == {}


def test_profile_identity_and_schema_mismatch_fail_closed(tmp_path) -> None:
    payload = _profile()
    payload["device"] = "NVIDIA GeForce RTX 4090"
    path = tmp_path / "wrong-device.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert schedules_for_linear(
        device="cuda:0",
        in_features=8192,
        out_features=2048,
        group_size=32,
        torch_module=_FakeTorch,
        profile_path=path,
    ) == {}

    path.write_text("not json", encoding="utf-8")
    assert schedules_for_linear(
        device="cuda:0",
        in_features=8192,
        out_features=2048,
        group_size=32,
        torch_module=_FakeTorch,
        profile_path=path,
    ) == {}
