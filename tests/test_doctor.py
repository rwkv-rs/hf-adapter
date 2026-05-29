import json
from types import SimpleNamespace

from rwkv7_hf import doctor


class _FakeCuda:
    def __init__(
        self,
        *,
        available: bool,
        devices: list[tuple[str, tuple[int, int], int]],
        arch_list: list[str] | None = None,
    ):
        self._available = available
        self._devices = devices
        self._arch_list = arch_list or []

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return len(self._devices)

    def current_device(self) -> int:
        return 0

    def get_arch_list(self) -> list[str]:
        return self._arch_list

    def get_device_name(self, index: int) -> str:
        return self._devices[index][0]

    def get_device_capability(self, index: int) -> tuple[int, int]:
        return self._devices[index][1]

    def get_device_properties(self, index: int) -> SimpleNamespace:
        return SimpleNamespace(total_memory=self._devices[index][2])


class _FakeMPS:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeTorch:
    __version__ = "2.12.0"

    def __init__(self, cuda: _FakeCuda):
        self.cuda = cuda
        self.version = SimpleNamespace(cuda="12.8", hip=None)
        self.backends = SimpleNamespace(mps=_FakeMPS())

    @staticmethod
    def device(value):
        if isinstance(value, int):
            return SimpleNamespace(index=value)
        text = str(value)
        index = int(text.split(":", 1)[1]) if ":" in text else None
        return SimpleNamespace(index=index)


def test_collect_diagnostics_cpu_is_ready_without_gpu_warning() -> None:
    torch = _FakeTorch(_FakeCuda(available=False, devices=[]))
    report = doctor.collect_diagnostics(torch_module=torch)

    assert report["status"] == "ready"
    assert report["accelerators"]["cuda_available"] is False
    assert report["devices"][0]["profile"]["family"] == "cpu_or_unknown"
    assert not any(
        "exact validated hardware" in warning for warning in report["warnings"]
    )
    assert doctor.render_text(report).endswith("RESULT: READY")


def test_collect_diagnostics_reports_each_cuda_policy() -> None:
    torch = _FakeTorch(
        _FakeCuda(
            available=True,
            devices=[
                ("NVIDIA GeForce RTX 5090", (12, 0), 32 * 1024**3),
                ("NVIDIA GeForce RTX 4090", (8, 9), 24 * 1024**3),
            ],
            arch_list=["sm_86", "sm_120"],
        )
    )
    report = doctor.collect_diagnostics(torch_module=torch)

    assert report["accelerators"]["cuda_device_count"] == 2
    assert [item["profile"]["family"] for item in report["devices"]] == [
        "blackwell",
        "ada",
    ]
    assert report["devices"][0]["memory_bytes"] == 32 * 1024**3
    assert report["devices"][1]["profile"]["device_index"] == 1
    assert all(item["torch_binary_compatible"] for item in report["devices"])
    assert "fused_recurrent_output" in report["devices"][0]["policy_defaults"]["decode"]


def test_incompatible_torch_cuda_binary_is_not_ready() -> None:
    torch = _FakeTorch(
        _FakeCuda(
            available=True,
            devices=[("Tesla V100-PCIE-32GB", (7, 0), 32 * 1024**3)],
            arch_list=["sm_75", "sm_80", "sm_90", "sm_120"],
        )
    )
    report = doctor.collect_diagnostics(torch_module=torch)

    assert report["status"] == "not_ready"
    assert report["devices"][0]["torch_binary_compatible"] is False
    assert any("matching PyTorch build" in warning for warning in report["warnings"])
    assert doctor.render_text(report).endswith("RESULT: FAIL")


def test_json_cli_writes_the_same_report(monkeypatch, tmp_path, capsys) -> None:
    expected = {
        "schema_version": 1,
        "status": "ready",
        "packages": {},
        "accelerators": {},
        "toolchain": {},
        "devices": [],
        "warnings": [],
        "notes": [],
    }
    monkeypatch.setattr(doctor, "collect_diagnostics", lambda **_: expected)
    output = tmp_path / "doctor.json"

    assert doctor.main(["--json", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert json.loads(output.read_text(encoding="utf-8")) == expected
