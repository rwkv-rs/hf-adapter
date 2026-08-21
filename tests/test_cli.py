from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rwkv7_hf import cli
from rwkv7_hf.adapter_manifest import (
    ADAPTER_FILES as PACKAGE_ADAPTER_FILES,
    LEGACY_REMOTE_CODE_FILES as PACKAGE_LEGACY_REMOTE_CODE_FILES,
)
from scripts.adapter_manifest import (
    ADAPTER_FILES as SCRIPT_ADAPTER_FILES,
    LEGACY_REMOTE_CODE_FILES as SCRIPT_LEGACY_REMOTE_CODE_FILES,
)


ROOT = Path(__file__).resolve().parents[1]


def test_unified_cli_help_and_unknown_command(capsys) -> None:
    assert cli.main([]) == 0
    help_text = capsys.readouterr().out
    for command in ("convert", "doctor", "kernels", "smoke"):
        assert command in help_text

    assert cli.main(["not-a-command"]) == 2
    error = capsys.readouterr().err
    assert "unknown command 'not-a-command'" in error


def test_unified_cli_source_version_matches_release(capsys) -> None:
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "rwkv7-hf 0.8.1"


def test_unified_convert_help_uses_subcommand_program_name() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "rwkv7_hf.cli", "convert", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage: rwkv7-hf convert" in proc.stdout
    assert "--adapter-layout {thin,bundled}" in proc.stdout
    assert "--low-memory" in proc.stdout


def test_legacy_converter_wrapper_runs_outside_checkout() -> None:
    script = ROOT / "scripts" / "convert_rwkv7_to_hf.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=Path("/tmp"),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--adapter-layout {thin,bundled}" in proc.stdout


def test_installed_and_source_manifest_lists_stay_synchronized() -> None:
    assert PACKAGE_ADAPTER_FILES == SCRIPT_ADAPTER_FILES
    assert PACKAGE_LEGACY_REMOTE_CODE_FILES == SCRIPT_LEGACY_REMOTE_CODE_FILES
