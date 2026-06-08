# coding=utf-8
"""Unified command-line interface for the RWKV-7 Hugging Face runtime."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Sequence


COMMANDS = {
    "convert": (
        "rwkv7_hf.converter",
        "convert an official RWKV-7 .pth checkpoint to Hugging Face format",
    ),
    "doctor": (
        "rwkv7_hf.doctor",
        "inspect the runtime, accelerator, and selected kernel policy",
    ),
    "kernels": (
        "rwkv7_hf.kernels_cli",
        "inspect, recommend, list, or install an exact kernel wheel",
    ),
    "smoke": (
        "rwkv7_hf.smoke",
        "load a local or Hub model and run a first-generation smoke test",
    ),
}


def _version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if pyproject.is_file():
        in_project = False
        for raw_line in pyproject.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("["):
                in_project = line == "[project]"
            elif in_project and line.startswith("version"):
                return line.split("=", 1)[1].strip().strip('"\'')
    try:
        return metadata.version("rwkv7-hf")
    except metadata.PackageNotFoundError:
        return "unknown"


def render_help() -> str:
    width = max(len(name) for name in COMMANDS)
    commands = "\n".join(
        f"  {name:<{width}}  {description}"
        for name, (_, description) in COMMANDS.items()
    )
    return (
        "usage: rwkv7-hf <command> [options]\n\n"
        "RWKV-7 Hugging Face runtime and checkpoint tools.\n\n"
        f"commands:\n{commands}\n\n"
        "options:\n"
        "  -h, --help     show this help message\n"
        "  -V, --version  show the installed rwkv7-hf version\n\n"
        "Run `rwkv7-hf <command> --help` for command-specific options."
    )


@contextmanager
def _program_name(value: str):
    previous = sys.argv[0]
    sys.argv[0] = value
    try:
        yield
    finally:
        sys.argv[0] = previous


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(render_help())
        return 0
    if args[0] in {"-V", "--version", "version"}:
        print(f"rwkv7-hf {_version()}")
        return 0

    command = args.pop(0)
    entry = COMMANDS.get(command)
    if entry is None:
        print(f"rwkv7-hf: unknown command {command!r}\n", file=sys.stderr)
        print(render_help(), file=sys.stderr)
        return 2

    module = importlib.import_module(entry[0])
    with _program_name(f"rwkv7-hf {command}"):
        if command == "convert":
            return int(module.main(args, prog=f"rwkv7-hf {command}"))
        return int(module.main(args))


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
