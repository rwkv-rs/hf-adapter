#!/usr/bin/env python3
"""Create a lightweight local HF directory that loads the pinned FLA model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    excluded = {
        "config.json",
        "configuration_rwkv7.py",
        "modeling_rwkv7.py",
        "cache_rwkv7.py",
        "ops_rwkv7.py",
    }
    for item in source.iterdir():
        if not item.is_file() or item.name in excluded:
            continue
        destination = output / item.name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(item)

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["auto_map"] = {
        "AutoConfig": "configuration_rwkv7_fla.RWKV7Config",
        "AutoModel": "modeling_rwkv7_fla.RWKV7Model",
        "AutoModelForCausalLM": "modeling_rwkv7_fla.RWKV7ForCausalLM",
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "configuration_rwkv7_fla.py").write_text(
        "from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config\n"
        "\n__all__ = [\"RWKV7Config\"]\n",
        encoding="utf-8",
    )
    (output / "modeling_rwkv7_fla.py").write_text(
        "from fla.models.rwkv7.modeling_rwkv7 import "
        "RWKV7ForCausalLM, RWKV7Model\n"
        "\n__all__ = [\"RWKV7Model\", \"RWKV7ForCausalLM\"]\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "rwkv7-fla-lm-eval-wrapper-v1",
        "source": str(source),
        "output": str(output),
        "weight_files": sorted(
            item.name for item in source.glob("*.safetensors") if item.is_file()
        ),
        "note": "Weights/tokenizer are symlinked; only AutoModel routing changes.",
    }
    (output / "fla-wrapper.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
