#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def iter_markdown_files() -> list[Path]:
    return sorted(
        p
        for p in ROOT.rglob("*.md")
        if ".git" not in p.parts and not any(part.startswith(".") and part != "." for part in p.parts)
    )


def strip_code_fences(text: str) -> str:
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```") or line.lstrip().startswith("````"):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return "\n".join(lines)


def is_external(target: str) -> bool:
    return (
        "://" in target
        or target.startswith("mailto:")
        or target.startswith("#")
        or target.startswith("app://")
    )


def resolve_target(source: Path, target: str) -> Path | None:
    target = target.split("#", 1)[0]
    if not target or is_external(target):
        return None
    target = unquote(target)
    if target.startswith("/"):
        return ROOT / target.lstrip("/")
    return (source.parent / target).resolve()


def main() -> int:
    missing: list[str] = []
    for md in iter_markdown_files():
        text = strip_code_fences(md.read_text(encoding="utf-8"))
        for match in LINK_RE.finditer(text):
            raw = match.group(1)
            target = resolve_target(md, raw)
            if target is None:
                continue
            try:
                target.relative_to(ROOT)
            except ValueError:
                missing.append(f"{md.relative_to(ROOT)} -> {raw} escapes repository")
                continue
            if not target.exists():
                missing.append(f"{md.relative_to(ROOT)} -> {raw} ({target.relative_to(ROOT)})")
    if missing:
        raise AssertionError("Broken local markdown links:\n" + "\n".join(missing))
    print("MARKDOWN LINKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
