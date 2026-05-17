#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
CURRENT_ARTIFACTS = ROOT / "bench" / "CURRENT_ARTIFACTS.json"
LINK_RE = re.compile(
    r'!?\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+"[^"]*")?\s*\)'
)
HEADING_RE = re.compile(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
FENCE_OPEN_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def frozen_artifact_roots() -> set[Path]:
    manifest = json.loads(CURRENT_ARTIFACTS.read_text(encoding="utf-8"))
    return {
        (ROOT / "bench" / artifact["path"]).resolve()
        for artifact in manifest["artifacts"]
    }


def iter_markdown_files() -> list[Path]:
    frozen_roots = frozen_artifact_roots()
    return sorted(
        p
        for p in ROOT.rglob("*.md")
        if ".git" not in p.parts and not any(part.startswith(".") and part != "." for part in p.parts)
        and not any(
            p.resolve().is_relative_to(root) and p.name != "README.md"
            for root in frozen_roots
        )
    )


def strip_code_fences(text: str) -> tuple[str, bool]:
    lines: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        if fence is not None:
            stripped = line.lstrip()
            if re.fullmatch(rf"{re.escape(fence[0])}{{{len(fence)},}}\s*", stripped):
                fence = None
            continue
        match = FENCE_OPEN_RE.match(line)
        if match:
            fence = match.group(1)
            continue
        lines.append(line)
    return "\n".join(lines), fence is None


def is_external(target: str) -> bool:
    return (
        "://" in target
        or target.startswith("mailto:")
        or target.startswith("app://")
    )


def resolve_target(source: Path, target: str) -> Path | None:
    if is_external(target):
        return None
    path_part = unquote(target.split("#", 1)[0])
    if not path_part:
        return source
    if path_part.startswith("/"):
        return (ROOT / path_part.lstrip("/")).resolve()
    return (source.parent / path_part).resolve()


def heading_anchors(text: str) -> set[str]:
    text, _ = strip_code_fences(text)
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for match in HEADING_RE.finditer(text):
        heading = html.unescape(match.group(1))
        heading = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading)
        heading = re.sub(r"<[^>]+>", "", heading)
        heading = re.sub(r"[`*_~]", "", heading).strip().lower()
        slug = "".join(
            character
            for character in heading
            if character.isalnum() or character in {" ", "-", "_"}
        )
        slug = re.sub(r"\s+", "-", slug)
        duplicate = counts.get(slug, 0)
        counts[slug] = duplicate + 1
        anchors.add(slug if duplicate == 0 else f"{slug}-{duplicate}")
    return anchors


def repository_entries() -> dict[str, str]:
    entries: dict[str, str] = {}
    for path in ROOT.rglob("*"):
        if ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        entries[relative.casefold()] = relative
    return entries


def main() -> int:
    missing: list[str] = []
    bad_case: list[str] = []
    bad_anchors: list[str] = []
    unclosed_fences: list[str] = []
    entries = repository_entries()
    anchors_by_path: dict[Path, set[str]] = {}
    for md in iter_markdown_files():
        text, fences_closed = strip_code_fences(md.read_text(encoding="utf-8"))
        if not fences_closed:
            unclosed_fences.append(str(md.relative_to(ROOT)))
        for match in LINK_RE.finditer(text):
            raw = match.group(1) or match.group(2)
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
                continue
            relative = target.relative_to(ROOT).as_posix()
            actual = entries.get(relative.casefold())
            if actual is not None and actual != relative:
                bad_case.append(f"{md.relative_to(ROOT)} -> {raw} (actual: {actual})")

            fragment = unquote(raw.split("#", 1)[1]) if "#" in raw else ""
            if fragment and target.suffix.lower() == ".md":
                anchors = anchors_by_path.setdefault(
                    target, heading_anchors(target.read_text(encoding="utf-8"))
                )
                if fragment.lower() not in anchors:
                    bad_anchors.append(f"{md.relative_to(ROOT)} -> {raw}")
    if missing:
        raise AssertionError("Broken local markdown links:\n" + "\n".join(missing))
    if bad_case:
        raise AssertionError("Case-mismatched markdown paths:\n" + "\n".join(bad_case))
    if bad_anchors:
        raise AssertionError("Broken local markdown anchors:\n" + "\n".join(bad_anchors))
    if unclosed_fences:
        raise AssertionError("Unclosed markdown code fences:\n" + "\n".join(unclosed_fences))
    print("MARKDOWN LINKS, IMAGES, ANCHORS, CASE, AND FENCES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
