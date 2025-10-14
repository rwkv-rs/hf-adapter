import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hf_parallel_download.py"


def load_module():
    spec = importlib.util.spec_from_file_location("hf_parallel_download", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_make_ranges_exact_multiple():
    mod = load_module()
    assert mod.make_ranges(100, 25) == [
        (0, 0, 24),
        (1, 25, 49),
        (2, 50, 74),
        (3, 75, 99),
    ]


def test_make_ranges_tail_chunk():
    mod = load_module()
    assert mod.make_ranges(103, 25)[-1] == (4, 100, 102)


def test_build_resolve_url_quotes_path_parts():
    mod = load_module()
    assert (
        mod.build_resolve_url(
            repo_id="org/model name",
            filename="nested/model shard.safetensors",
            revision="main",
            endpoint="https://huggingface.co/",
        )
        == "https://huggingface.co/org/model%20name/resolve/main/nested/model%20shard.safetensors"
    )


def test_human_bytes():
    mod = load_module()
    assert mod.human_bytes(1024) == "1.0KiB"
    assert mod.human_bytes(1024 * 1024) == "1.0MiB"
