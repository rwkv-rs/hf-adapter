"""Shared, dependency-light infrastructure for benchmark entry points."""

from .environment import collect_environment
from .model_loader import load_causal_lm, load_tokenizer
from .paths import BENCH_ROOT, DEFAULT_RESULTS_PATH, REPO_ROOT, default_run_directory
from .results import append_jsonl, write_json
from .timing import measure_ms, median, synchronize, temporary_environ

__all__ = [
    "BENCH_ROOT",
    "DEFAULT_RESULTS_PATH",
    "REPO_ROOT",
    "append_jsonl",
    "collect_environment",
    "default_run_directory",
    "load_causal_lm",
    "load_tokenizer",
    "measure_ms",
    "median",
    "synchronize",
    "temporary_environ",
    "write_json",
]
