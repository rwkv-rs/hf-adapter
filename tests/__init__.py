"""Repository test helpers.

Making ``tests`` an explicit package prevents unrelated site-packages named
``tests`` from shadowing cross-test helper imports in full-suite runs.
"""

from pathlib import Path
import sys

# A few executable smoke modules intentionally support direct invocation and
# therefore import sibling helpers by their top-level name.  Preserve that
# contract when a smoke module is imported through ``tests.<module>``.
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
