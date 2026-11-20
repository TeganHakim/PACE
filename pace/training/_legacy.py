"""Execute original training scripts without changing their implementation."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]


def run_legacy(relative_path: str) -> None:
    script = ROOT / relative_path
    if not script.is_file():
        raise FileNotFoundError(f"Original training script missing: {script}")
    # Original scripts rely on sibling imports and repository-relative paths.
    # Match the script execution environment while retaining CLI arguments.
    for directory in (ROOT, script.parent):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    old_argv0 = sys.argv[0]
    try:
        sys.argv[0] = str(script)
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv[0] = old_argv0
