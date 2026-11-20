"""Load legacy modules without modifying PACE's existing import paths."""
from pathlib import Path
import importlib.util
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def load_legacy():
    def load(relative_path):
        path = ROOT / relative_path
        if not path.exists():
            pytest.fail(f"Expected original source file not found: {path}")
        spec = importlib.util.spec_from_file_location(f"pace_legacy_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return load
