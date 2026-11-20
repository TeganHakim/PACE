"""Repair Phase 7 README safely using the committed README as a base."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- PACE-REFACTOR-DOCS-START -->"
END = "<!-- PACE-REFACTOR-DOCS-END -->"
SECTION = """<!-- PACE-REFACTOR-DOCS-START -->
## Refactor documentation

- [Architecture and model components](docs/refactor/ARCHITECTURE.md)
- [Data, training, evaluation and reproducibility](docs/refactor/WORKFLOWS.md)
- [Mac MPS and Jetson CUDA environment guidance](docs/refactor/ENVIRONMENTS.md)

The refactor preserves pretrained model behavior; Jetson CUDA compatibility requires validation on the physical device.
<!-- PACE-REFACTOR-DOCS-END -->"""

def update(text):
    if (START in text) != (END in text):
        raise ValueError("README contains only one Phase 7 marker")
    if START in text:
        before, remainder = text.split(START, 1)
        _, after = remainder.split(END, 1)
        return before.rstrip() + "\n\n" + SECTION + "\n" + after.lstrip("\n")
    return text.rstrip() + "\n\n" + SECTION + "\n"

def main():
    path = ROOT / "README.md"
    if not path.exists():
        raise SystemExit("Missing README.md")
    current = path.read_text()
    try:
        committed = subprocess.check_output(
            ["git", "show", "HEAD:README.md"], cwd=ROOT, text=True
        )
    except subprocess.CalledProcessError:
        raise SystemExit("Could not retrieve committed README; no files changed")
    if not committed.startswith("# PACE"):
        raise SystemExit("Committed README is not the expected PACE project README; no files changed")
    if current.startswith("# PACE"):
        base = current
        print("Current README already begins with PACE; preserving its content")
    elif current.startswith("# Phase 5 final batch"):
        base = committed
        print("Detected accidentally overwritten README; restoring committed project content")
    else:
        raise SystemExit("README has unrecognized content; refusing to overwrite it")
    result = update(base)
    if result == current:
        print("README already correct")
        return
    backup = ROOT / "README.md.phase7-repair.bak"
    if backup.exists():
        raise SystemExit(f"Backup already exists: {backup}; refusing to overwrite it")
    backup.write_text(current)
    path.write_text(result)
    print(f"Saved original current README to {backup}")
    print("Updated README.md with project content and Phase 7 links")

if __name__ == "__main__":
    main()
