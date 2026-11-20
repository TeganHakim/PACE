"""Idempotent Phase 7 documentation index updater; preserves README content."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
START="<!-- PACE-REFACTOR-DOCS-START -->"
END="<!-- PACE-REFACTOR-DOCS-END -->"
BLOCK="""<!-- PACE-REFACTOR-DOCS-START -->
## Refactor documentation

- [Architecture and model components](docs/refactor/ARCHITECTURE.md)
- [Data, training, evaluation and reproducibility](docs/refactor/WORKFLOWS.md)
- [Mac MPS and Jetson CUDA environment guidance](docs/refactor/ENVIRONMENTS.md)

The refactor preserves pretrained model behavior; Jetson CUDA compatibility requires validation on the physical device.
<!-- PACE-REFACTOR-DOCS-END -->"""
def update(text):
    if (START in text)!=(END in text):
        raise ValueError("README has only one Phase 7 marker; refusing to edit")
    if START in text:
        before,rest=text.split(START,1)
        _,after=rest.split(END,1)
        return before.rstrip()+"\n\n"+BLOCK+"\n"+after.lstrip("\n")
    return text.rstrip()+"\n\n"+BLOCK+"\n"
def main():
    path=ROOT/"README.md"
    original=path.read_text()
    updated=update(original)
    if original==updated:
        print("README documentation index already up to date")
        return
    path.write_text(updated)
    print("Updated README index (existing content preserved):",path)
if __name__=="__main__":main()
