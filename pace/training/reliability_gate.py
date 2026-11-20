"""Run the unchanged reliability gate training workflow.

Usage: python -m pace.training.reliability_gate --help
"""
from ._legacy import run_legacy


def main() -> None:
    run_legacy("prototyping/reliability_gate/train.py")


if __name__ == "__main__":
    main()
