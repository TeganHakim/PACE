"""Run the unchanged quality estimator training workflow.

Usage: python -m pace.training.quality_estimator --help
"""
from ._legacy import run_legacy


def main() -> None:
    run_legacy("prototyping/quality_estimator/train.py")


if __name__ == "__main__":
    main()
