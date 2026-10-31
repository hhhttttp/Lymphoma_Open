"""Validate saved CP/QP/PLAF models without clinical inputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from external_validation import run_external_validation

SEED = 27


def main():
    run_external_validation("none", seed=SEED)


if __name__ == "__main__":
    main()
