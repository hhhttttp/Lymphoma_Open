"""Validate saved baseline models using development-snapshot clinical fills."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from external_validation import run_external_validation

SEED = 27


def main():
    run_external_validation("imputed", seed=SEED)


if __name__ == "__main__":
    main()
