#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mimic_comal.visualization import compare_experiments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", action="append", nargs=2, metavar=("DIR", "NAME"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare_experiments(
        [value[0] for value in args.experiment],
        [value[1] for value in args.experiment],
        args.output,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
