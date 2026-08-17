#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from al_mimic.tasks.mimic_iii.visualization import compare_experiments


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
