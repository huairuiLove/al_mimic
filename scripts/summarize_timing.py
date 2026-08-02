#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    args = parser.parse_args()
    state = json.loads((Path(args.experiment) / "active_state.json").read_text())
    print("round\tlabeled\tclassifier_sec\tcomal_sec\ttotal_sec")
    for record in state["records"]:
        timing = record["timing"]
        print(
            f"{record['round_index']}\t{record['labeled_before_query']}\t"
            f"{timing['classifier_training_sec']:.3f}\t{timing['comal_training_sec']:.3f}\t"
            f"{timing['round_total_sec']:.3f}"
        )


if __name__ == "__main__":
    main()
