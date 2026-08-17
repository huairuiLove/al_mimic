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
    print("round\tlabeled\ttraining_sec\tcomal_sec\ttotal_sec")
    for record in state["records"]:
        timing = record["timing"]
        training_seconds = timing.get("joint_training_sec", timing.get("classifier_training_sec", 0.0))
        comal_seconds = timing.get("comal_training_sec", 0.0)
        print(
            f"{record['round_index']}\t{record.get('labeled_count', record.get('labeled_before_query', 0))}\t"
            f"{training_seconds:.3f}\t{comal_seconds:.3f}\t"
            f"{timing['round_total_sec']:.3f}"
        )


if __name__ == "__main__":
    main()
