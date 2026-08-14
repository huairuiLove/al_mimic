#!/usr/bin/env python
"""Decide whether a scenario can separate acquisition strategies at all.

Running six arms on a scenario that cannot resolve them wastes the GPU time and
produces a table of indistinguishable numbers. Two arms answer the question
first: a random-acquisition arm and a full-pool ceiling. If the ceiling is not
clearly above random at the target budget, no acquisition strategy can be
either, because the ceiling is what perfect acquisition converges to.

The comparison is against test-set noise, estimated by resampling test visits
and recomputing both arms on the same resample, so the reported margin is the
paired difference a second seed would have to beat.

Also reports each arm's normalised gain, (arm - random) / (ceiling - random),
which is comparable across scenarios in a way that raw recall is not.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class Arm:
    label: str
    labels: np.ndarray
    probabilities: np.ndarray
    recall: float


def recall_at_k(labels: np.ndarray, probabilities: np.ndarray, k: int) -> float:
    evaluable = labels.sum(axis=1) > 0
    truth = labels[evaluable]
    scores = probabilities[evaluable]
    top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    hits = np.take_along_axis(truth, top, axis=1).sum(axis=1)
    return float((hits / truth.sum(axis=1)).mean())


def load_arm(directory: Path, label: str, k: int) -> Arm:
    payload = np.load(directory / "final_predictions.npz")
    labels = payload["test_labels"]
    probabilities = payload["test_probabilities"]
    return Arm(label, labels, probabilities, recall_at_k(labels, probabilities, k))


def paired_interval(
    left: Arm, right: Arm, k: int, draws: int, seed: int
) -> tuple[float, float, float]:
    """Bootstrap the difference right - left over resampled test visits."""
    if left.labels.shape != right.labels.shape:
        raise SystemExit(
            f"{left.label} and {right.label} were evaluated on different test tensors "
            f"{left.labels.shape} vs {right.labels.shape}; they are not comparable"
        )
    rng = np.random.default_rng(seed)
    evaluable = np.flatnonzero(left.labels.sum(axis=1) > 0)
    differences = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sample = rng.choice(evaluable, size=evaluable.size, replace=True)
        differences[draw] = recall_at_k(
            right.labels[sample], right.probabilities[sample], k
        ) - recall_at_k(left.labels[sample], left.probabilities[sample], k)
    low, high = np.percentile(differences, [2.5, 97.5])
    return float(differences.mean()), float(low), float(high)


def budget_curve(directory: Path, k: int) -> list[tuple[int, float, float]]:
    state_path = directory / "active_state.json"
    if not state_path.is_file():
        return []
    records = json.loads(state_path.read_text(encoding="utf-8"))["records"]
    metric = f"recall_at_{k}"
    return [
        (
            int(record["labeled_count"]),
            float(record["labeled_fraction_of_train"]),
            float(record["test_metrics"][metric]),
        )
        for record in records
        if metric in record.get("test_metrics", {})
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=Path, required=True, help="random-acquisition arm dir")
    parser.add_argument("--ceiling", type=Path, required=True, help="full-pool ceiling dir")
    parser.add_argument(
        "--arm",
        nargs=2,
        action="append",
        metavar=("DIR", "LABEL"),
        default=[],
        help="additional arm to score against the random baseline",
    )
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--margin",
        type=float,
        default=3.0,
        help="required ratio of achievable range to the noise half-width",
    )
    args = parser.parse_args()

    baseline = load_arm(args.random, "Random", args.k)
    ceiling = load_arm(args.ceiling, "Ceiling", args.k)
    mean, low, high = paired_interval(baseline, ceiling, args.k, args.draws, args.seed)
    half_width = (high - low) / 2.0
    achievable = ceiling.recall - baseline.recall

    print(f"metric: recall@{args.k}   bootstrap draws: {args.draws}\n")
    print(f"  random   {baseline.recall:.4f}")
    print(f"  ceiling  {ceiling.recall:.4f}")
    print(f"  achievable range (ceiling - random)  {achievable:+.4f}")
    print(f"  paired 95% CI of that difference     [{low:+.4f}, {high:+.4f}]")
    print(f"  noise half-width                     {half_width:.4f}")

    ratio = achievable / half_width if half_width > 0 else float("inf")
    print(f"  range / noise                        {ratio:.1f}x (need >= {args.margin:.0f}x)\n")
    if low <= 0.0:
        verdict = 3
        print("VERDICT: no resolution. Full supervision is not measurably better than")
        print("random acquisition here, so no strategy can be either. Change the scenario.")
    elif ratio < args.margin:
        verdict = 2
        print(f"VERDICT: marginal. Any strategy difference would sit inside {ratio:.1f}x the")
        print("noise; expect an inconclusive table. Strengthen the scenario before running.")
    else:
        verdict = 0
        print("VERDICT: resolvable. There is room above random for acquisition to show an")
        print("effect; running the full method set is justified.")

    curve = budget_curve(args.random, args.k)
    if curve:
        # Progress is measured from the smallest budget towards the ceiling, so
        # a scenario that saturates early is visible as an early jump to 100%.
        start = curve[0][2]
        span = ceiling.recall - start
        print(f"\nrandom arm by budget (start {start:.4f} -> ceiling {ceiling.recall:.4f}):")
        for count, fraction, value in curve:
            closed = (value - start) / span if span > 0 else float("nan")
            print(
                f"  {count:6d} labels ({100 * fraction:4.1f}%)  recall@{args.k}={value:.4f}"
                f"   {100 * closed:5.1f}% of the gap to the ceiling closed"
            )

    if args.arm:
        print(f"\nnormalised gain over random, (arm - random) / {achievable:+.4f}:")
        for directory, label in args.arm:
            arm = load_arm(Path(directory), label, args.k)
            gain, arm_low, arm_high = paired_interval(baseline, arm, args.k, args.draws, args.seed)
            significant = "yes" if arm_low > 0.0 or arm_high < 0.0 else "no"
            normalised = gain / achievable if achievable > 0 else float("nan")
            print(
                f"  {label:16s} recall={arm.recall:.4f}  gain={gain:+.4f} "
                f"[{arm_low:+.4f}, {arm_high:+.4f}]  normalised={normalised:+.1%}  "
                f"beats noise: {significant}"
            )

    # Exit code lets a runner gate the expensive method sweep on the verdict.
    raise SystemExit(verdict)


if __name__ == "__main__":
    main()
