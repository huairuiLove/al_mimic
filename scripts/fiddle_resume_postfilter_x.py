#!/usr/bin/env python
"""Resume FIDDLE post-filter from X_all.npz (skip discretize)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sparse import load_npz, save_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--T", type=float, default=48.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--theta_2", type=float, default=0.001)
    parser.add_argument("--N", type=int, default=10258)
    args_cli = parser.parse_args()

    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "third_party", "FIDDLE-experiments", "FIDDLE")
    )
    sys.path.insert(0, root)
    import FIDDLE.steps as FIDDLE_steps

    output_dir = args_cli.output_dir
    if not output_dir.endswith("/"):
        output_dir += "/"

    x_all_path = output_dir + "X_all.npz"
    names_path = output_dir + "X_all.feature_names.json"
    if not (os.path.exists(x_all_path) and os.path.exists(names_path)):
        raise SystemExit(f"Missing X_all artifacts under {output_dir}")

    print("Loading", x_all_path, flush=True)
    t0 = time.time()
    X_all = load_npz(x_all_path)
    with open(names_path) as f:
        X_all_feature_names = np.asarray(json.load(f))
    print("Loaded", X_all.shape, f"in {time.time()-t0:.1f}s", flush=True)

    class Args:
        pass

    args = Args()
    args.T = args_cli.T
    args.dt = args_cli.dt
    args.L = int(np.floor(args.T / args.dt))
    args.N = args_cli.N
    args.theta_2 = args_cli.theta_2

    print("Post-filter...", flush=True)
    X, X_feature_names, X_feature_aliases = FIDDLE_steps.post_filter_time_series(
        X_all, X_all_feature_names, args.theta_2, args
    )
    save_npz(output_dir + "X.npz", X)
    with open(output_dir + "X.feature_names.json", "w") as f:
        json.dump(list(X_feature_names), f, sort_keys=True)
    with open(output_dir + "X.feature_aliases.json", "w") as f:
        json.dump(X_feature_aliases, f, sort_keys=True)
    print("X", X.shape, "density", X.density, flush=True)
    print("FIDDLE_X_DONE", flush=True)


if __name__ == "__main__":
    main()
