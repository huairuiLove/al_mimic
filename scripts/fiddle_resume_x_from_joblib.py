#!/usr/bin/env python
"""Resume FIDDLE X feature mapping from df_time_series.joblib (skip transform)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sparse import save_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        required=True,
        help="FIDDLE feature dir containing df_time_series.joblib",
    )
    parser.add_argument("--T", type=float, default=48.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--theta_2", type=float, default=0.001)
    parser.add_argument("--N", type=int, default=None)
    args_cli = parser.parse_args()

    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "third_party", "FIDDLE-experiments", "FIDDLE")
    )
    sys.path.insert(0, root)
    import FIDDLE.steps as FIDDLE_steps
    import FIDDLE.config as FIDDLE_config

    output_dir = args_cli.output_dir
    if not output_dir.endswith("/"):
        output_dir += "/"

    joblib_df = output_dir + "df_time_series.joblib"
    joblib_dtypes = output_dir + "dtypes_time_series.joblib"
    if not (os.path.exists(joblib_df) and os.path.exists(joblib_dtypes)):
        raise SystemExit(f"Missing joblib intermediates under {output_dir}")

    print("Loading", joblib_df, flush=True)
    t0 = time.time()
    df_time_series = joblib.load(joblib_df)
    dtypes_time_series = joblib.load(joblib_dtypes)
    print("Loaded", df_time_series.shape, f"in {time.time()-t0:.1f}s", flush=True)

    # Minimal args namespace matching FIDDLE.run
    class Args:
        pass

    args = Args()
    args.output_dir = output_dir
    args.T = args_cli.T
    args.dt = args_cli.dt
    args.theta_2 = args_cli.theta_2
    args.L = int(np.floor(args.T / args.dt))
    args.N = args_cli.N or (len(df_time_series) // args.L)
    args.discretize = True
    args.use_ordinal_encoding = False
    args.X_discretization_bins = None
    args.parallel = False
    args.n_jobs = 1
    args.postfilter = True

    dir_path = output_dir
    print(f"N={args.N} L={args.L} mapping features...", flush=True)
    X_all, X_all_feature_names, X_discretization_bins = FIDDLE_steps.map_time_series_features(
        df_time_series, dtypes_time_series, args
    )
    save_npz(dir_path + "X_all.npz", X_all)
    json.dump(list(X_all_feature_names), open(dir_path + "X_all.feature_names.json", "w"), sort_keys=True)
    json.dump(X_discretization_bins, open(dir_path + "X_all.discretization.json", "w"))
    print("X_all", X_all.shape, "density", getattr(X_all, "density", None), flush=True)

    print("Post-filter...", flush=True)
    X, X_feature_names, X_feature_aliases = FIDDLE_steps.post_filter_time_series(
        X_all, X_all_feature_names, args.theta_2, args
    )
    save_npz(dir_path + "X.npz", X)
    json.dump(list(X_feature_names), open(dir_path + "X.feature_names.json", "w"), sort_keys=True)
    json.dump(X_feature_aliases, open(dir_path + "X.feature_aliases.json", "w"), sort_keys=True)
    print("X", X.shape, "density", X.density, flush=True)
    print("FIDDLE_X_DONE", flush=True)


if __name__ == "__main__":
    main()
