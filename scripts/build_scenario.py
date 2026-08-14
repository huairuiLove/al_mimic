#!/usr/bin/env python
"""Rebuild a Diagnoses cohort at a new observation window, end to end.

Chains the five stages that turn raw MIMIC-III into a split artifact the active
learning loop can read, skipping any stage whose output already exists so an
interrupted rebuild resumes where it stopped:

    population -> input -> fiddle -> notes -> splits -> profile

The 12h window is the reason this exists. Requiring LOS >= 48h discards 11846 of
23620 MetaVision stays, which is why the current cohort is only 10258; at 12h it
is 16930, and because the time axis is a quarter as long the FIDDLE stage
expands 203160 rows instead of 492384. The cohort grows 1.65x while the most
expensive stage gets cheaper.

Stages read and write tens of gigabytes, so the host is checked up front rather
than letting the OOM killer end a multi-hour run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
FIDDLE_ROOT = ROOT / "third_party/FIDDLE-experiments"
STAGES = ("population", "input", "fiddle", "notes", "splits", "profile")
MINIMUM_MEMORY_GIB = {"input": 96.0, "fiddle": 96.0, "notes": 4.0, "splits": 16.0}


def available_memory_gib() -> float:
    """Honour the cgroup ceiling; /proc/meminfo reports the host, not the container."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw != "max":
            limit = int(raw) / 2**30
            if limit < 2**20:  # ignore the "effectively unlimited" sentinel
                return limit
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.check_call(command, cwd=cwd or ROOT, env=env)


def stage_population(args: argparse.Namespace, task_dir: Path) -> None:
    run(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts/build_diagnoses_population.py"),
            "--mimic-dir",
            str(args.mimic_dir),
            "--data-dir",
            str(args.data_dir),
            "--duration",
            str(args.duration),
        ]
    )


def stage_input(args: argparse.Namespace, task_dir: Path) -> None:
    run(
        [
            sys.executable,
            "-u",
            str(FIDDLE_ROOT / "mimic3_experiments/1_data_extraction/prepare_input.py"),
            "--outcome",
            args.task,
            "--T",
            str(args.duration),
            "--dt",
            str(args.timestep),
        ]
    )


def stage_fiddle(args: argparse.Namespace, task_dir: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(FIDDLE_ROOT / "FIDDLE")
    population = (
        args.data_dir / f"population/{args.task}_{args.duration}h.csv"
    )
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "FIDDLE.run",
            f"--output_dir={task_dir}/",
            f"--data_fname={task_dir}/input_data.p",
            f"--population={population}",
            f"--T={args.duration}",
            f"--dt={args.timestep}",
            "--theta_1=0.001",
            "--theta_2=0.001",
            "--theta_freq=1",
            "--stats_functions",
            "min",
            "max",
            "mean",
        ],
        env=environment,
    )


def stage_notes(args: argparse.Namespace, task_dir: Path) -> None:
    run(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts/extract_notes.py"),
            "--mimic-dir",
            str(args.mimic_dir),
            "--data-dir",
            str(args.data_dir),
            "--task",
            args.task,
            "--duration",
            str(args.duration),
            "--timestep",
            str(args.timestep),
        ]
    )


def stage_splits(args: argparse.Namespace, task_dir: Path) -> None:
    run(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts/build_splits.py"),
            "--data-dir",
            str(args.data_dir),
            "--task",
            args.task,
            "--duration",
            str(args.duration),
            "--timestep",
            str(args.timestep),
        ]
    )


def stage_profile(args: argparse.Namespace, task_dir: Path) -> None:
    """Record the built cohort's dimensions so the protocol validator can lock them."""
    import h5py
    import numpy as np

    with h5py.File(task_dir / "splits.hdf5", "r") as handle:
        root = handle["with_notes"]
        total = sum(int(root[split]["label"].shape[0]) for split in ("train", "val", "test"))
        train = root["train"]
        measured = {
            "observation_hours": int(args.duration),
            "timestep_hours": int(args.timestep),
            "max_note_tokens": int(train["input_ids"].shape[1]),
            "expected_total_samples": total,
            "expected_label_count": int(train["label"].shape[1]),
            "time_invariant_dim": int(train["s"].shape[1]),
            "time_series_dim": int(train["X"].shape[2]),
            "note_protocol": f"latest_per_category_description_within_{int(args.duration)}h",
        }

    path = ROOT / "configs/protocol_profiles.yaml"
    profiles = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    name = args.profile or f"yang_wu_diagnoses_{int(args.duration)}h"
    existing = profiles.get(name) or {}
    conflicts = {
        key: (existing[key], value)
        for key, value in measured.items()
        if existing.get(key) is not None and existing[key] != value
    }
    if conflicts and not args.force_profile:
        raise SystemExit(
            f"profile {name!r} already pins different dimensions {conflicts}; "
            "pass --force-profile only if the cohort was deliberately rebuilt"
        )
    profiles[name] = measured
    path.write_text(yaml.safe_dump(profiles, sort_keys=False), encoding="utf-8")
    print(f"registered profile {name!r}:", flush=True)
    for key, value in measured.items():
        print(f"    {key}: {value}", flush=True)

    write_base_config(args, task_dir, name, measured)


def write_base_config(
    args: argparse.Namespace, task_dir: Path, profile: str, measured: dict[str, object]
) -> Path:
    """Emit the base config for the rebuilt cohort.

    The protocol validator compares every dimension in a config against the
    profile, so transcribing them by hand is a guaranteed source of failed runs.
    """
    template = yaml.safe_load(
        (ROOT / "configs/mimic_a800_144c.yaml").read_text(encoding="utf-8")
    )
    window = int(args.duration)
    template["experiment"]["name"] = f"mimic_iii_yang_wu_bertencoder_{window}h"
    template["dataset"]["prepared_dir"] = f"prepared/yang_wu_diagnoses_{window}h"
    template["dataset"]["split_hdf5"] = str((task_dir / "splits.hdf5").relative_to(ROOT))
    template["preprocessing"]["protocol_profile"] = profile
    for key, value in measured.items():
        template["preprocessing"][key] = value
    template["model"]["output_size"] = measured["expected_label_count"]

    output = ROOT / f"configs/mimic_{window}h_base.yaml"
    header = (
        f"# Base config for the {window}h cohort, generated by scripts/build_scenario.py.\n"
        f"# Dimensions are copied from protocol profile {profile!r} so the validator\n"
        "# cannot disagree with the artifact that was actually built.\n\n"
    )
    output.write_text(header + yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    print(f"wrote {output}", flush=True)
    return output


STAGE_OUTPUTS = {
    "population": lambda a, d: a.data_dir / f"population/{a.task}_{a.duration}h.csv",
    "input": lambda a, d: d / "input_data.p",
    "fiddle": lambda a, d: d / "X.npz",
    "notes": lambda a, d: d / "notes.hdf5",
    "splits": lambda a, d: d / "splits.hdf5",
    "profile": lambda a, d: None,
}
STAGE_RUNNERS = {
    "population": stage_population,
    "input": stage_input,
    "fiddle": stage_fiddle,
    "notes": stage_notes,
    "splits": stage_splits,
    "profile": stage_profile,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-dir", type=Path, default=ROOT / "mimic-iii-clinical-database-1.4")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/fiddle_processed")
    parser.add_argument("--task", default="Diagnoses")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--profile", default=None, help="protocol profile name to register")
    parser.add_argument("--force-profile", action="store_true")
    parser.add_argument("--only", nargs="+", choices=STAGES, default=None)
    parser.add_argument("--skip", nargs="+", choices=STAGES, default=())
    parser.add_argument("--rebuild", action="store_true", help="ignore existing stage outputs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ignore-memory", action="store_true")
    args = parser.parse_args()

    task_dir = (
        args.data_dir / f"features/outcome={args.task},T={args.duration},dt={args.timestep}"
    )
    task_dir.mkdir(parents=True, exist_ok=True)
    memory = available_memory_gib()
    print(f"target: {task_dir}\navailable memory: {memory:.1f} GiB\n", flush=True)

    planned = [
        stage
        for stage in (args.only or STAGES)
        if stage not in args.skip
    ]
    for stage in planned:
        marker = STAGE_OUTPUTS[stage](args, task_dir)
        if marker is not None and marker.exists() and not args.rebuild:
            print(f"[skip] {stage}: {marker} already exists", flush=True)
            continue
        needed = MINIMUM_MEMORY_GIB.get(stage)
        if needed and memory < needed and not args.ignore_memory:
            raise SystemExit(
                f"[stop] {stage} needs about {needed:.0f} GiB but only {memory:.1f} GiB is "
                "available. This host looks like a no-GPU / low-memory instance; switch back "
                "to the full instance, or pass --ignore-memory to try anyway."
            )
        print(f"\n=== {stage} ===", flush=True)
        if args.dry_run:
            print("[dry-run] would run this stage", flush=True)
            continue
        STAGE_RUNNERS[stage](args, task_dir)

    print("\nBUILD_SCENARIO_DONE", flush=True)


if __name__ == "__main__":
    main()
