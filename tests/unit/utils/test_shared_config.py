from __future__ import annotations

from pathlib import Path

from al_mimic.tasks.mimic_iii.config import resolve_path as resolve_mimic_path
from al_mimic.tasks.registry import get_task
from al_mimic.utils.config import load_inherited_yaml, resolve_config_path

ROOT = Path(__file__).resolve().parents[3]


def test_inherited_yaml_keeps_the_final_experiment_as_path_anchor() -> None:
    path = ROOT / "configs" / "experiments" / "mds_ed" / "diagnoses.yaml"
    config = load_inherited_yaml(path)

    assert config["_config_path"] == str(path.resolve())
    assert config["task"]["family"] == "mds_ed"
    assert config["experiment"]["name"] == "mds_ed_diagnoses_native_temporal_adapter"
    assert resolve_config_path(config, config["dataset"]["release_csv"]) == (
        ROOT / "dataset" / "raw" / "mds-ed-1.0.0" / "mds_ed.csv"
    )


def test_all_task_configs_resolve_paths_without_using_the_current_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    mimic_path = ROOT / "configs" / "experiments" / "mimic_iii" / "random.yaml"
    mimic = get_task("mimic_iii").load_config(mimic_path)
    assert resolve_mimic_path(mimic, mimic["dataset"]["prepared_dir"]) == (
        ROOT / "dataset" / "prepared" / "yang_wu_diagnoses_48h"
    )
    assert resolve_mimic_path(mimic, mimic["experiment"]["output_root"]) == (ROOT / "experiments")

    brset_path = ROOT / "configs" / "experiments" / "brset" / "random.yaml"
    brset_plugin = get_task("brset")
    brset = brset_plugin.load_config(brset_path)
    from al_mimic.tasks.brset.config import resolve_path as resolve_brset_path

    assert resolve_brset_path(brset, brset["dataset"]["root"]) == (ROOT / "dataset" / "raw" / "brset-1.0.2")
    assert resolve_brset_path(brset, brset["experiment"]["output_root"]) == (ROOT / "experiments")

    mds_path = ROOT / "configs" / "experiments" / "mds_ed" / "diagnoses.yaml"
    mds = get_task("mds_ed").load_config(mds_path)
    assert resolve_config_path(mds, mds["dataset"]["prepared_dir"]) == (
        ROOT / "dataset" / "prepared" / "mds_ed"
    )
