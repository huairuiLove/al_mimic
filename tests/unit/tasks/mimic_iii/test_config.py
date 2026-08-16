from __future__ import annotations

from pathlib import Path

import pytest

from al_mimic.tasks.mimic_iii.config import load_config

ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = ROOT / "configs" / "experiments" / "mimic_iii"
SCENARIOS = ("missing_notes", "mid_labels")
STRATEGIES = ("comal", "mm_comal", "modis", "mosaic", "random")


def _scenario_configs() -> list[tuple[str, str, Path]]:
    return [
        (strategy, scenario, CONFIG_DIR / f"{strategy}_{scenario}.yaml")
        for scenario in SCENARIOS
        for strategy in STRATEGIES
        if (CONFIG_DIR / f"{strategy}_{scenario}.yaml").is_file()
    ]


@pytest.mark.parametrize(("strategy", "scenario", "path"), _scenario_configs())
def test_scenario_config_keeps_its_own_strategy(strategy: str, scenario: str, path: Path) -> None:
    """The scenario base inherits a default strategy; the method must still win.

    Getting this backwards silently runs the same strategy under every filename.
    """
    config = load_config(path)
    assert config["active_learning"]["strategy"] == strategy


@pytest.mark.parametrize(("strategy", "scenario", "path"), _scenario_configs())
def test_scenario_config_keeps_its_scenario(strategy: str, scenario: str, path: Path) -> None:
    config = load_config(path)
    assert config["scenario"]["name"].startswith(scenario.split("_")[0])


@pytest.mark.parametrize(("strategy", "scenario", "path"), _scenario_configs())
def test_scenario_config_keeps_method_tuning(strategy: str, scenario: str, path: Path) -> None:
    """Method-specific blocks must survive the merge with the scenario base."""
    config = load_config(path)
    baseline = load_config(CONFIG_DIR / f"{strategy}.yaml")
    for section in ("mosaic", "modis", "comal", "acquisition"):
        if section in baseline:
            assert config.get(section) == baseline[section], section


def test_multiple_parents_merge_in_order(tmp_path: Path) -> None:
    base = ROOT / "configs" / "tasks" / "mimic_iii" / "icd9_diagnoses.yaml"
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    child = tmp_path / "child.yaml"
    first.write_text(f"extends: {base}\nactive_learning:\n  strategy: comal\n", encoding="utf-8")
    second.write_text(f"extends: {base}\nactive_learning:\n  strategy: mosaic\n", encoding="utf-8")
    child.write_text(f"extends:\n  - {first}\n  - {second}\n", encoding="utf-8")

    assert load_config(child)["active_learning"]["strategy"] == "mosaic"


def test_single_parent_string_still_works() -> None:
    config = load_config(CONFIG_DIR / "mosaic.yaml")
    assert config["active_learning"]["strategy"] == "mosaic"
    assert config["mosaic"]["damping"] == 0.1


def test_unknown_protocol_profile_is_rejected(tmp_path: Path) -> None:
    child = tmp_path / "child.yaml"
    child.write_text(
        f"extends: {CONFIG_DIR / 'random.yaml'}\npreprocessing:\n  protocol_profile: does_not_exist\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown preprocessing.protocol_profile"):
        load_config(child)


def test_profile_without_measured_dimensions_is_rejected(tmp_path: Path) -> None:
    """A cohort that has not been built yet must not be runnable."""
    child = tmp_path / "child.yaml"
    child.write_text(
        f"extends: {CONFIG_DIR / 'random.yaml'}\npreprocessing:\n  protocol_profile: yang_wu_diagnoses_12h\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unresolved dimensions"):
        load_config(child)


def test_mistyped_scenario_key_is_rejected(tmp_path: Path) -> None:
    child = tmp_path / "child.yaml"
    child.write_text(
        f"extends: {CONFIG_DIR / 'random.yaml'}\nscenario:\n  missing_notes:\n    ratio: 0.4\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown scenario.missing_notes keys"):
        load_config(child)


def test_mistyped_scenario_section_is_rejected(tmp_path: Path) -> None:
    child = tmp_path / "child.yaml"
    child.write_text(
        f"extends: {CONFIG_DIR / 'random.yaml'}\nscenario:\n  missing_note:\n    rate: 0.4\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown scenario sections"):
        load_config(child)
