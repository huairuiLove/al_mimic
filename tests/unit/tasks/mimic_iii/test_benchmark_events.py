from __future__ import annotations

import numpy as np
import pandas as pd

from al_mimic.tasks.mimic_iii.preprocessing.clean_events import (
    clean_crr,
    clean_dbp,
    clean_events,
    clean_fio2,
    clean_height,
    clean_lab,
    clean_o2sat,
    clean_sbp,
    clean_temperature,
    clean_weight,
    load_itemid_map,
)
from al_mimic.tasks.mimic_iii.preprocessing.discretize import (
    BenchmarkDiscretizer,
    BenchmarkNormalizer,
    iter_stay_events,
    load_layout,
)


def _series(values: list[object]) -> pd.Series:
    return pd.Series(values, dtype=object)


def test_sbp_dbp_split_slash_strings() -> None:
    assert clean_sbp(_series(["120/80"])).iloc[0] == 120.0
    assert clean_dbp(_series(["120/80"])).iloc[0] == 80.0
    assert clean_sbp(_series(["130"])).iloc[0] == 130.0
    assert np.isnan(clean_sbp(_series(["garbage"])).iloc[0])


def test_crr_maps_wording_to_binary() -> None:
    assert clean_crr(_series(["Brisk"])).iloc[0] == 0.0
    assert clean_crr(_series(["Normal <3 secs"])).iloc[0] == 0.0
    assert clean_crr(_series(["Delayed"])).iloc[0] == 1.0
    assert clean_crr(_series(["Abnormal >3 secs"])).iloc[0] == 1.0
    assert np.isnan(clean_crr(_series(["Other"])).iloc[0])


def test_fio2_rescales_only_unitless_percent_values() -> None:
    values = _series(["40", "0.4", "40"])
    uom = _series(["%", "%", "torr"])
    cleaned = clean_fio2(values, uom)
    assert cleaned.tolist() == [0.4, 0.4, 40.0]


def test_temperature_converts_fahrenheit() -> None:
    cleaned = clean_temperature(
        _series(["98.6", "37.0", "85.0"]),
        _series(["F", "C", "C"]),
        _series(["", "", ""]),
    )
    assert np.isclose(cleaned.iloc[0], 37.0)
    assert np.isclose(cleaned.iloc[1], 37.0)
    assert np.isclose(cleaned.iloc[2], (85.0 - 32.0) * 5.0 / 9.0)


def test_weight_converts_ounces_and_pounds() -> None:
    cleaned = clean_weight(_series(["160", "32"]), _series(["lb", "oz"]), _series(["", ""]))
    assert np.isclose(cleaned.iloc[0], 160.0 * 0.453592)
    assert np.isclose(cleaned.iloc[1], 2.0 * 0.453592)


def test_height_converts_inches() -> None:
    cleaned = clean_height(_series(["70"]), _series(["in"]), _series([""]))
    assert cleaned.iloc[0] == round(70 * 2.54)


def test_o2sat_rescales_fractional_values() -> None:
    cleaned = clean_o2sat(_series(["0.97", "97", "ERROR"]))
    assert cleaned.tolist()[:2] == [97.0, 97.0]
    assert np.isnan(cleaned.iloc[2])


def test_lab_drops_non_numeric_values() -> None:
    cleaned = clean_lab(_series(["128", "ERROR"]))
    assert cleaned.iloc[0] == 128.0
    assert np.isnan(cleaned.iloc[1])


def test_clean_events_maps_and_drops() -> None:
    var_map = load_itemid_map()
    target = var_map[var_map["VARIABLE"] == "Glucose"].iloc[0]
    other = var_map[var_map["VARIABLE"] == "Heart Rate"].iloc[0]
    chunk = pd.DataFrame(
        {
            "SUBJECT_ID": [1, 1, 1],
            "ICUSTAY_ID": [10, 10, 10],
            "CHARTTIME": ["2101-01-01"] * 3,
            "ITEMID": [int(target["ITEMID"]), int(target["ITEMID"]), int(other["ITEMID"])],
            "VALUE": ["128", "ERROR", "80"],
            "VALUEUOM": ["", "", ""],
        }
    )
    cleaned = clean_events(chunk, var_map)
    assert set(cleaned["VARIABLE"]) == {"Glucose", "Heart Rate"}
    assert len(cleaned[cleaned["VARIABLE"] == "Glucose"]) == 1
    # variables without a unit fixer keep their raw text; the discretizer casts
    heart_rate = cleaned.loc[cleaned["VARIABLE"] == "Heart Rate", "VALUE"].iloc[0]
    assert float(heart_rate) == 80.0


def test_layout_is_the_76_channel_contract() -> None:
    layout = load_layout()
    assert len(layout.columns) == 76
    assert len(layout.data_columns) == 59
    assert layout.columns[-1] == f"mask->{layout.channels[-1]}"
    assert len(layout.continuous_columns) == 12
    # masks live after the data block, so continuous indices are inside it
    assert max(layout.continuous_columns) < 59


def test_discretizer_bins_follow_the_benchmark_formula() -> None:
    discretizer = BenchmarkDiscretizer(max_steps=256)
    # int(end/timestep + 1 - eps): a 47.5h stay yields 48 bins, an integral
    # 24h stay stays at 24 because the +1 is consumed by the epsilon.
    assert discretizer.bin_count(47.5) == 48
    assert discretizer.bin_count(24.0) == 24
    frame, mask = discretizer.transform({"Heart Rate": {0.5: 80.0}}, 24.0)
    assert frame.shape == (24, 76)
    assert mask.sum() == 24


def test_discretizer_keeps_latest_value_per_bin_and_imputes_previous() -> None:
    discretizer = BenchmarkDiscretizer(max_steps=256)
    layout = discretizer.layout
    column = layout.columns.index("Heart Rate")
    mask_column = layout.columns.index("mask->Heart Rate")
    frame, _ = discretizer.transform({"Heart Rate": {0.5: 80.0, 0.9: 90.0, 2.3: 70.0}}, 5.0)
    assert frame[0, column] == 90.0  # 0.9h overwrites 0.5h inside bin 0
    assert frame[1, column] == 90.0  # previous-value imputation
    assert frame[2, column] == 70.0
    assert frame[4, column] == 70.0
    assert frame[:, mask_column].tolist() == [1, 0, 1, 0, 0]


def test_discretizer_imputes_the_normal_value_before_first_observation() -> None:
    discretizer = BenchmarkDiscretizer(max_steps=256)
    layout = discretizer.layout
    column = layout.columns.index("Glucose")
    # bin_id = int(t/timestep - eps), so 1.5h lands in bin 1
    frame, _ = discretizer.transform({"Glucose": {1.5: 200.0}}, 3.0)
    normal = float(layout.normal_values["Glucose"])
    assert frame[0, column] == normal
    assert frame[1, column] == 200.0
    assert frame[2, column] == 200.0


def test_discretizer_one_hots_categoricals_including_normal_values() -> None:
    discretizer = BenchmarkDiscretizer(max_steps=256)
    layout = discretizer.layout
    channel = "Glascow coma scale eye opening"
    base = layout.offsets()[channel]
    width = len(layout.possible_values[channel])
    frame, _ = discretizer.transform({channel: {1.5: "1 No Response"}}, 2.0)
    observed = frame[1, base : base + width]
    normal = frame[0, base : base + width]
    observed_position = layout.possible_values[channel].index("1 No Response")
    normal_position = layout.possible_values[channel].index(layout.normal_values[channel])
    assert observed[observed_position] == 1.0 and observed.sum() == 1.0
    assert normal[normal_position] == 1.0 and normal.sum() == 1.0
    assert observed_position != normal_position


def test_discretizer_caps_steps_and_masks_the_padding() -> None:
    discretizer = BenchmarkDiscretizer(max_steps=8)
    series = {float(hour): 60.0 + hour for hour in range(0, 40)}
    frame, mask = discretizer.transform({"Heart Rate": series}, 48.0)
    assert frame.shape == (8, 76)
    assert mask.sum() == 8


def test_normalizer_zscores_only_continuous_columns() -> None:
    layout = load_layout()
    normalizer = BenchmarkNormalizer(layout.continuous_columns)
    rng = np.random.default_rng(3)
    frame = np.zeros((100, 76), dtype=np.float64)
    continuous = layout.continuous_columns
    for position, column in enumerate(continuous):
        frame[:, column] = rng.normal(loc=position, scale=2.0, size=100)
    categorical = 0
    frame[:, categorical] = 5.0
    normalizer.feed(frame)
    normalizer.finalize()
    transformed = normalizer.transform(frame.copy())
    assert np.allclose(transformed[:, continuous].mean(axis=0), 0.0, atol=1e-9)
    # sample standard deviation, matching the benchmark's normalizer state
    assert np.allclose(transformed[:, continuous].std(axis=0, ddof=1), 1.0, atol=1e-6)
    assert np.all(transformed[:, categorical] == 5.0)


def test_iter_stay_events_collapses_duplicate_timestamps_by_max() -> None:
    collapsed = iter_stay_events(
        [(1.0, "Heart Rate", 80.0), (1.0, "Heart Rate", 90.0), (2.0, "Heart Rate", 60.0)]
    )
    assert collapsed["Heart Rate"] == {1.0: 90.0, 2.0: 60.0}
