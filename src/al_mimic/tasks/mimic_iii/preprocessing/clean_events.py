"""First-party event cleaning for the 17 benchmark variables.

Reimplements the executed cleaning chain of the MIMIC-III Benchmark over
CHARTEVENTS/LABEVENTS/OUTPUTEVENTS rows:

1. map ITEMID to one of the 17 LEVEL2 variables via the materialised
   ``itemid_to_variable_map.csv`` (rows with STATUS 'ready' and COUNT > 0);
2. apply the per-variable unit and format fixes (blood-pressure strings,
   capillary-refill wording, FIO2 percent scales, Fahrenheit temperatures,
   ounce/pound weights, inch heights);
3. drop rows whose value is still missing.

The benchmark's ``variable_ranges.csv`` outlier clamp is deliberately absent:
the upstream scripts import the table but never call the clamp, so the executed
protocol does not include it.

Raw values stay strings until a cleaner converts them, mirroring the CSV
round-trip the benchmark feeds its discretizer; categorical channel values
must remain their original text for the one-hot lookup.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESOURCES = Path(__file__).with_name("resources")
ITEMID_MAP_CSV = RESOURCES / "itemid_to_variable_map.csv"

EVENT_COLUMNS = ["SUBJECT_ID", "ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUE", "VALUEUOM"]


def load_itemid_map(path: str | Path = ITEMID_MAP_CSV) -> pd.DataFrame:
    """ITEMID -> (VARIABLE, MIMIC LABEL) for the 17 benchmark variables."""
    var_map = pd.read_csv(path, dtype=str).fillna("")
    var_map["COUNT"] = var_map["COUNT"].astype(int)
    var_map = var_map[(var_map["LEVEL2"] != "") & (var_map["COUNT"] > 0) & (var_map["STATUS"] == "ready")]
    return var_map[["ITEMID", "LEVEL2", "MIMIC LABEL"]].rename(
        columns={"LEVEL2": "VARIABLE", "MIMIC LABEL": "MIMIC_LABEL"}
    )


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype(float)


def clean_sbp(values: pd.Series) -> pd.Series:
    """Split '120/80' style strings, keeping the systolic member."""
    text = values.astype(str)
    numeric = _numeric(text)
    slash = text.str.extract(r"^(\d+)/(\d+)$", expand=False)[0]
    return numeric.fillna(pd.to_numeric(slash, errors="coerce"))


def clean_dbp(values: pd.Series) -> pd.Series:
    """Split '120/80' style strings, keeping the diastolic member."""
    text = values.astype(str)
    numeric = _numeric(text)
    slash = text.str.extract(r"^(\d+)/(\d+)$", expand=False)[1]
    return numeric.fillna(pd.to_numeric(slash, errors="coerce"))


def clean_crr(values: pd.Series) -> pd.Series:
    text = values.astype(str)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    result[text.isin(["Normal <3 secs", "Brisk"])] = 0.0
    result[text.isin(["Abnormal >3 secs", "Delayed"])] = 1.0
    return result


def clean_fio2(values: pd.Series, uom: pd.Series) -> pd.Series:
    """Map 0-100 scale fractions without a torr unit down to 0-1."""
    numeric = _numeric(values)
    torr_free = uom.fillna("").astype(str).str.lower().str.contains("torr")
    rescale = (~torr_free) & (numeric > 1.0)
    numeric.loc[rescale] = numeric.loc[rescale] / 100.0
    return numeric


def clean_lab(values: pd.Series) -> pd.Series:
    """Glucose and pH: anything that is not a plain number becomes missing."""
    return _numeric(values)


def clean_o2sat(values: pd.Series) -> pd.Series:
    numeric = _numeric(values)
    fractional = numeric <= 1
    numeric.loc[fractional] = numeric.loc[fractional] * 100.0
    return numeric


def clean_temperature(values: pd.Series, uom: pd.Series, labels: pd.Series) -> pd.Series:
    numeric = _numeric(values)
    fahrenheit = (
        uom.fillna("").astype(str).str.lower().str.contains("f")
        | labels.fillna("").astype(str).str.lower().str.contains("f")
        | (numeric >= 79)
    )
    numeric.loc[fahrenheit] = (numeric.loc[fahrenheit] - 32.0) * 5.0 / 9.0
    return numeric


def clean_weight(values: pd.Series, uom: pd.Series, labels: pd.Series) -> pd.Series:
    numeric = _numeric(values)
    uom_text = uom.fillna("").astype(str).str.lower()
    label_text = labels.fillna("").astype(str).str.lower()
    ounces = uom_text.str.contains("oz") | label_text.str.contains("oz")
    pounds = ounces | uom_text.str.contains("lb") | label_text.str.contains("lb")
    numeric.loc[ounces] = numeric.loc[ounces] / 16.0
    numeric.loc[pounds] = numeric.loc[pounds] * 0.453592
    return numeric


def clean_height(values: pd.Series, uom: pd.Series, labels: pd.Series) -> pd.Series:
    numeric = _numeric(values)
    inches = uom.fillna("").astype(str).str.lower().str.contains("in") | labels.fillna("").astype(
        str
    ).str.lower().str.contains("in")
    numeric.loc[inches] = np.round(numeric.loc[inches] * 2.54)
    return numeric


def clean_events(chunk: pd.DataFrame, var_map: pd.DataFrame) -> pd.DataFrame:
    """Map a raw event chunk to benchmark variables and clean it in place."""
    mapped = chunk.merge(var_map.assign(ITEMID=var_map["ITEMID"].astype(int)), on="ITEMID", how="inner")
    if mapped.empty:
        return mapped
    mapped["VALUE"] = mapped["VALUE"].astype(object)

    for variable, cleaner in (
        ("Capillary refill rate", lambda frame: clean_crr(frame["VALUE"])),
        ("Systolic blood pressure", lambda frame: clean_sbp(frame["VALUE"])),
        ("Diastolic blood pressure", lambda frame: clean_dbp(frame["VALUE"])),
        ("Fraction inspired oxygen", lambda frame: clean_fio2(frame["VALUE"], frame.get("VALUEUOM"))),
        ("Oxygen saturation", lambda frame: clean_o2sat(frame["VALUE"])),
        ("Glucose", lambda frame: clean_lab(frame["VALUE"])),
        ("pH", lambda frame: clean_lab(frame["VALUE"])),
        (
            "Temperature",
            lambda frame: clean_temperature(frame["VALUE"], frame.get("VALUEUOM"), frame["MIMIC_LABEL"]),
        ),
        ("Weight", lambda frame: clean_weight(frame["VALUE"], frame.get("VALUEUOM"), frame["MIMIC_LABEL"])),
        ("Height", lambda frame: clean_height(frame["VALUE"], frame.get("VALUEUOM"), frame["MIMIC_LABEL"])),
    ):
        selected = mapped["VARIABLE"] == variable
        if not selected.any():
            continue
        cleaned = cleaner(mapped.loc[selected])
        numeric = pd.to_numeric(cleaned, errors="coerce")
        # Categorical channels keep their original string text; every cleaned
        # variable here is numeric, so a failed parse means the row is dropped.
        mapped.loc[selected, "VALUE"] = numeric.to_numpy()
    return mapped[mapped["VALUE"].notna()]
