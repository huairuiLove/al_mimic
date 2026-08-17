"""Published MDS-ED release and benchmark constants."""

from __future__ import annotations

DIAGNOSIS_LABEL_COUNT = 1_428
DETERIORATION_LABEL_COUNT = 15
RAW_TABULAR_FEATURE_COUNT = 470
STRATIFIED_FOLDS = frozenset(range(20))
TRAIN_FOLDS = tuple(range(18))
VALIDATION_FOLDS = (18,)
TEST_FOLDS = (19,)
ECG_CHANNEL_COUNT = 12
ECG_SAMPLE_COUNT = 1_000
ECG_SAMPLE_RATE = 100

FEATURE_PREFIXES = ("demographics_", "biometrics_", "vitals_", "labvalues_")
REQUIRED_RELEASE_COLUMNS = (
    "general_subject_id",
    "general_study_id",
    "general_strat_fold",
    "general_ecg_no_within_stay",
    "general_data",
)

CHANNEL_TO_INDEX = {
    "i": 0,
    "ii": 1,
    "v1": 2,
    "v2": 3,
    "v3": 4,
    "v4": 5,
    "v5": 6,
    "v6": 7,
    "iii": 8,
    "avr": 9,
    "avl": 10,
    "avf": 11,
}
