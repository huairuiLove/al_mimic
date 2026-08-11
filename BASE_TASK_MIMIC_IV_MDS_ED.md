# Base task: MIMIC-IV-Ext-MDS-ED multimodal diagnosis prediction

This is a separate base task from the MIMIC-III Yang-Wu task. It follows
Lopez Alcaraz et al., *MIMIC-IV-Ext-MDS-ED: Multimodal Decision Support in the
Emergency Department*, and the released benchmark implementation:

- Paper: https://arxiv.org/abs/2407.17856
- PhysioNet release: https://physionet.org/content/multimodal-emergency-benchmark/1.0.0/
- Code: https://github.com/AI4HealthUOL/MDS-ED

## 1. Formal target

The diagnosis task is multi-label ICD-10-CM prediction. Each ECG/ED encounter
has a 1,428-dimensional multi-hot target. Codes are converted to five digits
and parent codes are propagated through the third digit, following
MIMIC-IV-ECG-Ext-ICD. This is not the MIMIC-III ICD-9/CAML task and must not be
merged with the 1,042-label Yang-Wu configuration.

The official paper cohort contains 71,098 patients, 121,195 visits and 129,057
ECG samples. The PhysioNet table should be audited at runtime because release
metadata reports a slightly different row/column count across versions. The
actual CSV dimensions, label count and fold counts are recorded by the
preparation command rather than guessed.

## 2. Data to download

### Recommended training path

Download the prepared MDS-ED table and the linked ECG waveform archive:

1. `mds_ed.csv` from MIMIC-IV-Ext-MDS-ED v1.0.0.
2. `mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip` from
   MIMIC-IV-ECG v1.0.

The prepared table already contains the MIMIC-IV and MIMIC-IV-ED-derived
demographics, biometrics, vital-sign trends, laboratory-value trends, labels,
and official fold assignment. Raw MIMIC-IV tables are not needed again for
training when this table is used.

Prepare the ECG memmap and audit the table with:

```bash
cd MDS-ED-main/src
python prepare_release.py \
  --mdsed-csv data/mds_ed.csv \
  --ecg-zip data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip \
  --output data/memmap
```

`prepare_release.py` also accepts an already extracted
`mimic-iv-ecg_1.0` directory via `--ecg-path` and auto-discovers the extracted
layout when the complete `MDS-ED-main` directory is uploaded. The release
contains 129,057 ECG rows and 1,428 diagnosis columns; rows with no positive
diagnosis code are valid negative examples and are retained.

After conversion, validate the row-to-waveform mapping and metadata:

```bash
python prepare_release.py \
  --mdsed-csv data/memmap/mds_ed.csv \
  --output data/memmap \
  --validate-prepared
```

Do not pass MDS-ED files to `main.py` or `mimic_comal`; those commands belong to
the independent MIMIC-III Yang-Wu task.

### Optional raw reconstruction path

Only use this when reproducing the dataset table itself. It requires the
credentialed parent releases and the exact files consumed by the upstream
preprocessor:

| Source | Required files |
|---|---|
| MIMIC-IV-ECG-Ext-ICD v1.0.1 | `records_w_diag_icd10.csv` |
| MIMIC-IV-ECG v1.0 | matched-subset waveform archive (`.hea`/`.dat`) |
| MIMIC-IV v2.2 | `admissions.csv.gz`, `diagnoses_icd.csv.gz`, `d_labitems.csv.gz`, `labevents.csv.gz`, `icustays.csv.gz`, `procedures_icd.csv.gz`, `omr.csv.gz` |
| MIMIC-IV-ED v2.2 | `edstays.csv.gz`, `diagnosis.csv.gz`, `pyxis.csv.gz`, `vitalsign.csv.gz`, `triage.csv.gz` |

All four source releases require PhysioNet credentialing, CITI training and a
data-use agreement. Pin these versions; do not mix MIMIC-IV v3.x or another
ECG-Ext-ICD release with the v1.0.0 benchmark.

## 3. Inputs and preprocessing

Each sample uses a single 10-second, 12-lead ECG and 470 engineered clinical
features from the first 90 minutes after ED arrival:

- demographics;
- biometrics (height, weight, BMI);
- vital parameters and trends;
- laboratory values and trends.

Chief complaints and prior medications are excluded. ECG is resampled to 100 Hz,
missing signal values are linearly interpolated, boundary gaps are zero-filled,
and amplitude is clipped to 3 mV. Tabular missing values are median-imputed from
the training folds only, with one binary mask feature for each imputed variable.

## 4. Official split and evaluation

The supplied `general_strat_fold` has 20 stratified folds: folds `0..17` are
training, `18` validation/model selection, and `19` test. Training retains all
ECGs; validation and test retain only the first ECG in each stay, identified by
`general_ecg_no_within_stay == 0`, to avoid repeated-ECG evaluation bias.

The formal diagnosis metric is macro AUROC with 1,000 empirical bootstrap
iterations and a 95% confidence interval. Binary cross-entropy with logits is
used for the 1,428 independent labels. Recall@K, F1 and thresholded accuracy
are not substitutes for the published primary metric.

## 5. Published multimodal base model

The paper's multimodal deep-learning base is:

- four-layer S4 waveform encoder, state size 8, model dimension 512;
- three-layer tabular MLP;
- embedding layers for categorical gender, race and acuity, then concatenation
  of ECG and tabular representations;
- AdamW, learning rate `1e-3`, weight decay `1e-3`, constant schedule,
  batch size `64`, 20 epochs;
- model selection by validation macro AUROC.

The corrected diagnosis config is
`MDS-ED-main/src/conf/config_supervised_multimodal_mdsed_diagnoses_s4.yaml`.
The upstream YAML previously used 250 ECG samples, batch 32, 40 epochs and no
missingness columns; those values are not the paper base.

## 6. Active-learning adapter contract

For comparison with the existing active-learning experiments, the MDS-ED
adapter must keep the base task fixed and apply the same six cold-start rounds:
10%, 15%, 20%, 25%, 30%, 35% of the official training pool, with a fresh model
and optimizer each round and a 5% query increment. Acquisition methods may
rank training ECG encounters, but may not alter the 1,428-label target, fold
assignment, 90-minute observation window, waveform preprocessing, or macro
AUROC evaluation. This adapter is a separate integration target; the current
Yang-Wu runner does not silently treat MDS-ED as MIMIC-III data.
