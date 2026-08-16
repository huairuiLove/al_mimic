# Base task: MIMIC-IV MDS-ED diagnosis prediction

## Scope and current status

This task is the `mds_ed` plugin. It prepares the MIMIC-IV-Ext-MDS-ED release
and trains a package-local supervised diagnosis model. One row maps an ECG study
to a 1,428-dimensional ICD-10-CM multi-hot target.

The current plugin is explicitly `supervised_only`. Its actions are `prepare`,
`validate-data`, `train`, and `hardware`; it has no `active` action and supports
none of the acquisition methods. The local release CSV and extracted
MIMIC-IV-ECG source are present, but `dataset/prepared/mds_ed/` and supervised
training outputs are not present.

## Sources and MIT provenance

- Paper: https://arxiv.org/abs/2407.17856
- MDS-ED release: https://physionet.org/content/multimodal-emergency-benchmark/1.0.0/
- Author code: https://github.com/AI4HealthUOL/MDS-ED
- MIMIC-IV-ECG: https://physionet.org/content/mimic-iv-ecg/1.0/

The package-local preparation and task integration preserve the upstream MIT
license and notice in `src/al_mimic/tasks/mds_ed/LICENSE` and
`src/al_mimic/tasks/mds_ed/NOTICE`. Runtime code uses only the package-local
implementation and official data files.

## Important model boundary

The published deep-learning baseline uses an S4 waveform encoder. The current
`NativeMdsEdTemporalAdapter` instead uses four residual depthwise/pointwise
convolutional temporal blocks, mean pooling, and a three-layer tabular MLP.

It preserves the broad four-layer temporal plus tabular fusion shape, but it is
**not a word-for-word, operator-for-operator, or numerically equivalent
implementation of the upstream S4 benchmark**. Its checkpoint records backend
`native_temporal_adapter`. Results must use that name and must not be reported as
an exact S4 reproduction, even though the current config and output directory
names retain `s4` as historical protocol naming.

## Data contract

The recommended inputs are:

- `dataset/raw/mds-ed-1.0.0/mds_ed.csv` from MDS-ED v1.0.0;
- the extracted `dataset/raw/mimic-iv-ecg-1.0/` waveform tree.

The prepared MDS-ED table already contains engineered demographics, biometrics,
vital/laboratory trends, labels, study/subject IDs, and fold assignments. Raw
MIMIC-IV and MIMIC-IV-ED tables are not read by the current `prepare` action.
They are needed only to reconstruct the release table outside this runtime path.

The release audit verifies required columns, 470 raw clinical features, 1,428
diagnosis labels, fold values, and study identifiers. Preparation then:

1. Discovers every ECG study referenced by the release CSV.
2. Resamples each 12-lead waveform to 100 Hz and produces 1,000 samples.
3. Interpolates internal missing values, zero-fills boundary gaps, and clips
   amplitude to 3 mV.
4. Builds ECG memmaps and a study-to-record index.
5. Fits tabular medians and categorical values on training folds only.
6. Writes continuous/categorical feature shards, missingness masks, labels, and
   provenance manifests.

The official 20-fold split is preserved: folds 0-17 train, 18 validation, and 19
test. Validation and test retain only the first ECG within a stay; all training
ECGs are retained.

## Training and evaluation status

The native trainer uses AdamW, BCE with logits, learning rate `1e-3`, weight
decay `1e-3`, batch size 64, 20 epochs, model dimension 512, four temporal
blocks, and a 128-dimensional tabular branch.

The published diagnosis endpoint uses macro AUROC with a 1,000-sample empirical
bootstrap confidence interval. The current package trainer does **not** yet
implement that benchmark evaluation: it records train and validation BCE loss,
saves the final checkpoint, and does not produce test macro AUROC or bootstrap
intervals. Consequently, the current MDS-ED integration is a preparation and
native supervised-training adapter, not a completed published benchmark or an
active-learning result.

## Commands

Install optional data dependencies, then prepare from the configured release and
ECG paths:

```bash
uv sync --dev --extra mds-ed

uv run al-mimic prepare \
  --task mds_ed \
  --config configs/experiments/mds_ed/diagnoses.yaml
```

Preparation writes under `dataset/prepared/mds_ed/` and can resume by default.
Use `--no-resume` to rebuild waveform preparation. CLI path overrides are
available when official files are stored elsewhere:

```bash
uv run al-mimic prepare \
  --task mds_ed \
  --config configs/experiments/mds_ed/diagnoses.yaml \
  --release-csv /path/to/mds_ed.csv \
  --ecg-root /path/to/mimic-iv-ecg-1.0 \
  --prepared-dir /path/to/prepared-mds-ed
```

Validate the release and prepared memmap, then train:

```bash
uv run al-mimic validate-data \
  --task mds_ed \
  --config configs/experiments/mds_ed/diagnoses.yaml

uv run al-mimic train \
  --task mds_ed \
  --config configs/experiments/mds_ed/diagnoses.yaml
```

## Artifacts

Preparation writes `mds_ed.csv`, waveform memmap files, `memmap_meta.npz`,
`df_memmap.pkl`, `ecg_prepare_manifest.jsonl`, `tabular_spec.json`, tabular NPZ
shards, `tabular_manifest.json`, and `manifest.json` in the prepared directory.

Training writes `native_temporal_adapter.pt` and `training_summary.json` below
the configured experiment output. No active-learning state or query artifacts
are expected for this task in its current supervised-only form.
