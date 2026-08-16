# Dataset and artifact layout

`dataset/` is the only supported home for local data assets. Credentialed
clinical data, derived tensors, prepared manifests, and external checkpoints are
not source code and are ignored by Git.

```text
dataset/
|-- raw/         # official downloaded releases, unchanged where practical
|-- processed/   # derived tensors and intermediate feature tables
|-- prepared/    # validated manifests, split metadata, memmaps, and shards
`-- pretrained/  # external initialization checkpoints and tokenizers
```

Run `al-mimic` from the repository root with a runnable file under
`configs/experiments/`. Do not use a local upstream source checkout as a data
path or preprocessing runtime.

## Ownership rules

- `raw/` contains source releases obtained from their official distributor.
- `processed/` contains expensive transformations such as FIDDLE/HDF5 task
  tensors. They remain data and are not committed.
- `prepared/` contains compact runtime manifests and task-specific stores
  produced or validated by `al-mimic prepare`.
- `pretrained/` contains checkpoints such as ClinicalBERT. Their upstream
  licenses still apply.
- `experiments/` is separate from `dataset/`; it contains model checkpoints,
  predictions, metrics, query histories, figures, and logs.
- `thirdparty/` is not a data directory. If present, it is ignored in full and
  used only for manual source comparison; runtime code and tests do not refer to
  it.

Formal loaders fail when required files, dimensions, IDs, folds, or checksums do
not match. They do not generate synthetic clinical samples as a fallback.

## MIMIC-III diagnosis

The `mimic_iii` / `icd9_diagnoses` task expects:

```text
dataset/processed/yang_wu_mimic/
`-- features/outcome=Diagnoses,T=48.0,dt=1.0/splits.hdf5

dataset/pretrained/clinicalbert/
|-- config.json
|-- pytorch_model.bin
`-- vocab.txt

dataset/prepared/yang_wu_diagnoses_48h/
`-- manifest.json                    # written by al-mimic prepare
```

The exact HDF5 location is controlled by the resolved config. The file must have
`with_notes/{train,val,test}` groups and row-aligned subject IDs. The current
local rebuild contract is 10,258 stays, 915 labels, 48 hourly steps, 7,749
hourly features, 97 invariant features, and 512 note tokens.

This workspace contains a FIDDLE-derived HDF5 under a different local processed
subdirectory and historical completed experiment outputs. The configured HDF5
path is a broken symbolic link to an absolute path on another host, so a fresh
`validate-data` cannot open it. Replace that local link with the intended local
artifact before rerunning; do not treat historical outputs as a validation
substitute.

## MIMIC-III phenotyping

The native phenotyping configs expect:

```text
dataset/processed/mimic_phenotyping_25/splits.hdf5
dataset/processed/mimic_phenotyping_ccs_172/splits.hdf5

dataset/prepared/mimic_phenotyping_25/manifest.json
dataset/prepared/mimic_phenotyping_ccs_172/manifest.json
```

The HDF5 files contain task-native multi-hot labels, structured measurements,
notes, masks, `subject_id`, `stay_id`, and label metadata. `al-mimic prepare`
audits an existing HDF5 and writes the corresponding prepared manifest; it does
not expose raw-table-to-HDF5 construction as a public task action.

Neither phenotyping HDF5 nor its prepared manifest or experiment output is
present in this workspace as of 2026-08-16.

## BRSET v1.0.2

The `brset` task expects the official release in:

```text
dataset/raw/brset-1.0.2/
|-- label_brset.csv
|-- fundus_photos/
|-- LICENSE.txt
`-- SHA256SUMS.txt
```

`al-mimic prepare --task brset` is designed to write:

```text
dataset/prepared/brset_v1_0_2/
|-- split_manifest.csv
|-- metadata_schema.json
`-- data_audit.json
```

The source release is present locally, while the prepared directory and BRSET
experiment outputs are absent. The split manifest must remain patient-disjoint;
copying an image-level random split into this directory is not valid
preparation.

## MDS-ED

The `mds_ed` config expects:

```text
dataset/raw/mds-ed-1.0.0/
|-- mds_ed.csv
|-- LICENSE.txt
`-- SHA256SUMS.txt

dataset/raw/mimic-iv-ecg-1.0/
`-- files/                            # extracted waveform tree
```

`al-mimic prepare --task mds_ed` copies and transforms those official inputs
into `dataset/prepared/mds_ed/`. The output includes:

```text
dataset/prepared/mds_ed/
|-- mds_ed.csv
|-- manifest.json
|-- ecg_prepare_manifest.jsonl
|-- df_memmap.pkl
|-- memmap_meta.npz
|-- tabular_spec.json
|-- tabular_manifest.json
|-- tabular_*.npz
`-- *.npy or memmap data files
```

The release CSV and extracted ECG source are present locally. The prepared
memmap/shards and supervised training outputs are absent as of 2026-08-16.
Preparation requires the optional `mds-ed` dependencies and preserves official
folds 0-17/18/19 for train/validation/test.

Other local MIMIC-IV or MIMIC-IV-ED releases are source material for rebuilding
the published MDS-ED table, but the current `al-mimic prepare` action consumes
the released `mds_ed.csv` and ECG waveforms; it does not reconstruct that table
from raw hospital and ED tables.

## Experiment artifacts

Active MIMIC-III and BRSET runs write below the experiment name configured under
`experiments/`:

```text
experiments/<name>/
|-- checkpoints/
|-- active_state.json
|-- final_metrics.json
|-- final_predictions.npz
`-- resolved_config.json
```

Task-specific audit files and MIMIC-III `figures/` may also be present.
`active_state.json` and `final_metrics.json` contain the authoritative data-usage
and evaluation records; directory names alone do not prove a run completed.

MDS-ED supervised training writes `native_temporal_adapter.pt` and
`training_summary.json` to its configured experiment output. It does not produce
active-learning query artifacts in the current supervised-only implementation.

All generated data and experiment directories are local, reproducible artifacts
and should remain untracked. Preserve manifests beside transferred artifacts so
source versions, dimensions, folds, and resolved paths can be audited.
