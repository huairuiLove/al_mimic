# Multimodal active learning on MIMIC EHR benchmarks

## Base tasks

| Base | Dataset and target | Specification |
|---|---|---|
| Yang-Wu BertEncoder | MIMIC-III, 915 local ICD-9 groups (1,042 in paper) | [`BASE_TASK_MIMIC_III_MULTIMODAL.md`](BASE_TASK_MIMIC_III_MULTIMODAL.md) |
| Acute-care phenotyping | MIMIC-III, 25 native multi-hot phenotypes | [`BASE_TASK_MIMIC_III_PHENOTYPING.md`](BASE_TASK_MIMIC_III_PHENOTYPING.md) |
| Broad CCS phenotyping | MIMIC-III, 172 native multi-hot phenotypes | [`BASE_TASK_MIMIC_III_PHENOTYPING.md`](BASE_TASK_MIMIC_III_PHENOTYPING.md) |
| MDS-ED S4 + MLP | MIMIC-IV/ECG, 1,428 ICD-10-CM labels | [`BASE_TASK_MIMIC_IV_MDS_ED.md`](BASE_TASK_MIMIC_IV_MDS_ED.md) |
| BRSET ResNet-50 + metadata fusion | BRSET v1.0.2, 13 retinal disease labels | [`BASE_TASK_BRSET_MULTIMODAL.md`](BASE_TASK_BRSET_MULTIMODAL.md) |

## MIMIC-III Yang-Wu base

The formal base classifier is Yang and Wu's **BertEncoder** for the MIMIC-III
48-hour multi-label Diagnoses task. It jointly uses ClinicalBERT notes, FIDDLE
hourly time-series features, and FIDDLE time-invariant features to predict 1,042
three-digit ICD-9 groups. CAML is not used by any formal experiment.

The complete fixed protocol is documented in
[`BASE_TASK_MIMIC_III_MULTIMODAL.md`](BASE_TASK_MIMIC_III_MULTIMODAL.md).

The MDS-ED task remains a separate integration target; its data and labels are
never passed through this MIMIC-III runner.

## MIMIC-III task registry

The MIMIC runner now exposes three native multi-label tasks:

```bash
uv run python main.py tasks --config configs/mimic_comal.yaml
```

`icd9_diagnoses` preserves the existing 48-hour Yang-Wu protocol and
Recall@10/20/30 evaluation. `phenotyping_25` and `phenotyping_ccs_172` use one
complete ICU-stay label vector per query and are evaluated primarily by
macro-AUPRC. Their author repositories, preprocessing adapters, provenance
requirements, base-model adaptation boundary, and commands are documented in
[`BASE_TASK_MIMIC_III_PHENOTYPING.md`](BASE_TASK_MIMIC_III_PHENOTYPING.md).

Run MDS-ED only from its own directory after preparing its ECG memmap:

```bash
cd MDS-ED-main/src
python prepare_release.py --audit-only
python prepare_release.py --validate-prepared --output data/memmap
python main_all.py --config-name config_supervised_multimodal_mdsed_diagnoses_s4
```

The root `main.py` and `mimic-comal` CLI are intentionally MIMIC-III-only; the
MDS-ED loader, folds, labels and macro-AUROC evaluation stay under
`MDS-ED-main/`.

## BRSET multimodal base

BRSET is implemented as an independent full-data runner. It fuses an ImageNet
ResNet-50 fundus branch with pre-diagnostic clinical metadata and predicts 13
independent disease labels. Its active-learning unit is the patient, so paired
eye images remain together. The complete protocol, data counts, leakage rules,
metrics, and commands are documented in
[`BASE_TASK_BRSET_MULTIMODAL.md`](BASE_TASK_BRSET_MULTIMODAL.md).

```bash
python -m brset_al.cli prepare --config configs/brset_comal.yaml
bash scripts/run_brset_four_methods.sh
```

## Formal methods

| Config | Acquisition method |
|---|---|
| `configs/mimic_comal.yaml` | CoMAL |
| `configs/mimic_mm_comal.yaml` | MM-CoMAL |
| `configs/mimic_modis.yaml` | MoDIS |
| `configs/mimic_mosaic.yaml` | MoSAIC |

Every method runs six cold-start rounds at 10%, 15%, 20%, 25%, 30%, and 35% of
the actual official train split. Each round reloads the same ClinicalBERT source,
reinitializes every other model/optimizer parameter, and uses the same 1,200-step
optimization budget (roughly the former 20-epoch budget at the largest round)
with an 80-epoch ceiling and validation-loss early stopping.
Evaluation reports only Recall@10, Recall@20, and Recall@30.

## Required data

Generate the official Yang-Wu/FIDDLE MIMIC-III artifact with the upstream code:

https://github.com/emnlp-mimic/mimic

Then set these paths in `configs/mimic_a800_144c.yaml`:

```yaml
dataset:
  split_hdf5: data/yang_wu_mimic/features/outcome=Diagnoses,T=48.0,dt=1.0/splits.hdf5
  clinicalbert_checkpoint: pretrained/clinicalbert/pytorch_model.bin
```

The loader requires `splits.hdf5/with_notes/{train,val,test}` and validates the
paper dimensions: 10,210 visits total, 1,042 labels, `[48,7411]` time series,
97 static features, 512 note tokens, and a row-aligned `SUBJECT_ID` (or
`subject_id`) array. The subject identifier is used for MoDIS grouped OOF
folds; global row indices are rejected as a substitute. No fallback or
synthetic data path exists.

The current executable artifact is the local Yang-Wu rebuild: 10,258 visits,
915 diagnosis groups, and `[48,7749]` time-series tensors. The paper's
10,210/1,042/7,411 dimensions above remain reference values, not the dimensions
reported for local runs.

`configs/mimic_full_cohort.yaml` is still the same Yang-Wu task and BertEncoder.
It accepts only a separately rebuilt, larger Yang-Wu HDF5 that freezes the same
48-hour input contract and 915-label vocabulary, with disjoint subject-grouped
train/validation/test splits. Its cohort includes all MIMIC-III ICU care systems,
so it is an extended-cohort Yang-Wu result rather than a reproduction of the
paper's MetaVision cohort. Run that full-data upper bound with:

```bash
uv run python main.py full-data --config configs/mimic_full_cohort.yaml
```

## Run

```bash
uv sync --dev
uv run python main.py validate-data --config configs/mimic_comal.yaml
uv run scripts/run_four_methods.sh active
```

Each experiment writes six checkpoints, per-round metrics and queries,
`active_state.json`, `final_metrics.json`, and final validation/test predictions.
The `data_usage` block records the exact amount of data used.
