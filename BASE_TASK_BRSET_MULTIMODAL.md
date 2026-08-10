# Base task: BRSET multimodal 13-label active learning

This is a separate base from the MIMIC-III and MIMIC-IV/MDS-ED tasks. It uses
the local **A Brazilian Multilabel Ophthalmological Dataset (BRSET) v1.0.2**
release at `brazilian-ophthalmological_1.0.2/`.

## What is and is not reproduced

The published BRSET multi-label baseline is Gould, Yang, and Clifton,
*Deep Learning for Multi-Label Disease Classification of Retinal Images*:

- ImageNet-pretrained ResNet-50;
- 13 independent sigmoid outputs;
- Adam and binary cross-entropy;
- 60/20/20 train/validation/test split;
- image-only input.

The BRSET data paper uses ConvNeXt V2 for four selected image-only tasks,
not the complete 13-label multimodal task. A separate multimodal paper uses
fundus embeddings and limited metadata to predict four diabetes-duration
groups, not the 13 disease labels.

This repository therefore defines a new, explicit extension: ResNet-50 image
features fused with pre-diagnostic clinical metadata, followed by a 13-label
sigmoid head. It must be reported as a **BRSET multimodal extension baseline**,
not as an exact reproduction of a published multimodal BRSET model.

Sources:

- Dataset: https://physionet.org/content/brazilian-ophthalmological/1.0.2/
- Data paper: https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000454
- 13-label paper: https://www.medrxiv.org/content/10.1101/2024.02.12.24302676v1.full
- Multimodal metadata paper: https://www.sciencedirect.com/science/article/pii/S1751991824000743

## Dataset audit

The local v1.0.2 release contains:

- 16,266 JPEG fundus photographs;
- 8,524 patients;
- 34 CSV columns;
- 13 binary disease labels;
- 9,754/3,259/3,253 images and 5,114/1,705/1,705 patients in the generated
  patient-level train/validation/test split.

The prepared audit is written to
`prepared/brset_v1_0_2/data_audit.json`. The current local positive counts are
also recorded there. They differ slightly from the original paper's earlier
release, so v1.0.2 results must not be presented as the paper's exact numeric
reproduction.

The split is performed on patient-level label unions using deterministic
multi-label stratification. Patients with paired right/left eye photographs
never cross a split. The active-learning query unit is also a patient: when a
patient is selected, every train image for that patient is added together.

## Inputs and leakage control

The metadata branch uses only fields available before the target annotation:

- numeric: `patient_age`, `diabetes_time_y`;
- categorical: `camera`, `insulin`, `patient_sex`, `exam_eye`, `diabetes`;
- tokenized comorbidity indicators from `comorbidities`.

Numeric statistics and vocabularies are fitted on the train split only. Missing
values have explicit indicators. `nationality` is constant in this release and
is excluded.

The following annotation fields are never model inputs: `optic_disc`,
`vessels`, `macula`, `DR_SDRG`, `DR_ICDR`, `focus`, `illumination`,
`image_field`, `artifacts`, and `quality`. They are post-diagnostic labels or
quality annotations and would leak target information.

Images are resized to 256 pixels, cropped to 224 pixels, normalized using the
ResNet-50 ImageNet statistics, and augmented in train only with random resized
crop, 30-degree rotation, and horizontal flip. All images, including images
marked inadequate by the quality annotation, remain in the dataset.

## Model and formal training

The model is implemented in `brset_al/model.py`:

1. ResNet-50 with `IMAGENET1K_V2` weights produces a 2,048-dimensional image
   representation.
2. A two-layer MLP encodes the 45-dimensional fitted metadata vector.
3. Both branches are projected to 512 dimensions and fused by summation and
   LayerNorm.
4. A linear head produces 13 independent logits; sigmoid is used only for
   probabilities and decisions.

Every round creates a new model from the same ImageNet source weights and fresh
fusion/head parameters. It never loads a previous round checkpoint. The formal
training configuration is:

- Adam, learning rate `1e-4`, weight decay `0`;
- ordinary BCE-with-logits loss;
- batch size `8`;
- exactly `20` classifier epochs per round;
- fp32 and CUDA;
- no early stopping, dry-run, smoke mode, row cap, or warm start.

The original Gould paper reports six epochs as its image-only hyperparameter
search result. The active-learning protocol here requires 20 fresh epochs per
round, matching the repository's formal six-round experiment contract; this is
an explicit active-learning protocol choice.

## Active-learning protocol

All four methods run against the same full train pool and the same patient-level
initial sample:

- six cold-start rounds;
- cumulative targets of 10%, 15%, 20%, 25%, 30%, and 35% of the 5,114-patient
  train pool;
- schedule for this release: 511, 767, 1,023, 1,279, 1,534, and 1,790 labeled
  patients;
- each query selects patients, then adds all their train images;
- validation and test patients are never queried.

The four configurations are:

- `configs/brset_comal.yaml`
- `configs/brset_mm_comal.yaml`
- `configs/brset_modis.yaml`
- `configs/brset_mosaic.yaml`

CoMAL uses the fused patient representation. MM-CoMAL exposes three views:
image, metadata, and fused. MoDIS trains stop-gradient probes for the two
branches and the fused view. MoSAIC evaluates the two-modality lattice. The
shared acquisition code has been generalized to support both the existing
three-modality MIMIC model and this two-modality model.

## Metrics

Per-label F1 thresholds are fitted on validation probabilities using the paper's
grid `{0, 0.04, ..., 1}` and then frozen for test evaluation. The reported
metrics are:

- macro/micro AUROC;
- macro/micro AUPRC;
- macro/micro F1, precision, and recall;
- per-label support, AUROC, AUPRC, F1, sensitivity, specificity, and NPV;
- subset accuracy and Hamming loss.

Because retinal detachment has only 7 positive images in v1.0.2 (4/2/1 in the
three splits), every result must include per-label support. Accuracy alone is
not a valid headline metric for this long-tailed multi-label task.

## Prepare and run

Run from the repository root:

```bash
uv sync --dev
python -m brset_al.cli prepare --config configs/brset_comal.yaml
python -m brset_al.cli validate-data --config configs/brset_comal.yaml
```

Run all four formal methods:

```bash
bash scripts/run_brset_four_methods.sh
```

Run one method directly:

```bash
python -m brset_al.cli active --config configs/brset_mm_comal.yaml
```

Each output directory contains six round checkpoints, query patient/image IDs,
per-round validation/test metrics, final predictions, `active_state.json`, and
a `data_usage` block. That block is the authoritative record of how many
patients and images were actually used, because paired-eye grouping makes image
counts vary slightly by round.
