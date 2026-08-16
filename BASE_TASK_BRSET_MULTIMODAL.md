# Base task: BRSET multimodal retinal diagnosis

## Scope and status

This task is the `brset` plugin. It predicts 13 retinal disease labels from a
fundus photograph and pre-diagnostic clinical metadata. The query unit is a
patient: selecting a patient labels all of that patient's train images, so
paired eyes never cross an acquisition boundary.

The first-party implementation and configs exist under `src/al_mimic/tasks/brset`
and `configs/`. The local BRSET v1.0.2 source release is present at
`dataset/raw/brset-1.0.2`, but the prepared patient split and experiment outputs
are absent. The task plugin loads successfully and declares the full active-
learning capability set described below; the source release still needs to be
prepared before a run can start.

## Provenance and reproduction boundary

Sources:

- Dataset v1.0.2: https://physionet.org/content/brazilian-ophthalmological/1.0.2/
- Data paper: https://doi.org/10.1371/journal.pdig.0000454
- 13-label image baseline:
  https://www.medrxiv.org/content/10.1101/2024.02.12.24302676v1
- Related metadata study:
  https://www.sciencedirect.com/science/article/pii/S1751991824000743

The published 13-label baseline is image-only. The BRSET data paper evaluates
selected image-only tasks, and the related multimodal paper has a different
four-group target. This repository therefore defines an explicit extension:
ImageNet ResNet-50 features plus leakage-controlled clinical metadata followed
by a 13-label sigmoid head. It must be reported as the repository's BRSET
multimodal extension, not an exact reproduction of a published multimodal model.

## Data and split

The configured v1.0.2 contract expects:

- 16,266 JPEG fundus photographs;
- 8,524 patients;
- 13 binary disease labels in `label_brset.csv`;
- images in `fundus_photos/`.

`al-mimic prepare` audits the source and creates a deterministic patient-level,
multi-label-stratified 60/20/20 split. It writes a split manifest, train-fitted
metadata schema, and `data_audit.json` under
`dataset/prepared/brset_v1_0_2/`. The audit is the authority for actual split
counts and per-label support. The prepared directory is not present locally at
the time of this documentation update.

## Inputs and leakage control

The metadata branch uses only fields available before target annotation:

- numeric: `patient_age`, `diabetes_time_y`;
- categorical: `camera`, `insulin`, `patient_sex`, `exam_eye`, `diabetes`;
- tokenized indicators from `comorbidities`.

Means, standard deviations, categorical vocabularies, and comorbidity vocabulary
are fitted on the train split only. Numeric missingness is represented
explicitly, and unseen categories have an unknown bucket.

Post-diagnostic and quality annotations are excluded from model input, including
`optic_disc`, `vessels`, `macula`, `DR_SDRG`, `DR_ICDR`, `focus`,
`illumination`, `image_field`, `artifacts`, and `quality`.

Images are resized to 256, cropped to 224, and normalized with ImageNet
ResNet-50 statistics. Train augmentation uses random resized crop, rotation, and
horizontal flip. Images marked inadequate remain in the formal dataset.

## Model and training

`BrsetMultimodalClassifier` contains:

1. An ImageNet `IMAGENET1K_V2` ResNet-50 image encoder.
2. A two-layer metadata encoder.
3. Separate 512-dimensional image and metadata tokens.
4. Sum fusion followed by LayerNorm and a 13-logit linear head.

Training uses Adam, learning rate `1e-4`, BCE with logits, batch size 8, fp32,
and 20 epochs. Each round reloads the same ImageNet source and freshly
initializes fusion, head, method, and optimizer state.

The configured active-learning schedule has six cumulative patient budgets at
10%, 15%, 20%, 25%, 30%, and 35% of the train patient pool. Validation and test
patients never enter the query pool.

The plugin declares support for Random, CoMAL, MM-CoMAL, MoDIS, and MoSAIC.
Image-level model outputs are aggregated to stable patient candidates before
selection. Compatibility is checked before the run starts.

## Evaluation

Per-label F1 thresholds are fitted on validation probabilities over the grid
`{0, 0.04, ..., 1}` and frozen for test evaluation. Reported outputs include:

- macro/micro AUROC and AUPRC;
- macro/micro F1;
- macro precision and recall;
- subset accuracy and Hamming loss;
- per-label support, threshold, AUROC, AUPRC, accuracy, precision, recall, F1,
  specificity, and negative predictive value.

Per-label support must accompany aggregate results because the task is strongly
long-tailed.

## Commands and artifacts

Prepare and validate with:

```bash
uv run al-mimic prepare \
  --task brset \
  --config configs/experiments/brset/random.yaml

uv run al-mimic validate-data \
  --task brset \
  --config configs/experiments/brset/random.yaml
```

Run one method with a matching config and explicit method:

```bash
uv run al-mimic active \
  --task brset \
  --method random \
  --config configs/experiments/brset/random.yaml
```

Available experiment configs cover Random, CoMAL, MM-CoMAL, MoDIS, and MoSAIC.
An active run writes per-round and final checkpoints, patient/image query
records in `active_state.json`, final metrics and predictions,
`resolved_config.json`, and `source_audit.json` below its experiment directory.
The `data_usage` record is authoritative because patient grouping makes image
counts vary with the selected patients.
