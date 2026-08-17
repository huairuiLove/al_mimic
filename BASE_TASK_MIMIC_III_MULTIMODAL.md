# Base task: MIMIC-III multimodal diagnosis

## Scope and status

This task is the `mimic_iii` plugin with native task ID `icd9_diagnoses`. One
query labels one ICU stay. The first-party implementation is under
`src/al_mimic/tasks/mimic_iii`, and all supported operations use `al-mimic`.

The current executable contract is a local Yang-Wu/FIDDLE rebuild, not an exact
copy of the artifact reported in the paper:

| Item | Paper reference | Current local contract |
|---|---:|---:|
| ICU visits | 10,210 | 10,258 |
| diagnosis labels | 1,042 | 915 |
| hourly feature width | 7,411 | 7,749 |
| time-invariant width | 97 | 97 |
| observation window | 48 hours | 48 hours |
| note length | 512 tokens | 512 tokens |

Completed six-round experiment directories for Random, CoMAL, MoDIS,
and MoSAIC are present locally under `experiments/`. They contain checkpoints,
state, final metrics, predictions, and resolved configs. They are ignored by Git.
The configured HDF5 target is currently a broken symbolic link to an absolute
path on another host. A fresh `validate-data` therefore fails in this workspace,
so the recorded outputs should not be treated as proof that the current data
layout is immediately rerunnable.

## Provenance and reproduction boundary

The task follows Yang and Wu, *How to Leverage Multimodal EHR Data for Better
Medical Predictions?*

- Paper: https://aclanthology.org/2021.emnlp-main.329/
- Author code: https://github.com/emnlp-mimic/mimic
- FIDDLE: https://github.com/MLD3/FIDDLE

The maintained classifier, loaders, metrics, and acquisition adapters are
package-local. Upstream work is a conceptual and protocol source; no external
source checkout is imported or executed. Because the local dimensions differ,
results must be described as the 915-label local rebuild, not as a numerical
reproduction of the 1,042-label paper task.

## Cohort, target, and inputs

The formal configuration uses MIMIC-III v1.4 MetaVision ICU stays, a 48-hour
observation window, and a fixed train/validation/test partition. The target is a
915-dimensional multi-hot vector of current-visit three-digit ICD-9 diagnosis
groups.

Every sample requires all three modalities:

1. Clinical notes within the 48-hour window, tokenized to 512 WordPiece tokens.
2. FIDDLE hourly time-series tensors with shape `[48, 7749]` in the local build.
3. A 97-dimensional FIDDLE time-invariant vector.

The HDF5 group is `with_notes/{train,val,test}`. It must contain `X`, `s`,
`input_ids`, `token_type_ids`, `attention_mask`, `label`, and a row-aligned
`SUBJECT_ID` or `subject_id`. MoDIS uses the subject identifier for grouped
out-of-fold probes; a global row index is not an acceptable substitute.

`al-mimic prepare` audits an already-built HDF5 and ClinicalBERT checkpoint and
writes a provenance manifest. It does not reconstruct FIDDLE tensors from raw
MIMIC tables. The final assembly step that combines the FIDDLE tensors, notes,
labels, and split assignments into `splits.hdf5` is
`al_mimic.tasks.mimic_iii.preprocessing.build_splits`; it also carries an
`--attach-only` mode that adds the subject-aligned arrays to an older artifact
after verifying its label row order.

## Classifier

`YangWuBertEncoderClassifier` implements the task-owned classifier:

- ClinicalBERT produces the note representation and is fine-tuned;
- a linear/ReLU path encodes the 97 time-invariant features;
- a projected three-layer, 16-head Transformer encodes the hourly features;
- a MAG-style gate fuses structured paths into the note representation;
- a linear head emits 915 independent logits.

Training uses BCE with logits and sigmoid probabilities. This follows the
multi-label objective rather than reproducing an upstream softmax-before-BCE
inconsistency. Each active-learning round starts from the same ClinicalBERT
source and freshly initializes the other model, method, and optimizer state.

## Active-learning protocol

The configured diagnosis experiments use nine cold-start rounds at cumulative
fractions 10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, and 50% of the actual train
pool. Counts use half-up rounding. Validation and test rows are fixed and never
queried.

The task plugin declares support for these registered methods:

| Method | Task-provided inputs |
|---|---|
| `random` | ICU-stay IDs and query budget |
| `comal` | fused probabilities and label-prototype evidence |
| `modis` | modality tokens, grouped probes, and token interventions |
| `modimix` | MoDIS-style acquisition inputs plus synchronized modality-space Mixup during classifier training |
| `mosaic` | modality coalitions, labeled outputs, and fixed validation reference outputs |

Every round has an 80-epoch ceiling, a 1,200 optimizer-step budget, and
validation-loss early stopping. No prior-round classifier or optimizer state is
inherited. Each round persists its final classifier checkpoint
(`checkpoints/round_XXX.pt`, plus `final.pt` for the last round) and resumable
loop state (`checkpoints/progress.json`); rerunning `al-mimic active` resumes
from the last completed round by default, and `--no-resume` forces a fresh run.

## Evaluation

The native metrics are Recall@10, Recall@20, and Recall@30. Recall is calculated
per ICU stay as the fraction of positive diagnoses found in the top-k scores and
then averaged. Recall@30 is the primary metric.

The paper's full-data values are references only. They are not repository
results and are not directly comparable to a 915-label local rebuild.

## Data and artifacts

The task config names:

- a processed split HDF5 under `dataset/processed/`;
- a ClinicalBERT checkpoint under `dataset/pretrained/`;
- a preparation manifest directory under `dataset/prepared/`;
- an experiment name below `experiments/`.

An active run writes:

- `checkpoints/round_000.pt` through `round_005.pt` and `checkpoints/final.pt`;
- `active_state.json` with round records, selected IDs, and data usage;
- `final_metrics.json` with validation/test recall values;
- `final_predictions.npz` with aligned final predictions and labels;
- `resolved_config.json`.

A full-data run writes equivalent final outputs below the experiment's
`full_data/` directory. A visualization run consumes `active_state.json` and
writes figures below `figures/`.

## Commands

Use runnable configs under `configs/experiments/mimic_iii/`:

```bash
uv run al-mimic validate-data \
  --task mimic_iii \
  --config configs/experiments/mimic_iii/random.yaml

uv run al-mimic active \
  --task mimic_iii \
  --method random \
  --config configs/experiments/mimic_iii/random.yaml

uv run al-mimic full-data \
  --task mimic_iii \
  --config configs/experiments/mimic_iii/full_cohort_random.yaml
```

Run `validate-data` successfully before starting a new training run. Fix the
local dataset/config path mismatch rather than pointing execution at an
upstream source directory.
