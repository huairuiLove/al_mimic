# Base tasks: MIMIC-III multimodal phenotyping

## Scope and status

The `mimic_iii` plugin includes two phenotyping task IDs alongside diagnosis:

| Task ID | Labels | Label definition | Primary metric | Runnable configs |
|---|---:|---|---|---|
| `phenotyping_25` | 25 | MIMIC-III Benchmark acute-care groups | macro-AUPRC | `configs/experiments/mimic_iii/phenotyping_25_*.yaml` |
| `phenotyping_ccs_172` | 172 | HCUP CCS 2015 groups represented in at least 30 episodes | macro-AUPRC | `configs/experiments/mimic_iii/phenotyping_ccs_172_*.yaml` |

One active-learning query is one ICU stay and returns its complete native
multi-hot vector. The first-party task specs, adapters, preprocessing modules,
model, metrics, and configs exist. In this workspace, neither task has its
configured processed HDF5, prepared manifest, or experiment output. These tasks
are implemented integration targets, not locally completed benchmarks.

## Provenance and reproduction boundary

The task definitions draw on:

- MIMIC-III Benchmark: https://github.com/YerevaNN/mimic3-benchmarks
- Multimodal clinical pretraining: https://arxiv.org/abs/2312.06855
- Its author code: https://github.com/kingrc15/multimodal-clinical-pretraining
- 172-label multimodal phenotyping paper:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8378600/
- Its author code: https://github.com/amoldwin/notes_benchmark

Reference revisions are recorded in [THIRD_PARTY_SOURCES.md](THIRD_PARTY_SOURCES.md).
They are provenance metadata, not runtime dependencies.

The released multimodal-pretraining downstream script and this active-learning
classifier are not identical. The package-local classifier keeps both note and
measurement representations so multimodal methods can inspect and intervene on
them. Results must not be described as a numerical reproduction of an upstream
downstream script.

The 172-label paper code also needs careful interpretation: its checked-in CCS
definitions mark the original 25 benchmark groups by default. The package-local
label builder enforces the paper's broader rule of CCS groups occurring in at
least 30 episodes and fails unless the result contains exactly 172 labels.

## Artifact contract

Both tasks consume one HDF5 with `with_notes/{train,val,test}`. Each row contains:

- hourly structured measurements and a time-series padding mask;
- chronological notes from the ICU stay, tokenized to 512 tokens;
- one native multi-hot label vector;
- stable `subject_id` and `stay_id` values;
- task and label-name metadata.

The 25-label task uses the official 76-feature one-hour discretization and
subject split. The 172-label task uses the notes-benchmark subject split and an
explicit 172-column CCS label table. The adapter validates split leakage, binary
labels, label width, feature width, and identifiers.

The prepared HDF5 is expected at the path in the selected task config:

```text
dataset/processed/mimic_phenotyping_25/splits.hdf5
dataset/processed/mimic_phenotyping_ccs_172/splits.hdf5
```

`al-mimic prepare` validates that already-created HDF5 and ClinicalBERT
checkpoint and writes `manifest.json` below the configured `dataset/prepared/`
directory. The public CLI does not currently expose raw-table-to-HDF5
construction as an action. Creating those HDF5 inputs is therefore a separate
first-party preprocessing step that must be completed before the documented
runtime commands can succeed.

## Model and training

The native classifier combines:

- ClinicalBERT notes;
- the 76-feature hourly measurement sequence;
- a three-layer structured Transformer with masked mean pooling;
- a fusion gate and independent sigmoid label head.

The 25-label head emits 25 logits; the CCS head emits 172. Training uses BCE with
logits and the same cold-start rule as the diagnosis runner: each round reloads
the same ClinicalBERT source and freshly initializes all other state.

The current phenotyping configs use six low-budget rounds with a 1% initial
fraction and 1% increments, ending at 6% of the train pool. This differs from
the 10%-35% schedule used by the MIMIC-III diagnosis and BRSET configs.

The task plugin declares Random, CoMAL, MM-CoMAL, MoDIS, and MoSAIC capability.
Checked-in runnable phenotyping configs currently cover Random and CoMAL only.
Other methods require a matching experiment config before they are runnable.

## Evaluation

Both tasks report macro-AUPRC, micro-AUPRC, macro-AUROC, and micro-AUROC.
Macro-AUPRC is primary. Recall@30 is not used as the primary phenotyping metric
and is not computed for the 25-label task merely to resemble diagnosis.

## Commands

After the configured HDF5 and ClinicalBERT files exist:

```bash
uv run al-mimic prepare \
  --task mimic_iii \
  --config configs/experiments/mimic_iii/phenotyping_25_random.yaml

uv run al-mimic validate-data \
  --task mimic_iii \
  --config configs/experiments/mimic_iii/phenotyping_25_random.yaml

uv run al-mimic active \
  --task mimic_iii \
  --method random \
  --config configs/experiments/mimic_iii/phenotyping_25_random.yaml
```

For 172 labels, use
`configs/experiments/mimic_iii/phenotyping_ccs_172_random.yaml`. CoMAL variants
use the corresponding `_comal.yaml` config and `--method comal`.

The output contract is the same as other MIMIC-III active runs:
`checkpoints/`, `active_state.json`, `final_metrics.json`,
`final_predictions.npz`, and `resolved_config.json` below the configured
experiment directory.
