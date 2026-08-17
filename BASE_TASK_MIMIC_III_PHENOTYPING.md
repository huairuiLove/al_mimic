# Base tasks: MIMIC-III multimodal phenotyping

## Scope and status

The `mimic_iii` plugin includes two phenotyping task IDs alongside diagnosis:

| Task ID | Labels | Label definition | Primary metric | Runnable configs |
|---|---:|---|---|---|
| `phenotyping_25` | 25 | MIMIC-III Benchmark acute-care groups | macro-AUPRC | `configs/experiments/mimic_iii/phenotyping_25_*.yaml` |
| `phenotyping_ccs_239` | 239 | HCUP CCS 2015 groups occurring in >=30 episodes of the final cohort | macro-AUPRC | `configs/experiments/mimic_iii/phenotyping_ccs_239_*.yaml` |

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
definitions mark the original 25 benchmark groups by default, and its released
code selects only those. The paper itself states one selection rule -- CCS
groups occurring in at least 30 episodes -- and reports 172 phenotypes
without ever enumerating them. That count is not reproducible from any
published rule: the >=30-episode rule selects 239 CCS groups on this
repository's final cohort (243 across the full 42,276-stay benchmark
population). The package-local label builder therefore commits to the stated
rule and its actual yield, and fails unless the result contains exactly 239
labels.

## Artifact contract

Both tasks consume one HDF5 with `with_notes/{train,val,test}`. Each row contains:

- hourly structured measurements and a time-series padding mask;
- chronological notes from the ICU stay, tokenized to 512 tokens;
- one native multi-hot label vector;
- stable `subject_id` and `stay_id` values;
- task and label-name metadata.

The 25-label task uses the official 76-feature one-hour discretization and
subject split. The 239-label task uses the notes-benchmark subject split and an
explicit 239-column CCS label table. The adapter validates split leakage, binary
labels, label width, feature width, and identifiers.

The prepared HDF5 is expected at the path in the selected task config:

```text
dataset/processed/mimic_phenotyping_25/splits.hdf5
dataset/processed/mimic_phenotyping_ccs_239/splits.hdf5
```

`al-mimic prepare` validates an already-created HDF5 and ClinicalBERT
checkpoint and writes `manifest.json` below the configured `dataset/prepared/`
directory. The raw-table construction is implemented by the first-party
modules under `src/al_mimic/tasks/mimic_iii/preprocessing/`; it does not execute
any checkout under `thirdparty/`. The pipeline is documented in
[`dataset/README.md`](dataset/README.md): build the benchmark cohort, extract
chronological notes, stream the 17 structured event variables into the official
76-channel representation, and assemble `splits.hdf5`.

The 239-label builder applies the >=30-episode rule directly and refuses to
continue unless the final cohort selects exactly 239 CCS groups. This makes a
cohort mismatch visible before an experiment starts.

## Model and training

The native classifier combines:

- ClinicalBERT notes;
- the 76-feature hourly measurement sequence;
- a three-layer structured Transformer with masked mean pooling;
- a fusion gate and independent sigmoid label head.

The 25-label head emits 25 logits; the CCS head emits 239. Training uses BCE with
logits and the same cold-start rule as the diagnosis runner: each round reloads
the same ClinicalBERT source and freshly initializes all other state.

The current phenotyping configs use six low-budget rounds with a 1% initial
fraction and 1% increments, ending at 6% of the train pool. This differs from
the 10%-35% schedule used by the MIMIC-III diagnosis and BRSET configs.

The task plugin declares Random, CoMAL, MoDIS, and MoSAIC capability.
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

For 239 labels, use
`configs/experiments/mimic_iii/phenotyping_ccs_239_random.yaml`. CoMAL variants
use the corresponding `_comal.yaml` config and `--method comal`.

The output contract is the same as other MIMIC-III active runs:
`checkpoints/`, `active_state.json`, `final_metrics.json`,
`final_predictions.npz`, and `resolved_config.json` below the configured
experiment directory.
