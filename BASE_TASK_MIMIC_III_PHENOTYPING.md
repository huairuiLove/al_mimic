# MIMIC-III native multi-label phenotyping tasks

This repository registers two phenotyping tasks alongside the existing ICD-9
diagnosis task. In every task, one active-learning query is one ICU stay and the
annotation returned by that query is one native multi-hot vector.

## Task definitions

| Task id | Labels | Label source | Primary metric | Config |
|---|---:|---|---|---|
| `phenotyping_25` | 25 | MIMIC-III Benchmark acute-care groups | macro-AUPRC | `configs/mimic_phenotyping_25.yaml` |
| `phenotyping_ccs_172` | 172 | HCUP CCS 2015 groups occurring in at least 30 episodes | macro-AUPRC | `configs/mimic_phenotyping_ccs_172.yaml` |
| `icd9_diagnoses` | 915 | Three-digit ICD-9 groups in the local Yang-Wu rebuild | Recall@30 | `configs/mimic_comal.yaml` |

The phenotype tasks also report micro-AUPRC, macro-AUROC, and micro-AUROC.
Recall@30 is deliberately not computed for the 25-label task.

## Fixed upstream sources

The author repositories are checked out as Git submodules:

| Source | Pinned revision | Purpose |
|---|---|---|
| `third_party/mimic3-benchmarks` | `ea0314c7cbd369f62e2237ace6f683740f867e3a` | 25-label definitions, patient split, 76-feature structured preprocessing |
| `third_party/multimodal-clinical-pretraining` | `655c26a23880950cc270df5681b981e6869e26df` | Published multimodal pretraining and phenotyping reference implementation |
| `third_party/notes_benchmark` | `fa378b828fb1f832635c4259c3dff97ab81bd19d` | 172-label paper implementation and HCUP CCS definitions |

Initialize them after cloning this repository:

```bash
git submodule update --init --recursive
```

MIMIC data, derived listfiles, HDF5 files, note indexes, and model weights are
not stored in Git.

The task definitions are grounded in the
[MIMIC-III Benchmark](https://github.com/YerevaNN/mimic3-benchmarks), the
[multimodal pretraining paper](https://arxiv.org/abs/2312.06855) and its
[author code](https://github.com/kingrc15/multimodal-clinical-pretraining), and
the [172-label multimodal phenotyping paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8378600/)
with its [author code](https://github.com/amoldwin/notes_benchmark).

The released 2023 downstream script disables its notes branch and fine-tunes
only the measurement encoder. It is therefore pinned here as a published
pretraining/architecture reference, not as an exact executable downstream
baseline. The runner-native phenotype classifier instead uses ClinicalBERT
notes, the official 76-feature hourly measurements, a structured Transformer,
and a fusion gate. Keeping both modality representations is necessary for
multimodal active acquisition. Results from this runner must not be described
as a numerical reproduction of that upstream downstream script.

## Important 172-label upstream detail

The `notes_benchmark` repository is the code URL named by the 172-label paper,
but its checked-in `hcup_ccs_2015_definitions.yaml` still marks only the original
25 benchmark groups with `use_in_benchmark: True`. Running its default
`create_phenotyping.py` therefore produces 25 labels, not 172.

The paper defines the broader target as CCS groups represented by at least 30
episodes. Build that label matrix from the author's filtered cohort tables with:

```bash
uv run python scripts/build_ccs_172_labels.py \
  --stays-csv /path/to/notes_benchmark/root/all_stays.csv \
  --diagnoses-csv /path/to/notes_benchmark/root/all_diagnoses.csv \
  --output data/mimic_phenotyping_ccs_172/labels.csv
```

The command fails unless the rule yields exactly 172 labels. It writes label
names, positive counts, source revision, and threshold to a JSON manifest.

## Unified multimodal artifact

First run the upstream benchmark preprocessing and patient split. The resulting
phenotyping directory must contain `train_listfile.csv`, `val_listfile.csv`,
`test_listfile.csv`, and the corresponding per-stay time-series CSV files. The
subject root must retain each `episode*.csv` file so the adapter can recover the
true `ICUSTAY_ID` rather than treating a row index as a patient identifier.

Build the 25-label HDF5:

```bash
uv run python scripts/build_mimic_phenotyping_hdf5.py \
  --task phenotyping_25 \
  --task-root /path/to/mimic3-benchmark/phenotyping \
  --subject-root /path/to/mimic3-benchmark/root \
  --mimic-root /path/to/mimic-iii-clinical-database-1.4 \
  --tokenizer /path/to/Bio_ClinicalBERT \
  --output data/mimic_phenotyping_25/splits.hdf5
```

Build the 172-label HDF5 using the explicit label table:

```bash
uv run python scripts/build_mimic_phenotyping_hdf5.py \
  --task phenotyping_ccs_172 \
  --task-root /path/to/notes-benchmark/phenotyping \
  --subject-root /path/to/notes-benchmark/root \
  --mimic-root /path/to/mimic-iii-clinical-database-1.4 \
  --tokenizer /path/to/Bio_ClinicalBERT \
  --ccs-labels-csv data/mimic_phenotyping_ccs_172/labels.csv \
  --output data/mimic_phenotyping_ccs_172/splits.hdf5
```

The adapter uses the official one-hour discretizer and normalizer, keeps the
first 256 structured timesteps, concatenates notes charted during the same ICU
stay in chronological order, tokenizes to 512 tokens, and removes rows without
an ICU-window note. It records `subject_id`, `stay_id`, label names, a time-series
padding mask, and the task id in the artifact. Validation rejects split leakage,
wrong label widths, non-binary labels, and feature dimension drift.

## Run

```bash
uv run python main.py tasks --config configs/mimic_phenotyping_25.yaml
uv run python main.py validate-data --config configs/mimic_phenotyping_25.yaml
uv run python main.py active --config configs/mimic_phenotyping_25.yaml

uv run python main.py validate-data --config configs/mimic_phenotyping_ccs_172.yaml
uv run python main.py active --config configs/mimic_phenotyping_ccs_172.yaml
```

Random baselines are available in `configs/mimic_phenotyping_25_random.yaml`
and `configs/mimic_phenotyping_ccs_172_random.yaml`.
