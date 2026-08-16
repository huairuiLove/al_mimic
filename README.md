# al-mimic

`al-mimic` is a first-party Python package for multimodal active-learning
experiments on clinical prediction tasks. The maintained implementation lives
under `src/al_mimic`; installed commands, task plugins, method plugins, data
preparation, and experiment outputs all use that package.

The project has one public command-line entry point: `al-mimic`. Old root
scripts, legacy package commands, and commands executed from local upstream
checkouts are not part of the supported interface.

## Install and inspect

Python 3.10-3.13 is supported. From the repository root:

```bash
uv sync --dev
uv run al-mimic tasks
uv run al-mimic methods
uv run al-mimic capabilities
```

MDS-ED preparation additionally needs the optional data dependencies:

```bash
uv sync --dev --extra mds-ed
```

Every task command has the same shape:

```text
al-mimic ACTION --task TASK --config CONFIG [task-specific path options]
al-mimic active --task TASK --method METHOD --config CONFIG
```

`--method` is required for `active` and rejected for all other actions. It must
match `active_learning.strategy` in the resolved YAML.

## Current task status

The table separates implemented interfaces from artifacts observed in this
workspace on 2026-08-16. Credentialed datasets and generated outputs are ignored
by Git, so another checkout can have a different local-data status.

| Task plugin | Native task | Query unit | Implemented actions | Local status |
|---|---|---|---|---|
| `mimic_iii` | `icd9_diagnoses` (915 labels) | ICU stay | `prepare`, `validate-data`, `explore`, `active`, `full-data`, `visualize`, `hardware` | Yang-Wu/FIDDLE tensors, ClinicalBERT, and completed six-round Random/CoMAL/MM-CoMAL/MoDIS/MoSAIC outputs are present; the configured HDF5 target is currently a broken cross-host symlink |
| `mimic_iii` | `phenotyping_25` | ICU stay | same MIMIC-III actions | adapter and configs implemented; no task-specific processed HDF5, prepared manifest, or experiment output is present |
| `mimic_iii` | `phenotyping_ccs_172` | ICU stay | same MIMIC-III actions | adapter and configs implemented; no task-specific processed HDF5, prepared manifest, or experiment output is present |
| `brset` | BRSET v1.0.2, 13 labels | patient | `prepare`, `validate-data`, `active`, `full-data`, `hardware` | source release is present; prepared split and experiment outputs are not present |
| `mds_ed` | MDS-ED diagnoses, 1,428 labels | ECG study | `prepare`, `validate-data`, `train`, `hardware` | release CSV and ECG source are present; prepared memmap and supervised output are not present |

Task protocols and reproduction boundaries are detailed in:

- [MIMIC-III multimodal diagnosis](BASE_TASK_MIMIC_III_MULTIMODAL.md)
- [MIMIC-III phenotyping](BASE_TASK_MIMIC_III_PHENOTYPING.md)
- [BRSET multimodal diagnosis](BASE_TASK_BRSET_MULTIMODAL.md)
- [MIMIC-IV MDS-ED](BASE_TASK_MIMIC_IV_MDS_ED.md)

## Capability matrix

This is the compatibility declared by the task and method plugins. MDS-ED is
currently supervised-only and therefore accepts no acquisition method.

| Task | Random | CoMAL | MM-CoMAL | MoDIS | MoSAIC |
|---|:---:|:---:|:---:|:---:|:---:|
| `mimic_iii` | yes | yes | yes | yes | yes |
| `brset` | yes | yes | yes | yes | yes |
| `mds_ed` | no | no | no | no | no |

Method requirements are explicit:

| Method | Required task capabilities |
|---|---|
| `random` | candidate IDs and a query budget only |
| `comal` | multi-label probabilities and label prototypes |
| `mm_comal` | multi-label probabilities, modality tokens, and label prototypes |
| `modis` | multi-label probabilities, modality tokens, and token fusion |
| `mosaic` | multi-label probabilities, modality tokens, token fusion, and reference labels |

The CLI validates the task action, method allow-list, required capabilities, and
the method named in the resolved config before starting an experiment.

## Run examples

Validate and run one MIMIC-III experiment:

```bash
uv run al-mimic validate-data \
  --task mimic_iii \
  --config configs/experiments/mimic_iii/comal.yaml

uv run al-mimic active \
  --task mimic_iii \
  --method comal \
  --config configs/experiments/mimic_iii/comal.yaml
```

Prepare BRSET and run one declared method:

```bash
uv run al-mimic prepare \
  --task brset \
  --config configs/experiments/brset/random.yaml

uv run al-mimic active \
  --task brset \
  --method random \
  --config configs/experiments/brset/random.yaml
```

Prepare, validate, and train the package-local MDS-ED supervised adapter:

```bash
uv run al-mimic prepare \
  --task mds_ed \
  --config configs/experiments/mds_ed/diagnoses.yaml

uv run al-mimic validate-data \
  --task mds_ed \
  --config configs/experiments/mds_ed/diagnoses.yaml

uv run al-mimic train \
  --task mds_ed \
  --config configs/experiments/mds_ed/diagnoses.yaml
```

Build a metric matrix from completed experiment directories:

```bash
uv run al-mimic matrix \
  --experiment experiments/mimic_iii_yang_wu_bertencoder_random \
  --experiment experiments/mimic_iii_yang_wu_bertencoder_comal \
  --output experiments/evaluation_matrix.csv
```

## Configuration

Configuration is organized by responsibility:

```text
configs/
|-- tasks/          # dataset, model, protocol, and task-owned training defaults
|-- methods/        # acquisition-method overlays for a task family
|-- scenarios/      # MIMIC-III missing-modality and label-subset overlays
`-- experiments/    # runnable compositions passed to al-mimic
```

Use a file under `configs/experiments/` for normal runs. Its `extends` chain
resolves task and method definitions. Paths without a `./` prefix are resolved
from the repository working directory by the MIMIC-III loader; BRSET paths are
resolved from the runnable config location. Run commands from the repository
root and keep the checked-in config layout intact.

## Data and outputs

Local assets live under `dataset/`:

```text
dataset/raw/         credentialed source releases
dataset/processed/   derived tensors and intermediate tables
dataset/prepared/    validated manifests, splits, and MDS-ED memmaps
dataset/pretrained/  external model checkpoints
```

See [dataset/README.md](dataset/README.md) for task-specific inputs, preparation
outputs, and Git policy. Formal runs fail on missing or incompatible artifacts;
they do not replace clinical data with synthetic examples.

MIMIC-III and BRSET active/full-data runs write below the configured experiment
directory. The active runners produce `checkpoints/`, `active_state.json`,
`final_metrics.json`, `final_predictions.npz`, and `resolved_config.json`;
task-specific audit/provenance files may also be present. MIMIC-III `visualize`
writes `figures/`. MDS-ED preparation writes memmap and tabular manifests to its
prepared directory, while supervised training writes
`native_temporal_adapter.pt` and `training_summary.json`.

`experiments/` and generated dataset directories are local outputs and are not
tracked by Git. Do not infer benchmark completion from the presence of a config.

## Architecture and provenance

[ARCHITECTURE.md](ARCHITECTURE.md) defines the `tasks`, `methods`, `core`, and
`utils` boundaries and gives extension checklists. [THIRD_PARTY_SOURCES.md](THIRD_PARTY_SOURCES.md)
records conceptual sources, upstream URLs, and license provenance.

Any local `thirdparty/` directory is an ignored, manual-comparison workspace.
Production execution and tests have zero references to it. It is neither a data
directory nor a supported execution path.
