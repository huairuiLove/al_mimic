# Architecture

## Runtime surface

The maintained package is `src/al_mimic`. Packaging exposes one console script:

```text
al-mimic -> al_mimic.cli:main
```

`al_mimic.cli` parses an action, resolves a task plugin, validates task/method
capabilities, asks the task to load its config, and delegates execution to that
task. Acquisition logic is loaded only for `active`. The `matrix` command is a
repository-level evaluation operation and does not load a task.

```text
CLI
  -> task registry -> selected task plugin -> task runner/data/model
  -> method registry -> selected method plugin -> acquisition result
  -> capability checks and shared contracts
  -> local dataset and experiment artifacts
```

Local upstream source checkouts are not in this graph.

## Package boundaries

### `al_mimic.tasks`

Tasks own everything that changes with a dataset or prediction problem:

- source discovery, schema audit, preprocessing, and prepared-data loading;
- label definition, split protocol, query-unit semantics, and metrics;
- classifier architecture and task-specific training loops;
- adaptation from task rows to canonical acquisition contexts;
- supported actions, supported methods, and provided capabilities;
- task-specific checkpoints, predictions, manifests, and summaries.

Each task family exports one `PLUGIN` through `al_mimic.tasks.registry`.
`mimic_iii` contains three native task IDs selected by `task.id` in YAML;
`brset` and `mds_ed` each expose one current task protocol.

Task code may depend on `core`, method public APIs, and `utils`. A task must not
import another task's runner, model, data loader, or private helpers.

### `al_mimic.methods`

Methods own task-independent acquisition behavior:

- method-specific fitting state and auxiliary modules;
- scoring, interventions, prototype estimation, and selection;
- required task capabilities and canonical context fields;
- an `acquire` implementation returning stable IDs and candidate-relative
  positions through `AcquisitionResult`.

The method registry stores lazy import paths for `random`, `comal`,
`modis`, `modimix`, and `mosaic`. A method must not import a concrete task. It receives all
task data through `AcquisitionContext` or its prepared mapping.

### `al_mimic.core`

Core defines stable orchestration policy rather than dataset logic:

- `contracts.py`: task capabilities, acquisition context, and plugin protocols;
- `capabilities.py`: action/method compatibility checks;
- `engine.py`: initial selection and acquisition-result validation;
- `methods.py`: method context preparation and contract validation;
- `artifacts.py`: generic experiment layout and provenance manifest helpers.

Core must not import concrete task or method implementations. Registries and
the CLI are the composition layer.

### `al_mimic.utils`

Utils contains small, reusable mechanisms with no task ownership: IO,
contrastive/prototype helpers, fusion and linear operations, schedules, and
runtime controls. A utility must not select a task, method, dataset path, label
set, or experiment protocol.

### `al_mimic.evaluation`

Evaluation consumes completed artifacts. The current matrix helper reads
`final_metrics.json` from one or more experiment directories and writes a flat
CSV comparison. It does not train models or participate in acquisition.

## Contracts and flow

1. `al-mimic` selects a task name from the lazy task registry.
2. The CLI checks that the requested action is declared by the task.
3. The task loader resolves and validates the YAML, including `task.family`.
4. For `active`, the CLI requires an explicit method, compares it with
   `active_learning.strategy`, loads the method plugin, and checks capabilities.
5. The task runner trains its classifier and creates row- or group-level model
   outputs required by the selected method.
6. The task adapter constructs canonical candidate IDs, query size, outputs,
   labels/groups, method state, seed, round index, and metadata.
7. The method returns selected IDs and candidate-relative positions. Core
   verifies exact budget, uniqueness, and ID/position agreement.
8. The task owns state transitions, fixed validation/test evaluation, and all
   task-specific artifacts.

This keeps acquisition ranking reusable without pretending that ICU stays,
patients with paired eyes, and ECG studies have interchangeable data loaders.

## Capability model

Task plugins declare features such as:

- `multilabel_probabilities`
- `modality_tokens`
- `token_fusion`
- `label_prototypes`
- `reference_labels`

Method plugins declare only the features they need. Compatibility is the
intersection of the task's method allow-list and these required features. The
current declared matrix is:

| Task | Random | CoMAL | MoDIS | ModiMix | MoSAIC |
|---|:---:|:---:|:---:|:---:|:---:|
| `mimic_iii` | yes | yes | yes | yes | yes |
| `brset` | yes | yes | yes | no | yes |
| `mds_ed` | no | no | no | no | no |

`mds_ed` is deliberately `supervised_only` today. Adding method names to a
config does not make an unsupported combination valid.

## Configuration ownership

- `configs/tasks/<task>/`: task protocol and task-owned defaults.
- `configs/methods/<task>/<method>.yaml`: method overlay compatible with that
  task family.
- `configs/scenarios/`: controlled scenario overlays.
- `configs/experiments/<task>/`: runnable compositions.

The runnable file is passed unchanged to `al-mimic`; task loaders resolve
`extends`, validate strict protocol fields, and retain `_config_path` internally.
A method name appears both in CLI input and resolved YAML so accidental
cross-method runs fail before training.

## Artifact ownership

Source assets and derived data stay under `dataset/`; model runs stay under
`experiments/`. Tasks write their own detailed schemas, but completed active
runs share these consumer-facing files:

- `active_state.json`: round histories, queries, metrics, and data usage;
- `final_metrics.json`: final validation/test metrics and method identity;
- `final_predictions.npz`: aligned final labels, probabilities, and IDs;
- `resolved_config.json`: the effective protocol;
- `checkpoints/`: per-round and final model states.

Prepared-data manifests bind generated artifacts to source paths, dimensions,
splits, and task definitions. These generated directories are ignored by Git.

## Add a task

1. Create `src/al_mimic/tasks/<task>/` with data audit/loading, config loading,
   metrics, model/training, and a plugin.
2. Give the plugin a stable `task_id`, actions, query unit, capabilities,
   supported methods, `load_config`, and `execute`.
3. Convert the native query unit to stable candidate IDs. Preserve group
   boundaries and keep validation/test examples outside the query pool.
4. Populate every canonical context field needed by each advertised method.
   Do not advertise a method until the adapter satisfies its full contract.
5. Add the lazy import path to `al_mimic.tasks.registry`.
6. Add strict task config under `configs/tasks/` and runnable experiment configs
   under `configs/experiments/`.
7. Test registry loading, capabilities, invalid combinations, data leakage,
   action dispatch, acquisition ID mapping, and artifact schemas.
8. Document data licensing, preparation, metrics, reproduction boundaries, and
   the real local/executable status.

## Add a method

1. Create `src/al_mimic/methods/<method>/` without imports from concrete tasks.
2. Export a `PLUGIN` with stable `method_id`, display name,
   `required_capabilities`, `required_context_fields`, and `acquire`.
3. Put method fitting and prepared-context logic in the plugin when needed;
   keep classifier training and query-unit aggregation in tasks.
4. Return an `AcquisitionResult` with exactly the requested number of unique
   candidate IDs and matching candidate-relative positions.
5. Add the lazy import path to `al_mimic.methods.registry`.
6. Add task-specific method overlays only for adapters that satisfy the method
   contract, then add the method to those task plugins' allow-lists.
7. Test deterministic selection, budget edges, contract failures, lazy imports,
   and at least one integration path for every advertised task.
8. Update the capability matrix and explain any methodological provenance.

## Third-party boundary

Concepts and protocol details may be reimplemented from published work, with
licenses and URLs recorded in [THIRD_PARTY_SOURCES.md](THIRD_PARTY_SOURCES.md).
The production package and tests must not import, execute, or construct paths
into a local `thirdparty/` checkout. Such a directory may exist only for manual
human comparison and is ignored as a whole.
