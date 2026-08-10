# Base task: MIMIC-III multimodal multi-label diagnosis prediction

## 1. Baseline source

The common classifier for every active-learning method is **BertEncoder** from:

- Bo Yang and Lijun Wu. *How to Leverage Multimodal EHR Data for Better Medical Predictions?*
  EMNLP 2021: https://aclanthology.org/2021.emnlp-main.329/
- Official implementation: https://github.com/emnlp-mimic/mimic


## 2. Cohort and prediction target

The formal task is the paper's **Diagnoses** task, not CAML full-label coding:

| Item | Formal setting |
|---|---|
| Database | MIMIC-III v1.4 |
| Source system | MetaVision, 2008-2012 |
| Observation window | First 48 hours of an ICU visit |
| Samples after filtering | 10,210 ICU visits |
| Split | Official 7:1.5:1.5 train/validation/test partition |
| Target | Current-visit diagnoses |
| Labels | 1,042 three-digit ICD-9 diagnosis groups |
| Target representation | Multi-hot vector |

The exact train/validation/test row counts are read from the official
`splits.hdf5/with_notes` artifact. They are not estimated or replaced with a new
random split. Their sum must equal 10,210 or preparation fails.

## 3. Input modalities

All three paper modalities are mandatory:

1. **Clinical notes**: notes charted in `[0, 48]` hours. The latest note for each
   `CATEGORY/DESCRIPTION` group is concatenated and tokenized to 512 WordPiece
   tokens by the official preprocessing code.
2. **Time-series data**: FIDDLE hourly features with shape `[48, 7411]`, including
   vital signs, laboratory measurements, medications, and missingness/value
   encodings produced by FIDDLE.
3. **Time-invariant data**: FIDDLE vector with 97 dimensions.

Only `splits.hdf5` group `with_notes` is accepted. Visits shorter than 48 hours,
incorrect notes, and visits without notes have already been excluded by the
official preprocessing protocol.

## 4. Classifier

The classifier is the paper's best diagnosis model, **BertEncoder**:

- ClinicalBERT/BERT-base encodes the clinical note into 768 dimensions and is
  fully fine-tuned.
- A linear layer plus ReLU maps 97 time-invariant inputs to 64 dimensions.
- A linear projection and a 3-layer, 16-head Transformer encoder map the 7,411
  hourly features to a 1,024-dimensional time-series representation.
- The clinical-note representation is the main modality. A MAG-style gate uses
  the other two modalities to adjust it.
- A linear head emits 1,042 diagnosis logits.
- Dropout is `0.1`, Adam learning rate is `1e-4`, the ClinicalBERT scheduler uses
  10% linear warmup/decay, and gradient clipping is `1.0`.

The paper defines independent binary cross entropy for this multi-label task.
Therefore, this implementation trains with `BCEWithLogitsLoss` and uses sigmoid
probabilities. The released upstream code applies a softmax before BCE, which is
inconsistent with both Equation 7 and multi-label semantics; that coding
inconsistency is deliberately not reproduced.

## 5. Evaluation

Validation and test report exactly the diagnosis metrics in the paper:

- Recall@10
- Recall@20
- Recall@30

Recall is computed per ICU visit as the fraction of its positive diagnoses found
in the top-k predictions, then averaged across visits. AUROC, AUPRC, F1, and
Precision@k are not formal result columns for this task.

The paper's full-data reference values for BertEncoder are Recall@30 `0.587`,
Recall@20 `0.490`, and Recall@10 `0.334`. These are reference values, not results
claimed by this repository.

## 6. Active-learning protocol

Each method uses the same official training pool and the same seeded initial
indices. The labeled schedule is calculated from the **actual official train
split size**:

| Round | Cumulative labeled fraction |
|---:|---:|
| 0 | 10% |
| 1 | 15% |
| 2 | 20% |
| 3 | 25% |
| 4 | 30% |
| 5 | 35% |

Counts use half-up rounding. The run output records the exact schedule, initial
count, queried count, final labeled count, validation count, test count, and
total 10,210-visit cohort size.

For every round:

1. Reload the same ClinicalBERT source checkpoint.
2. Reinitialize the static encoder, time-series encoder, fusion gate, diagnosis
   head, acquisition auxiliaries, and optimizer.
3. Train the classifier for exactly 20 epochs.
4. Never load a prior-round checkpoint or inherit optimizer state.
5. Evaluate on the fixed validation and test splits.
6. Query 5% of the original train pool unless this is the final round.

There are no smoke, dry-run, row-limit, batch-limit, or shortened-epoch formal
configurations.

## 7. Four acquisition methods

All four methods receive the same classifier outputs. For modality-aware methods,
the frozen classifier exposes three 768-dimensional path-contribution tokens
(notes, time series, time invariant) whose sum is exactly its fused feature.

- **CoMAL**: one prototype per positive diagnosis group plus one shared negative
  background prototype, so `1,042 + 1` prototypes.
- **MM-CoMAL**: separate prototype banks for notes, time series, time invariant,
  and fused views, so `4 x (1,042 + 1)` prototypes; reliability-weighted evidence
  and cross-view dispersion determine acquisition.
- **MoDIS**: stop-gradient per-modality probes, grouped out-of-fold reliability,
  modality disagreement, intervention instability, and a sufficiency penalty.
- **MoSAIC**: Fisher-design screening, eight coalitions over three modalities,
  on-manifold token interventions, Mobius additive/synergy decomposition, and
  greedy Fisher deflation.

These are acquisition strategies. None may change the baseline classifier,
cohort, labels, split, epoch count, or evaluation metrics.

## 8. Required private artifacts

Formal execution requires:

1. The official Yang-Wu/FIDDLE diagnosis artifact at
   `features/outcome=Diagnoses,T=48.0,dt=1.0/splits.hdf5`.
2. Group `with_notes` containing `train`, `val`, and `test` with arrays `X`, `s`,
   `input_ids`, `token_type_ids`, `attention_mask`, and `label`.
3. The ClinicalBERT BERT-base state dictionary used as the common source
   initialization.
4. CUDA hardware capable of training the approximately 150M-parameter model.

The repository fails loudly if these private inputs or their exact dimensions
are missing. It never substitutes synthetic tensors or a smaller model.
