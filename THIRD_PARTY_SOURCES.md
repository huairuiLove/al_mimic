# Third-party sources and provenance

This file records conceptual and data provenance for the first-party
`src/al_mimic` implementation. It is not an installation manifest or an
execution guide.

## Boundary

Local upstream checkouts may be kept in an ignored `thirdparty/` directory for
manual, human-readable comparison. The entire directory is outside source
control. Production code and tests have zero runtime references to it: they do
not import modules from it, execute its scripts, or resolve data/config paths
through it.

Use the published URLs below to identify upstream work. Obtain credentialed
datasets from their official distributors and place data under `dataset/` as
documented in [dataset/README.md](dataset/README.md). There is no supported
production execution path through an upstream checkout.

## Methods

### CoMAL

- Concept: contrastive active learning for multi-label text classification.
- Source: https://github.com/JunW15/CoMAL
- Paper: https://aclanthology.org/2022.coling-1.323/
- Repository role: the first-party `al_mimic.methods.comal` implementation and
  its multimodal extension use CoMAL's label-prototype acquisition concepts.
- Reproduction boundary: task classifiers, multimodal representations, data
  protocols, and integration code are repository-native; results are not a
  claim that the original text-classification program was run unchanged.

### MM-CoMAL, MoDIS, and MoSAIC

These are repository-native multimodal acquisition methods or extensions built
on the shared plugin contract. MM-CoMAL extends label-prototype evidence across
modality and fused views. MoDIS uses modality probes, disagreement, and
intervention stability. MoSAIC combines Fisher-design screening with a
modality-coalition decomposition. Their implementation is under
`src/al_mimic/methods/`; no external checkout is executed.

## MIMIC-III sources

### Yang-Wu multimodal diagnosis

- Paper: Bo Yang and Lijun Wu, *How to Leverage Multimodal EHR Data for Better
  Medical Predictions?* https://aclanthology.org/2021.emnlp-main.329/
- Author code: https://github.com/emnlp-mimic/mimic
- FIDDLE paper/code: https://github.com/MLD3/FIDDLE
- ClinicalBERT: https://github.com/EmilyAlsentzer/clinicalBERT
- Repository role: cohort and feature contracts, BertEncoder architecture,
  FIDDLE-derived structured inputs, and ClinicalBERT initialization.

The executable local artifact used by recorded experiments has 10,258 visits,
915 labels, and 7,749 hourly features. The paper reports 10,210 visits, 1,042
labels, and 7,411 hourly features. Local runs must be identified as a rebuild
rather than an exact numeric reproduction of the paper artifact.

### MIMIC-III phenotyping

- 25-label benchmark: https://github.com/YerevaNN/mimic3-benchmarks
- Multimodal pretraining reference:
  https://github.com/kingrc15/multimodal-clinical-pretraining
- Multimodal pretraining paper: https://arxiv.org/abs/2312.06855
- 172-label paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC8378600/
- 172-label author code: https://github.com/amoldwin/notes_benchmark
- HCUP CCS overview: https://hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp

The task manifests retain reference revisions used during implementation:

| Source | Reference revision |
|---|---|
| `mimic3-benchmarks` | `ea0314c7cbd369f62e2237ace6f683740f867e3a` |
| `multimodal-clinical-pretraining` | `655c26a23880950cc270df5681b981e6869e26df` |
| `notes_benchmark` | `fa378b828fb1f832635c4259c3dff97ab81bd19d` |

These revisions are provenance metadata, not tracked submodules or runtime
dependencies. The first-party preprocessing modules and task runner implement
the actual executable path.

## BRSET sources

- Dataset v1.0.2: https://physionet.org/content/brazilian-ophthalmological/1.0.2/
- Data paper: https://doi.org/10.1371/journal.pdig.0000454
- 13-label image baseline:
  https://www.medrxiv.org/content/10.1101/2024.02.12.24302676v1
- Related metadata study:
  https://www.sciencedirect.com/science/article/pii/S1751991824000743

The repository's ResNet-50 plus pre-diagnostic metadata model is an explicit
13-label multimodal extension. It is not represented as a verbatim reproduction
of a published BRSET multimodal classifier.

## MDS-ED sources and MIT provenance

- MDS-ED paper: https://arxiv.org/abs/2407.17856
- MDS-ED code: https://github.com/AI4HealthUOL/MDS-ED
- MDS-ED release: https://physionet.org/content/multimodal-emergency-benchmark/1.0.0/
- MIMIC-IV-ECG: https://physionet.org/content/mimic-iv-ecg/1.0/
- Upstream copyright: Copyright (c) 2024 AI4HealthUOL.
- License: MIT.

The package-local preparation and task integration preserve MIT provenance in
`src/al_mimic/tasks/mds_ed/LICENSE` and `src/al_mimic/tasks/mds_ed/NOTICE`.

The package-local `NativeMdsEdTemporalAdapter` is a residual convolutional
temporal encoder fused with a tabular MLP. It preserves the broad four-layer
temporal plus tabular fusion contract, but it is **not a word-for-word,
operator-for-operator, or numerically equivalent implementation of the upstream
S4 benchmark**. It must be reported as the native temporal adapter, not as an
exact S4 benchmark reproduction. No local upstream MDS-ED checkout is imported
or executed during preparation or training.

## Data licenses

MIMIC releases and BRSET are distributed through PhysioNet under their own
credentialing and data-use terms. They are not redistributed by this repository.
Model checkpoints and derived artifacts may have additional upstream terms.
Users are responsible for obtaining access and preserving all applicable
licenses, citations, and privacy restrictions.
