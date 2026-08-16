"""CoMAL method plugin, prototype module, scoring, and training helpers."""

from al_mimic.utils.contrastive import supervised_contrastive_loss
from al_mimic.utils.prototypes import (
    attach_prototype_outputs as attach_comal_outputs,
)
from al_mimic.utils.prototypes import (
    refresh_prototypes,
)

from .plugin import PLUGIN, CoMALPlugin, CoMALState
from .scoring import (
    AcquisitionComponents,
    PaperAcquisitionComponents,
    comal_acquisition_scores,
    paper_comal_acquisition_scores,
    positive_similarity_thresholds,
)
from .training import (
    CoMALModule,
    comal_training_loss,
    finalize_comal_outputs,
    refresh_comal_prototypes,
)

__all__ = [
    "AcquisitionComponents",
    "CoMALModule",
    "CoMALPlugin",
    "CoMALState",
    "PLUGIN",
    "PaperAcquisitionComponents",
    "attach_comal_outputs",
    "comal_acquisition_scores",
    "comal_training_loss",
    "finalize_comal_outputs",
    "paper_comal_acquisition_scores",
    "positive_similarity_thresholds",
    "refresh_comal_prototypes",
    "refresh_prototypes",
    "supervised_contrastive_loss",
]
