from __future__ import annotations

import torch

from mimic_comal.multimodal_training import _sample_contrastive_labels


def test_full_label_contrastive_sampling_keeps_batch_positives() -> None:
    labels = torch.zeros(4, 1042)
    positive_columns = torch.tensor([2, 101, 400, 1041])
    labels[torch.arange(4), positive_columns] = 1
    latents = torch.randn(4, 1042, 8)
    sampled_latents, sampled_labels = _sample_contrastive_labels(latents, labels, 256)
    assert sampled_latents.shape == (4, 256, 8)
    assert sampled_labels.shape == (4, 256)
    assert int(sampled_labels.sum()) == 4
