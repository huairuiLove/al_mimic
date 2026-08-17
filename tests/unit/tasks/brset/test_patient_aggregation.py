from __future__ import annotations

import torch
import torch.nn as nn

from al_mimic.tasks.brset.training import aggregate_patient_outputs


class SumFusionClassifier(nn.Module):
    modality_names = ("fundus_image", "clinical_metadata")
    feature_dim = 2

    def fuse_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.sum(dim=1)

    def probabilities_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(fused)


def test_image_outputs_are_aggregated_to_unique_patient_query_rows() -> None:
    classifier = SumFusionClassifier()
    outputs = {
        "labels": torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        "probabilities": torch.full((5, 2), 0.123),
        "features": torch.full((5, 2), -99.0),
        "modality_tokens": torch.tensor(
            [
                [[4.0, 0.0], [0.0, 2.0]],
                [[1.0, 0.0], [0.0, 1.0]],
                [[3.0, 0.0], [0.0, 3.0]],
                [[2.0, 0.0], [0.0, 4.0]],
                [[5.0, 1.0], [1.0, 5.0]],
            ]
        ),
    }

    aggregated, patient_ids = aggregate_patient_outputs(
        classifier,
        outputs,
        ("p2", "p1", "p1", "p2", "p3"),
        patient_ids=("p1", "p2", "p3"),
    )

    expected_tokens = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 2.0]],
            [[3.0, 0.0], [0.0, 3.0]],
            [[5.0, 1.0], [1.0, 5.0]],
        ]
    )
    expected_features = expected_tokens.sum(dim=1)
    assert patient_ids == ("p1", "p2", "p3")
    assert torch.equal(aggregated["image_counts"], torch.tensor([2, 2, 1]))
    assert torch.equal(aggregated["labels"], torch.tensor([[1.0, 1.0]] * 3))
    assert torch.equal(aggregated["modality_tokens"], expected_tokens)
    assert torch.equal(aggregated["features"], expected_features)
    assert torch.allclose(aggregated["probabilities"], torch.sigmoid(expected_features))
    assert not torch.allclose(aggregated["probabilities"], torch.full((3, 2), 0.123))


def test_patient_aggregation_rejects_duplicate_requested_ids() -> None:
    classifier = SumFusionClassifier()
    outputs = {
        "labels": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "modality_tokens": torch.zeros(2, 2, 2),
    }

    try:
        aggregate_patient_outputs(
            classifier,
            outputs,
            ("p1", "p2"),
            patient_ids=("p1", "p1"),
        )
    except ValueError as error:
        assert "unique patient IDs" in str(error)
    else:
        raise AssertionError("duplicate patient IDs must be rejected")
