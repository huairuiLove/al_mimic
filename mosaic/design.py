"""Head-local block Fisher c-optimal design used as MoSAIC's scalar value."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .tokens import augment_bias


@dataclass
class FisherDesign:
    inverse: torch.Tensor
    reference_gradient: torch.Tensor
    direction: torch.Tensor
    information_trace: torch.Tensor
    damping: float

    @classmethod
    def build(
        cls,
        labeled_features: torch.Tensor,
        labeled_probabilities: torch.Tensor,
        reference_features: torch.Tensor,
        reference_probabilities: torch.Tensor,
        reference_labels: torch.Tensor,
        *,
        damping: float,
        closure_features: torch.Tensor | None = None,
        closure_probabilities: torch.Tensor | None = None,
    ) -> FisherDesign:
        if damping <= 0.0:
            raise ValueError("mosaic.damping must be positive")
        labeled = augment_bias(labeled_features.float())
        reference = augment_bias(reference_features.float())
        fisher_weights = labeled_probabilities.float() * (1.0 - labeled_probabilities.float())
        if closure_features is not None:
            if closure_probabilities is None:
                raise ValueError("closure probabilities are required with closure features")
            closure = augment_bias(closure_features.float())
            closure_weights = closure_probabilities.float() * (1.0 - closure_probabilities.float())
            labeled = torch.cat((labeled, closure), dim=0)
            fisher_weights = torch.cat((fisher_weights, closure_weights), dim=0)
        dimension = int(labeled.shape[1])
        identity = torch.eye(dimension, dtype=torch.float32, device=labeled.device)
        information = torch.einsum("nc,nd,ne->cde", fisher_weights, labeled, labeled)
        information = information + float(damping) * identity[None, :, :]
        cholesky, info = torch.linalg.cholesky_ex(information)
        if bool((info != 0).any()):
            raise RuntimeError("Fisher blocks are not positive definite; increase mosaic.damping")
        inverse = torch.cholesky_inverse(cholesky)
        residual = reference_probabilities.float() - reference_labels.float()
        reference_gradient = torch.einsum("nc,nd->cd", residual, reference) / max(
            int(reference.shape[0]), 1
        )
        direction = torch.einsum("cde,ce->cd", inverse, reference_gradient)
        return cls(inverse, reference_gradient, direction, information.diagonal(dim1=-2, dim2=-1).sum(-1), damping)

    def upper_bound(self, features: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        augmented = augment_bias(features.float())
        variance = probabilities.float() * (1.0 - probabilities.float())
        alignment = torch.einsum("nd,cd->nc", augmented, self.direction)
        norm = augmented.square().sum(dim=1, keepdim=True)
        denominator = variance.clamp_min(1e-8).reciprocal() + norm / self.information_trace[None, :]
        return (alignment.square() / denominator).sum(dim=1)

    def marginal_gain(self, features: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        augmented = augment_bias(features.float())
        variance = probabilities.float() * (1.0 - probabilities.float())
        alignment = torch.einsum("nd,cd->nc", augmented, self.direction)
        coverage = torch.einsum("nd,cde,ne->nc", augmented, self.inverse, augmented)
        denominator = variance.clamp_min(1e-8).reciprocal() + coverage
        return (alignment.square() / denominator).sum(dim=1)

    def deflate(self, feature: torch.Tensor, probabilities: torch.Tensor) -> None:
        augmented = augment_bias(feature.float().reshape(1, -1))[0]
        variance = probabilities.float() * (1.0 - probabilities.float())
        projected = torch.einsum("cde,e->cd", self.inverse, augmented)
        coverage = torch.einsum("cd,d->c", projected, augmented)
        coefficient = variance / (1.0 + variance * coverage)
        self.inverse -= coefficient[:, None, None] * torch.einsum(
            "cd,ce->cde", projected, projected
        )
        self.direction = torch.einsum("cde,ce->cd", self.inverse, self.reference_gradient)
        self.information_trace += variance * augmented.square().sum()
