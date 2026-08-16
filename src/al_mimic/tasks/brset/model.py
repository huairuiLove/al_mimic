"""ResNet-50 and clinical-metadata fusion for 13-label BRSET diagnosis."""

from __future__ import annotations

import torch
import torch.nn as nn


class BrsetMultimodalClassifier(nn.Module):
    modality_names = ("fundus_image", "clinical_metadata")

    def __init__(
        self,
        metadata_dim: int,
        *,
        num_labels: int = 13,
        image_feature_dim: int = 2048,
        metadata_hidden_dim: int = 128,
        fusion_dim: int = 512,
        dropout: float = 0.2,
        image_weights: str = "IMAGENET1K_V2",
        image_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if min(metadata_dim, num_labels, image_feature_dim, metadata_hidden_dim, fusion_dim) < 1:
            raise ValueError("BRSET model dimensions must be positive")
        self.image_encoder = image_encoder or self._load_resnet50(image_weights)
        self.image_projection = nn.Sequential(
            nn.Linear(image_feature_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
        )
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, metadata_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(metadata_hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
        )
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, num_labels)
        self.feature_dim = fusion_dim
        self.num_labels = num_labels

    @staticmethod
    def _load_resnet50(image_weights: str) -> nn.Module:
        try:
            from torchvision.models import ResNet50_Weights, resnet50
        except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
            raise RuntimeError("torchvision is required for the BRSET ResNet-50 base") from exc
        if image_weights != "IMAGENET1K_V2":
            raise ValueError("formal BRSET training requires ResNet50 IMAGENET1K_V2 weights")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Identity()
        return model

    def encode_modalities(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        image_features = self.image_encoder(batch["image"].float())
        if image_features.ndim != 2:
            image_features = image_features.flatten(1)
        image_token = self.image_projection(image_features)
        metadata_token = self.metadata_encoder(batch["metadata"].float())
        return torch.stack((image_token, metadata_token), dim=1)

    def fuse_from_tokens(self, tokens: torch.Tensor, *, apply_dropout: bool = False) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (2, self.feature_dim):
            raise ValueError(f"tokens must have shape [N, 2, {self.feature_dim}]")
        fused = self.fusion_norm(tokens.sum(dim=1))
        return self.fusion_dropout(fused) if apply_dropout else fused

    def probabilities_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.classifier(fused))

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        tokens = self.encode_modalities(batch)
        fused = self.fuse_from_tokens(tokens, apply_dropout=self.training)
        logits = self.classifier(fused)
        result = {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "features": fused,
        }
        if return_tokens:
            result["modality_tokens"] = tokens
        return result


def initialize_fusion_layers(model: BrsetMultimodalClassifier) -> None:
    """Initialize every non-ImageNet layer identically at the start of each round."""
    for module in (
        model.image_projection,
        model.metadata_encoder,
        model.classifier,
    ):
        for child in module.modules():
            if isinstance(child, nn.Linear):
                nn.init.xavier_uniform_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
