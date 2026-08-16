"""Yang-Wu multimodal classifiers for MIMIC-III tasks."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class YangWuGate(nn.Module):
    """Clinical-note-primary multimodal adaptation gate from Yang and Wu."""

    def __init__(
        self,
        text_dim: int,
        time_invariant_dim: int,
        time_series_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.has_static = time_invariant_dim > 0
        self.static_weight = nn.Linear(text_dim + time_invariant_dim, 1) if self.has_static else None
        self.time_series_weight = nn.Linear(text_dim + time_series_dim, 1)
        self.adjustment = nn.Linear(time_invariant_dim + time_series_dim, text_dim)
        self.beta = nn.Parameter(torch.randn(1))
        self.norm = nn.LayerNorm(text_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text: torch.Tensor,
        time_invariant: torch.Tensor | None,
        time_series: torch.Tensor,
    ) -> torch.Tensor:
        series_gate = torch.sigmoid(self.time_series_weight(torch.cat((text, time_series), dim=-1)))
        parts: list[torch.Tensor] = []
        if self.has_static:
            if time_invariant is None or self.static_weight is None:
                raise ValueError("the configured fusion gate requires time-invariant features")
            static_gate = torch.sigmoid(self.static_weight(torch.cat((text, time_invariant), dim=-1)))
            parts.append(static_gate * time_invariant)
        elif time_invariant is not None and time_invariant.shape[-1] != 0:
            raise ValueError("time-invariant features were supplied to a two-modality gate")
        parts.append(series_gate * time_series)
        adjustment = self.adjustment(torch.cat(parts, dim=-1))
        ratio = text.norm() / adjustment.norm().clamp_min(1e-12)
        alpha = torch.minimum(ratio * self.beta, adjustment.new_ones(()))
        return self.dropout(self.norm(text + alpha * adjustment))


class YangWuBertEncoderClassifier(nn.Module):
    """ClinicalBERT plus structured Transformer for MIMIC-III multi-label tasks.

    The three-modality form preserves the Yang-Wu ICD-9 baseline. The two-modality
    form is used by acute-care phenotyping tasks. ``modality_tokens`` exposes an
    exact telescoping decomposition of the fused representation for method layers.
    """

    modality_names = ("clinical_notes", "time_series", "time_invariant")

    def __init__(
        self,
        clinicalbert_checkpoint: str | None,
        *,
        num_labels: int = 915,
        time_invariant_dim: int = 97,
        time_invariant_hidden_dim: int = 64,
        time_series_dim: int = 7749,
        time_series_hidden_dim: int = 1024,
        time_series_layers: int = 3,
        time_series_heads: int = 16,
        text_hidden_dim: int = 768,
        dropout: float = 0.1,
        time_series_pooling: str = "first",
        text_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if time_series_hidden_dim % time_series_heads:
            raise ValueError("time-series hidden dimension must be divisible by attention heads")
        self.text_encoder = text_encoder or self._load_clinicalbert(clinicalbert_checkpoint)
        if (time_invariant_dim == 0) != (time_invariant_hidden_dim == 0):
            raise ValueError("time-invariant input and hidden dimensions must both be zero or positive")
        self.time_invariant_encoder = (
            nn.Linear(time_invariant_dim, time_invariant_hidden_dim) if time_invariant_dim > 0 else None
        )
        self.time_series_projection = nn.Linear(time_series_dim, time_series_hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=time_series_hidden_dim,
            nhead=time_series_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.time_series_encoder = nn.TransformerEncoder(encoder_layer, num_layers=time_series_layers)
        self.gate = YangWuGate(
            text_hidden_dim,
            time_invariant_hidden_dim,
            time_series_hidden_dim,
            dropout,
        )
        self.classifier = nn.Linear(text_hidden_dim, num_labels)
        self.feature_dim = text_hidden_dim
        self.num_labels = num_labels
        self.time_series_pooling = str(time_series_pooling)
        if self.time_series_pooling not in {"first", "masked_mean"}:
            raise ValueError("time_series_pooling must be 'first' or 'masked_mean'")
        self.modality_names = (
            ("clinical_notes", "time_series", "time_invariant")
            if self.time_invariant_encoder is not None
            else ("clinical_notes", "time_series")
        )

    @staticmethod
    def _load_clinicalbert(checkpoint: str | None) -> nn.Module:
        if not checkpoint:
            raise ValueError("a ClinicalBERT checkpoint is required")
        try:
            from transformers import BertConfig, BertModel
        except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
            raise RuntimeError("transformers is required for the Yang-Wu baseline") from exc
        checkpoint_path = Path(checkpoint)
        config_dir = checkpoint_path.parent if checkpoint_path.is_file() else checkpoint_path
        config_file = config_dir / "config.json"
        if config_file.is_file():
            encoder = BertModel(BertConfig.from_pretrained(config_dir))
        else:
            encoder = BertModel(BertConfig())
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - old torch compatibility
            state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError("ClinicalBERT checkpoint must contain a state dictionary")
        model_state = encoder.state_dict()
        compatible = {
            str(key).removeprefix("bert."): value
            for key, value in state.items()
            if str(key).removeprefix("bert.") in model_state
            and model_state[str(key).removeprefix("bert.")].shape == value.shape
        }
        coverage = sum(value.numel() for value in compatible.values()) / sum(
            value.numel() for value in model_state.values()
        )
        if coverage < 0.95:
            raise ValueError(f"ClinicalBERT checkpoint covers only {coverage:.1%} of BERT-base parameters")
        model_state.update(compatible)
        encoder.load_state_dict(model_state)
        return encoder

    def _encode_text(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.text_encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if isinstance(output, (tuple, list)) and len(output) > 1:
            return output[1]
        raise ValueError("ClinicalBERT encoder must return a pooled BERT-base representation")

    def encode_modalities(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        static = (
            F.relu(self.time_invariant_encoder(batch["time_invariant"].float()))
            if self.time_invariant_encoder is not None
            else None
        )
        series = F.relu(self.time_series_projection(batch["time_series"].float()))
        mask = batch.get("time_series_mask")
        available = None
        if mask is None:
            padding_mask = None
        else:
            mask = mask.bool()
            available = mask.any(dim=1)
            safe_mask = mask.clone()
            safe_mask[~available, 0] = True
            padding_mask = ~safe_mask
        encoded_series = self.time_series_encoder(series, src_key_padding_mask=padding_mask)
        if self.time_series_pooling == "masked_mean":
            if mask is None:
                mask = torch.ones(encoded_series.shape[:2], dtype=torch.bool, device=encoded_series.device)
            weights = mask.to(encoded_series.dtype).unsqueeze(-1)
            series = (encoded_series * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        else:
            series = encoded_series[:, 0]
        explicit_available = batch.get("time_series_available")
        if explicit_available is not None:
            explicit_available = explicit_available.bool()
            available = explicit_available if available is None else available & explicit_available
        if available is not None:
            series = torch.where(available[:, None], series, torch.zeros_like(series))
        text = self._encode_text(
            batch["input_ids"].long(),
            batch["token_type_ids"].long(),
            batch["attention_mask"].long(),
        )
        return text, series, static

    def _decompose(
        self,
        text: torch.Tensor,
        series: torch.Tensor,
        static: torch.Tensor | None,
        fused: torch.Tensor,
    ) -> torch.Tensor:
        zeros_series = torch.zeros_like(series)
        text_only = self.gate(
            text,
            torch.zeros_like(static) if static is not None else None,
            zeros_series,
        )
        if static is None:
            return torch.stack((text_only, fused - text_only), dim=1)
        text_static = self.gate(text, static, zeros_series)
        return torch.stack((text_only, fused - text_static, text_static - text_only), dim=1)

    def fuse_from_tokens(self, tokens: torch.Tensor, *, apply_dropout: bool = False) -> torch.Tensor:
        del apply_dropout
        expected = (len(self.modality_names), self.feature_dim)
        if tokens.ndim != 3 or tokens.shape[1:] != expected:
            raise ValueError(f"tokens must have shape [N, {expected[0]}, {expected[1]}]")
        return tokens.sum(dim=1)

    def probabilities_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.classifier(fused))

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        text, series, static = self.encode_modalities(batch)
        fused = self.gate(text, static, series)
        logits = self.classifier(fused)
        result = {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "features": fused,
        }
        if return_tokens:
            result["modality_tokens"] = self._decompose(text, series, static, fused)
        return result
