"""Task- and method-agnostic supervised contrastive objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _contrastive_chunk(
    flat: torch.Tensor,
    class_ids: torch.Tensor,
    start: int,
    stop: int,
    temperature: float,
    view_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk = flat[start:stop]
    width = stop - start
    logits = chunk.matmul(flat.transpose(0, 1)).div_(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    positive = class_ids[start:stop, None].eq(class_ids[None, :])
    eligible = None
    if view_ids is not None:
        eligible = view_ids[start:stop, None].ne(view_ids[None, :])
        positive = positive & eligible

    rows = torch.arange(width, device=flat.device)
    self_columns = start + rows
    if start == 0 and stop == int(flat.shape[0]):
        positive.fill_diagonal_(False)
        denominator_logits = logits.clone()
        if eligible is None:
            denominator_logits.fill_diagonal_(float("-inf"))
        else:
            denominator_logits.masked_fill_(~eligible, float("-inf"))
    else:
        self_mask = torch.zeros_like(positive)
        self_mask[rows, self_columns] = True
        positive = positive & ~self_mask
        excluded = self_mask if eligible is None else self_mask | ~eligible
        denominator_logits = logits.masked_fill(excluded, float("-inf"))

    denominator = torch.logsumexp(denominator_logits, dim=1, keepdim=True)
    denominator = torch.where(
        torch.isfinite(denominator),
        denominator,
        torch.zeros_like(denominator),
    )
    log_probability = logits - denominator
    positive_weights = positive.to(dtype=log_probability.dtype)
    counts = positive_weights.sum(dim=1)
    positive_log_probability = torch.where(
        positive,
        log_probability,
        torch.zeros_like(log_probability),
    )
    per_anchor = -positive_log_probability.sum(dim=1) / counts.clamp_min(1)
    active = counts > 0
    selected = torch.where(active, per_anchor, torch.zeros_like(per_anchor))
    return selected.sum(), active.sum()


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.07,
    anchor_chunk_size: int = 1024,
    view_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a memory-bounded label-wise supervised contrastive loss.

    Positive labels form one class per label column. All negative label entries
    share one background class. Optional ``view_ids`` restrict positives and the
    softmax denominator to cross-view pairs.
    """
    if features.ndim != 3 or labels.shape != features.shape[:2]:
        raise ValueError("expected latent features [B,L,D] and labels [B,L]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if anchor_chunk_size < 1:
        raise ValueError("anchor_chunk_size must be positive")

    batch_size, num_labels, feature_dim = features.shape
    flat = F.normalize(features, dim=-1).reshape(batch_size * num_labels, feature_dim)
    flat_view_ids = None
    if view_ids is not None:
        if view_ids.shape != labels.shape:
            raise ValueError("view_ids must have the same shape as labels")
        flat_view_ids = view_ids.reshape(-1)
    label_ids = torch.arange(num_labels, device=labels.device).expand(batch_size, -1)
    class_ids = torch.where(labels >= 0.5, label_ids, num_labels).reshape(-1)
    total = int(flat.shape[0])
    if total < 2:
        return features.sum() * 0

    bytes_needed = total * total * flat.element_size()
    max_full_bytes = 2 * 1024**3 if flat.is_cuda else 512 * 1024**2
    if bytes_needed <= max_full_bytes:
        loss_sum, loss_count = _contrastive_chunk(
            flat,
            class_ids,
            0,
            total,
            float(temperature),
            flat_view_ids,
        )
        return torch.where(
            loss_count > 0,
            loss_sum / loss_count.clamp_min(1),
            features.sum() * 0,
        )

    loss_sum = flat.new_zeros(())
    loss_count = flat.new_zeros(())
    for start in range(0, total, int(anchor_chunk_size)):
        stop = min(start + int(anchor_chunk_size), total)
        part_sum, part_count = _contrastive_chunk(
            flat,
            class_ids,
            start,
            stop,
            float(temperature),
            flat_view_ids,
        )
        loss_sum = loss_sum + part_sum
        loss_count = loss_count + part_count
    return torch.where(
        loss_count > 0,
        loss_sum / loss_count.clamp_min(1),
        features.sum() * 0,
    )


def sample_label_dimensions(
    latent: torch.Tensor,
    labels: torch.Tensor,
    maximum_labels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample label dimensions while retaining observed positives first."""
    if labels.ndim != 2 or latent.shape[-2] != labels.shape[1]:
        raise ValueError("latent label dimension must match labels [B,L]")
    label_count = int(labels.shape[1])
    maximum = min(max(1, int(maximum_labels)), label_count)
    if maximum == label_count:
        return latent, labels
    positive = torch.nonzero(labels.any(dim=0), as_tuple=False).flatten()
    if positive.numel() >= maximum:
        selected = positive[torch.randperm(positive.numel(), device=labels.device)[:maximum]]
    else:
        mask = torch.ones(label_count, dtype=torch.bool, device=labels.device)
        mask[positive] = False
        negative = torch.nonzero(mask, as_tuple=False).flatten()
        needed = maximum - int(positive.numel())
        negative = negative[torch.randperm(negative.numel(), device=labels.device)[:needed]]
        selected = torch.cat((positive, negative))
    selected = selected.sort().values
    return latent.index_select(-2, selected), labels.index_select(-1, selected)


def multiview_contrastive_loss(
    latent: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.07,
    anchor_chunk_size: int = 1024,
    cross_view_weight: float = 0.0,
) -> torch.Tensor:
    """Combine per-view and cross-view supervised contrastive losses."""
    if latent.ndim == 3:
        return supervised_contrastive_loss(
            latent,
            labels,
            temperature=temperature,
            anchor_chunk_size=anchor_chunk_size,
        )
    if latent.ndim != 4 or latent.shape[0] != labels.shape[0]:
        raise ValueError("latent must have shape [B,L,D] or [B,V,L,D]")
    if cross_view_weight < 0:
        raise ValueError("cross_view_weight must be non-negative")
    views = int(latent.shape[1])
    within = torch.stack(
        [
            supervised_contrastive_loss(
                latent[:, view],
                labels,
                temperature=temperature,
                anchor_chunk_size=anchor_chunk_size,
            )
            for view in range(views)
        ]
    ).mean()
    if cross_view_weight == 0:
        return within
    expanded_labels = (
        labels[:, None, :]
        .expand(-1, views, -1)
        .reshape(
            labels.shape[0] * views,
            labels.shape[1],
        )
    )
    view_ids = (
        torch.arange(views, device=labels.device)[None, :, None]
        .expand(
            labels.shape[0],
            -1,
            labels.shape[1],
        )
        .reshape(labels.shape[0] * views, labels.shape[1])
    )
    cross = supervised_contrastive_loss(
        latent.reshape(
            labels.shape[0] * views,
            labels.shape[1],
            latent.shape[-1],
        ),
        expanded_labels,
        temperature=temperature,
        anchor_chunk_size=anchor_chunk_size,
        view_ids=view_ids,
    )
    return within + float(cross_view_weight) * cross


__all__ = [
    "multiview_contrastive_loss",
    "sample_label_dimensions",
    "supervised_contrastive_loss",
]
