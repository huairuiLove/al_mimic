"""Uniform random acquisition with no task or sibling-method dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from al_mimic.methods.api import (
    AcquisitionResult,
    acquisition_result,
    context_field,
    validate_query_size,
)

RandomAcquisitionResult = AcquisitionResult


@dataclass(frozen=True)
class RandomPlugin:
    """Duck-typed uniform random acquisition plugin."""

    method_id: str = "random"
    display_name: str = "Random"
    required_capabilities: tuple[str, ...] = ()
    required_context_fields: tuple[str, ...] = ("candidate_ids", "query_size")

    def acquire(self, context: Any = None, **fields: Any) -> RandomAcquisitionResult:
        candidates: Sequence[Any] = context_field(context, fields, "candidate_ids")
        query_size = validate_query_size(context_field(context, fields, "query_size"), len(candidates))
        seed = int(context_field(context, fields, "seed", 0)) + int(
            context_field(context, fields, "round_index", 0)
        )
        generator = np.random.default_rng(seed)
        positions = generator.choice(len(candidates), size=query_size, replace=False)
        return acquisition_result(
            self.method_id,
            candidates,
            positions,
            diagnostics={"seed": seed},
        )

    __call__ = acquire


PLUGIN = RandomPlugin()


__all__ = ["PLUGIN", "RandomAcquisitionResult", "RandomPlugin"]
