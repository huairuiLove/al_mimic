from __future__ import annotations

import numpy as np

from mimic_comal.metrics import multilabel_metrics


def test_precision_prefixes_use_probability_order() -> None:
    labels = np.array([[1, 0, 0, 1, 0], [0, 1, 1, 0, 1]], dtype=np.int8)
    probabilities = np.array([[0.99, 0.7, 0.6, 0.8, 0.5], [0.4, 0.99, 0.9, 0.2, 0.8]], dtype=np.float32)
    metrics = multilabel_metrics(labels, probabilities)
    assert metrics["precision_at_1"] == 1.0
    assert np.isclose(metrics["precision_at_3"], 5.0 / 6.0)
