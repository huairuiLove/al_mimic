from __future__ import annotations

import numpy as np

from mimic_comal.metrics import multilabel_metrics, top_k_recall


def test_yang_wu_metrics_are_per_visit_top_k_recall() -> None:
    labels = np.zeros((2, 40), dtype=np.int8)
    labels[0, [0, 1, 20, 21]] = 1
    labels[1, [2, 3]] = 1
    probabilities = np.arange(40, 0, -1, dtype=np.float32)[None].repeat(2, axis=0)
    probabilities[1, 2] = 100
    probabilities[1, 3] = 99
    metrics = top_k_recall(labels, probabilities, (10, 20, 30))
    assert metrics["recall_at_10"] == 0.75
    assert metrics["recall_at_20"] == 0.75
    assert metrics["recall_at_30"] == 1.0
    assert multilabel_metrics(labels, probabilities) == metrics
    assert set(metrics) == {"recall_at_10", "recall_at_20", "recall_at_30"}
