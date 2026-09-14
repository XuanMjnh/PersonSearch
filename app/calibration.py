from __future__ import annotations

import numpy as np


def choose_threshold(
    positive_scores: list[float], negative_scores: list[float]
) -> dict[str, float | int]:
    """Choose the threshold with maximum F1; break ties toward fewer false alarms."""
    if not positive_scores or not negative_scores:
        raise ValueError("Cần cả mẫu cùng người và khác người để hiệu chỉnh")
    positives = np.asarray(positive_scores, dtype=np.float32)
    negatives = np.asarray(negative_scores, dtype=np.float32)
    best: dict[str, float | int] | None = None
    for threshold in np.arange(0.25, 0.951, 0.005):
        tp = int((positives >= threshold).sum())
        fn = int(len(positives) - tp)
        fp = int((negatives >= threshold).sum())
        tn = int(len(negatives) - fp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        candidate = {
            "threshold": round(float(threshold), 3),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "false_accept_rate": round(fp / max(fp + tn, 1), 4),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        }
        if best is None or (candidate["f1"], -candidate["false_positive"]) > (
            best["f1"], -best["false_positive"]
        ):
            best = candidate
    assert best is not None
    return best

