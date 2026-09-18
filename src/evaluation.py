"""Reusable evaluation metrics for binary credit-risk predictions."""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_predictions(
    target_values: Sequence[int],
    predicted_probabilities: Sequence[float],
    predicted_classes: Sequence[int],
) -> dict[str, float]:
    """Return consistent ranking, classification, and calibration metrics.

    The class predictions must be produced using the threshold currently being
    evaluated.  This function does not select or change thresholds.
    """
    target_array = np.asarray(target_values)
    probability_array = np.asarray(predicted_probabilities, dtype=float)
    class_array = np.asarray(predicted_classes)

    if not (
        len(target_array) == len(probability_array) == len(class_array)
    ):
        raise ValueError(
            "Target values, probabilities, and predicted classes must have "
            "the same length."
        )
    if len(target_array) == 0:
        raise ValueError("Cannot evaluate an empty set of predictions.")
    if not np.isfinite(probability_array).all() or not (
        (probability_array >= 0).all() and (probability_array <= 1).all()
    ):
        raise ValueError("Predicted probabilities must be finite values from 0 to 1.")

    true_negatives, false_positives, false_negatives, true_positives = (
        confusion_matrix(target_array, class_array, labels=[0, 1]).ravel()
    )
    specificity_denominator = true_negatives + false_positives
    specificity = (
        true_negatives / specificity_denominator
        if specificity_denominator
        else 0.0
    )

    return {
        "roc_auc_score": round(roc_auc_score(target_array, probability_array), 4),
        "pr_auc_score": round(
            average_precision_score(target_array, probability_array),
            4,
        ),
        "accuracy": round(accuracy_score(target_array, class_array), 4),
        "recall": round(
            recall_score(target_array, class_array, zero_division=0),
            4,
        ),
        "precision": round(
            precision_score(target_array, class_array, zero_division=0),
            4,
        ),
        "f1_score": round(
            f1_score(target_array, class_array, zero_division=0),
            4,
        ),
        "specificity": round(specificity, 4),
        "brier_score": round(
            brier_score_loss(target_array, probability_array),
            4,
        ),
        "true_negatives": float(true_negatives),
        "false_positives": float(false_positives),
        "false_negatives": float(false_negatives),
        "true_positives": float(true_positives),
    }
