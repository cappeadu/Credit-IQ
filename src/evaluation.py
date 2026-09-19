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


def analyze_calibration(
    target_values: Sequence[int],
    predicted_probabilities: Sequence[float],
    *,
    number_of_bins: int = 10,
) -> dict[str, object]:
    """Return reliability-curve data and summary calibration metrics.

    Each calibration bin compares the average predicted probability with the
    observed positive rate.  A well-calibrated model has similar values in
    each populated bin.  This function analyses probabilities only; it does
    not select a decision threshold.
    """
    target_array = np.asarray(target_values)
    probability_array = np.asarray(predicted_probabilities, dtype=float)

    if len(target_array) != len(probability_array):
        raise ValueError(
            "Target values and probabilities must have the same length."
        )
    if len(target_array) == 0:
        raise ValueError("Cannot analyse calibration for empty predictions.")
    if number_of_bins < 2:
        raise ValueError("number_of_bins must be at least 2.")
    if not np.isfinite(probability_array).all() or not (
        (probability_array >= 0).all() and (probability_array <= 1).all()
    ):
        raise ValueError("Predicted probabilities must be finite values from 0 to 1.")

    bin_edges = np.linspace(0.0, 1.0, number_of_bins + 1)
    bin_indexes = np.clip(
        np.digitize(probability_array, bin_edges[1:-1], right=False),
        0,
        number_of_bins - 1,
    )

    calibration_bins: list[dict[str, float]] = []
    for bin_index in range(number_of_bins):
        bin_values = probability_array[bin_indexes == bin_index]
        bin_targets = target_array[bin_indexes == bin_index]
        bin_count = len(bin_values)
        if bin_count == 0:
            continue
        mean_probability = float(np.mean(bin_values))
        positive_rate = float(np.mean(bin_targets))
        calibration_bins.append(
            {
                "bin_lower_bound": round(float(bin_edges[bin_index]), 4),
                "bin_upper_bound": round(float(bin_edges[bin_index + 1]), 4),
                "mean_predicted_probability": round(float(mean_probability), 4),
                "observed_positive_rate": round(float(positive_rate), 4),
                "sample_count": float(bin_count),
            }
        )

    bin_weights = np.array(
        [calibration_bin["sample_count"] for calibration_bin in calibration_bins],
        dtype=float,
    )
    absolute_calibration_gaps = np.array(
        [
            abs(
                calibration_bin["mean_predicted_probability"]
                - calibration_bin["observed_positive_rate"]
            )
            for calibration_bin in calibration_bins
        ],
        dtype=float,
    )
    expected_calibration_error = (
        float(np.average(absolute_calibration_gaps, weights=bin_weights))
        if bin_weights.sum()
        else 0.0
    )

    return {
        "brier_score": round(
            brier_score_loss(target_array, probability_array),
            4,
        ),
        "expected_calibration_error": round(expected_calibration_error, 4),
        "number_of_bins": number_of_bins,
        "calibration_bins": calibration_bins,
    }
