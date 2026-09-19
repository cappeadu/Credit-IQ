"""Reusable evaluation metrics for binary credit-risk predictions."""

from collections.abc import Sequence
from itertools import product

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

    if not (len(target_array) == len(probability_array) == len(class_array)):
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

    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        target_array, class_array, labels=[0, 1]
    ).ravel()
    specificity_denominator = true_negatives + false_positives
    specificity = (
        true_negatives / specificity_denominator if specificity_denominator else 0.0
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
        raise ValueError("Target values and probabilities must have the same length.")
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


def evaluate_threshold_policy(
    target_values: Sequence[int],
    predicted_probabilities: Sequence[float],
    *,
    lower_threshold: float,
    upper_threshold: float,
) -> dict[str, object]:
    """Evaluate the project's APPROVE/REVIEW/REJECT policy.

    ``REJECT`` is treated as the positive operational action when calculating
    binary metrics.  The returned group rates separately describe the three
    decisions so threshold effects remain visible.
    """
    target_array = np.asarray(target_values)
    probability_array = np.asarray(predicted_probabilities, dtype=float)
    if not 0 <= lower_threshold < upper_threshold <= 1:
        raise ValueError(
            "Thresholds must satisfy 0 <= lower_threshold < upper_threshold <= 1."
        )

    decisions = np.where(
        probability_array < lower_threshold,
        "APPROVE",
        np.where(probability_array < upper_threshold, "REVIEW", "REJECT"),
    )
    reject_predictions = (decisions == "REJECT").astype(int)
    binary_metrics = evaluate_binary_predictions(
        target_array,
        probability_array,
        reject_predictions,
    )

    total_count = len(target_array)
    default_count = int(np.sum(target_array == 1))
    non_default_count = int(np.sum(target_array == 0))
    decision_counts = {
        decision: int(np.sum(decisions == decision))
        for decision in ("APPROVE", "REVIEW", "REJECT")
    }
    decision_rates = {
        decision: round(count / total_count, 4)
        for decision, count in decision_counts.items()
    }
    default_rates_by_decision = {
        decision: round(
            float(np.mean(target_array[decisions == decision]))
            if decision_counts[decision]
            else 0.0,
            4,
        )
        for decision in ("APPROVE", "REVIEW", "REJECT")
    }

    return {
        "lower_threshold": lower_threshold,
        "upper_threshold": upper_threshold,
        "binary_metrics": binary_metrics,
        "decision_counts": decision_counts,
        "decision_rates": decision_rates,
        "default_rates_by_decision": default_rates_by_decision,
        "false_approval_rate": round(
            float(np.sum((decisions == "APPROVE") & (target_array == 1)))
            / default_count
            if default_count
            else 0.0,
            4,
        ),
        "false_rejection_rate": round(
            float(np.sum((decisions == "REJECT") & (target_array == 0)))
            / non_default_count
            if non_default_count
            else 0.0,
            4,
        ),
    }


def select_thresholds(
    target_values: Sequence[int],
    predicted_probabilities: Sequence[float],
    *,
    lower_thresholds: Sequence[float],
    upper_thresholds: Sequence[float],
    selection_metric: str = "recall",
    maximum_false_approval_rate: float | None = None,
) -> dict[str, object]:
    """Select a threshold pair from validation candidates.

    The default objective prioritises recall for the REJECT action.  A maximum
    false-approval constraint can be supplied when the project has an explicit
    risk requirement.  Ties prefer fewer false approvals, fewer rejections,
    then lower thresholds.  The function returns every evaluated candidate so
    the choice can be documented rather than treated as a hidden constant.
    """
    supported_metrics = {"recall", "precision", "f1_score", "specificity"}
    if selection_metric not in supported_metrics:
        raise ValueError(
            "selection_metric must be one of: " + ", ".join(sorted(supported_metrics))
        )
    if maximum_false_approval_rate is not None and not (
        0 <= maximum_false_approval_rate <= 1
    ):
        raise ValueError("maximum_false_approval_rate must be between 0 and 1.")

    evaluated_candidates = [
        evaluate_threshold_policy(
            target_values,
            predicted_probabilities,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )
        for lower_threshold, upper_threshold in product(
            lower_thresholds,
            upper_thresholds,
        )
        if lower_threshold < upper_threshold
    ]
    if not evaluated_candidates:
        raise ValueError("No valid lower and upper threshold pairs were supplied.")

    eligible_candidates = [
        candidate
        for candidate in evaluated_candidates
        if maximum_false_approval_rate is None
        or candidate["false_approval_rate"] <= maximum_false_approval_rate
    ]
    if not eligible_candidates:
        raise ValueError("No threshold candidates satisfy maximum_false_approval_rate.")

    selected_candidate = max(
        eligible_candidates,
        key=lambda candidate: (
            candidate["binary_metrics"][selection_metric],
            -candidate["false_approval_rate"],
            -candidate["decision_rates"]["REJECT"],
            -candidate["lower_threshold"],
            -candidate["upper_threshold"],
        ),
    )
    return {
        "selection_metric": selection_metric,
        "maximum_false_approval_rate": maximum_false_approval_rate,
        "selected_thresholds": {
            "lower_threshold": selected_candidate["lower_threshold"],
            "upper_threshold": selected_candidate["upper_threshold"],
        },
        "selected_evaluation": selected_candidate,
        "candidate_evaluations": evaluated_candidates,
    }
