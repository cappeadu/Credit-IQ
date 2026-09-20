"""SHAP explanation helpers for the packaged selected estimator."""

from typing import Any

import numpy as np
import pandas as pd
import shap


def _positive_class_values(shap_values: Any) -> np.ndarray:
    """Normalise SHAP output to one value per feature for class 1."""
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[0, :, 1]
    if values.ndim == 2:
        return values[0]
    raise ValueError(f"Unsupported SHAP value shape: {values.shape}")


def _positive_class_base_value(base_values: Any) -> float:
    """Normalise a SHAP base value to the positive-class output."""
    values = np.asarray(base_values)
    if values.ndim == 0:
        return float(values)
    if values.ndim == 1:
        return float(values[1] if len(values) > 1 else values[0])
    return float(values[0, 1] if values.shape[-1] > 1 else values[0, 0])


def build_shap_explanations(
    selected_estimator: Any,
    feature_data: pd.DataFrame,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    """Return one explanation payload per row for the selected estimator.

    The selected estimator is intentionally used instead of the calibrated
    wrapper. SHAP describes the underlying model's feature contributions;
    probability and decision remain those of the calibrated runtime model.
    """
    if list(feature_data.columns) != feature_names:
        raise ValueError("Feature data columns do not match the model schema.")

    explainer = shap.TreeExplainer(selected_estimator)
    shap_result = explainer(feature_data)
    shap_values = np.asarray(shap_result.values)
    base_values = np.asarray(shap_result.base_values)

    explanations = []
    for row_index in range(len(feature_data)):
        row_values = shap_values[row_index]
        if row_values.ndim == 2:
            row_values = row_values[:, 1]
        elif row_values.ndim != 1:
            raise ValueError(f"Unsupported SHAP value shape: {shap_values.shape}")
        row_base_value = base_values[row_index]
        if np.asarray(row_base_value).ndim > 0:
            row_base_value = _positive_class_base_value(row_base_value)

        contributions = []
        for feature_name, feature_value, shap_value in zip(
            feature_names,
            feature_data.iloc[row_index].to_numpy(dtype=float),
            row_values,
        ):
            contribution = float(shap_value)
            contributions.append(
                {
                    "feature_name": feature_name,
                    "feature_value": float(feature_value),
                    "shap_value": contribution,
                    "direction": (
                        "increases_risk" if contribution > 0 else "decreases_risk"
                    ),
                }
            )
        explanations.append(
            {
                "base_value": float(row_base_value),
                "output_space": "model_output",
                "contributions": contributions,
            }
        )
    return explanations
