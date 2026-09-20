"""Deterministic explanations for existing model decisions."""

from api.schemas import (
    DeterministicExplanationResponse,
    ExplanationRequest,
    FeatureContribution,
)

MAX_DISPLAYED_CONTRIBUTORS = 5
MATERIAL_SHAP_THRESHOLD = 0.01


def _format_contribution(contribution: FeatureContribution) -> str:
    """Format one contribution with enough detail for an auditable response."""
    return (
        f"{contribution.feature_name} "
        f"(value={contribution.feature_value:g}, "
        f"SHAP={contribution.shap_value:+.4f})"
    )


def generate_deterministic_explanation(
    explanation_request: ExplanationRequest,
) -> DeterministicExplanationResponse:
    """Build a reproducible explanation from the supplied SHAP context.

    A positive SHAP value is described as increasing this model's output and a
    negative value as decreasing it. Neither description is treated as a
    causal or real-world claim.
    """
    shap_context = explanation_request.prediction.explanation
    if shap_context is None:
        raise ValueError("Prediction explanation context is required.")

    material_contributions = [
        contribution
        for contribution in shap_context.contributions
        if abs(contribution.shap_value) >= MATERIAL_SHAP_THRESHOLD
    ]
    ranked_contributions = sorted(
        material_contributions,
        key=lambda contribution: abs(contribution.shap_value),
        reverse=True,
    )
    increasing = [
        _format_contribution(contribution)
        for contribution in ranked_contributions
        if contribution.shap_value > 0
    ][:MAX_DISPLAYED_CONTRIBUTORS]
    decreasing = [
        _format_contribution(contribution)
        for contribution in ranked_contributions
        if contribution.shap_value < 0
    ][:MAX_DISPLAYED_CONTRIBUTORS]

    prediction = explanation_request.prediction
    model_name = explanation_request.model_info.selected_model_name
    summary = (
        f"The {model_name} model assigned {prediction.decision} with a "
        f"predicted default probability of {prediction.probability:.3f}. "
        "The lists below show the strongest supplied model contributors."
    )
    limitations = [
        "SHAP values describe this model output; they do not prove causation.",
        "This is a model explanation, not financial advice or a new decision.",
        "Contributions with an absolute SHAP value below 0.01 are omitted.",
    ]
    return DeterministicExplanationResponse(
        summary=summary,
        increasing_contributors=increasing,
        decreasing_contributors=decreasing,
        limitations=limitations,
    )
