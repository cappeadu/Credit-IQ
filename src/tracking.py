"""Small local MLflow integration for reproducible training runs."""

from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import joblib
import mlflow

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI


def configure_mlflow(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> None:
    """Configure and verify the user-managed MLflow tracking server."""
    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        raise RuntimeError(
            "Unable to connect to MLflow at "
            f"{tracking_uri}. Start the MLflow server before training."
        ) from exc


def flatten_comparison_metrics(
    comparison_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    """Convert model comparison metrics into MLflow-safe metric keys."""
    flattened_metrics: dict[str, float] = {}
    for model_name, model_metrics in comparison_results.items():
        metric_prefix = model_name.lower().replace(" ", "_")
        for metric_name, metric_value in model_metrics.items():
            if metric_name == "model_name":
                continue
            flattened_metrics[f"{metric_prefix}.{metric_name}"] = float(metric_value)
    return flattened_metrics


def _stringify_model_parameters(model: Any) -> dict[str, str]:
    """Convert estimator parameters into values accepted by MLflow."""
    return {
        parameter_name: str(parameter_value)
        for parameter_name, parameter_value in model.get_params(
            deep=True
        ).items()
    }


def _log_model_metrics(model_metrics: Mapping[str, Any]) -> None:
    mlflow.log_metrics(
        {
            metric_name: float(metric_value)
            for metric_name, metric_value in model_metrics.items()
            if metric_name != "model_name"
        }
    )


def log_baseline_comparison_run(
    *,
    fitted_candidate_models: Mapping[str, Any],
    comparison_results: Mapping[str, Mapping[str, Any]],
    selected_model_name: str,
    selection_metrics: tuple[str, ...],
    training_parameters: Mapping[str, Any],
    artifact_paths: Mapping[str, Path],
) -> str:
    """Log a parent comparison run and one nested run per candidate model."""
    configure_mlflow()
    with mlflow.start_run(run_name="baseline-model-comparison") as parent_run:
        mlflow.log_params(
            {
                parameter_name: str(parameter_value)
                for parameter_name, parameter_value in training_parameters.items()
            }
        )
        mlflow.log_param("selected_model", selected_model_name)
        mlflow.log_param("selection_metrics", " > ".join(selection_metrics))
        mlflow.log_metrics(flatten_comparison_metrics(comparison_results))
        mlflow.set_tag("stage", "baseline-model-comparison")
        mlflow.set_tag("selected_model", selected_model_name)

        for model_name, fitted_model in fitted_candidate_models.items():
            with mlflow.start_run(
                run_name=model_name,
                nested=True,
            ):
                mlflow.set_tag("run_type", "candidate-model")
                mlflow.set_tag("candidate_model", model_name)
                mlflow.log_params(_stringify_model_parameters(fitted_model))
                _log_model_metrics(comparison_results[model_name])

                with TemporaryDirectory() as temporary_directory:
                    candidate_model_path = (
                        Path(temporary_directory) / "model.joblib"
                    )
                    joblib.dump(fitted_model, candidate_model_path)
                    mlflow.log_artifact(
                        str(candidate_model_path),
                        artifact_path="fitted_model",
                    )

        for artifact_name, artifact_path in artifact_paths.items():
            if artifact_path.exists():
                mlflow.log_artifact(
                    str(artifact_path),
                    artifact_path=artifact_name,
                )

        return parent_run.info.run_id
