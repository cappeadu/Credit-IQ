"""Build an immutable, self-contained model package for local deployment."""

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib

from src.splitting import SPLIT_MANIFEST_VERSION

PACKAGE_VERSION = 1
PACKAGE_ARTIFACTS = (
    "calibrated_model.joblib",
    "selected_estimator.joblib",
    "feature_engineering.joblib",
    "thresholds.json",
    "feature_schema.json",
    "split_manifest.json",
)


def create_model_package(
    *,
    package_directory: str | Path,
    calibrated_model_path: Path,
    selected_estimator_path: Path,
    feature_engineering_path: Path,
    thresholds_path: Path,
    feature_schema_path: Path,
    split_manifest_path: Path,
    metadata: Mapping[str, Any],
) -> Path:
    """Copy runtime artifacts and write package metadata.

    The package directory is expected to be unique to one MLflow run. Existing
    files are overwritten only inside that explicitly supplied package path.
    """
    package_path = Path(package_directory)
    source_artifacts = {
        "calibrated_model.joblib": calibrated_model_path,
        "selected_estimator.joblib": selected_estimator_path,
        "feature_engineering.joblib": feature_engineering_path,
        "thresholds.json": thresholds_path,
        "feature_schema.json": feature_schema_path,
        "split_manifest.json": split_manifest_path,
    }
    missing_artifacts = [
        str(path) for path in source_artifacts.values() if not path.exists()
    ]
    if missing_artifacts:
        raise FileNotFoundError(
            "Cannot create model package; missing artifacts: "
            + ", ".join(missing_artifacts)
        )

    package_path.mkdir(parents=True, exist_ok=True)
    for package_name, source_path in source_artifacts.items():
        shutil.copy2(source_path, package_path / package_name)

    package_metadata = {
        "package_version": PACKAGE_VERSION,
        "artifacts": list(source_artifacts),
        **dict(metadata),
    }
    metadata_path = package_path / "package_metadata.json"
    metadata_path.write_text(json.dumps(package_metadata, indent=2))
    return package_path


def validate_model_package(package_directory: str | Path) -> dict[str, Any]:
    """Validate package structure and return its metadata."""
    package_path = Path(package_directory)
    if not package_path.is_dir():
        raise FileNotFoundError(
            f"Model package directory was not found: {package_path}"
        )

    metadata_path = package_path / "package_metadata.json"
    if not metadata_path.exists():
        raise ValueError("Model package is missing package_metadata.json.")
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("Model package metadata is not valid JSON.") from exc

    if metadata.get("package_version") != PACKAGE_VERSION:
        raise ValueError("Unsupported model package version.")
    if tuple(metadata.get("artifacts", ())) != PACKAGE_ARTIFACTS:
        raise ValueError("Model package artifact manifest is incomplete or invalid.")
    if not metadata.get("mlflow_run_id"):
        raise ValueError("Model package is missing mlflow_run_id metadata.")

    missing_artifacts = [
        artifact_name
        for artifact_name in PACKAGE_ARTIFACTS
        if not (package_path / artifact_name).exists()
    ]
    if missing_artifacts:
        raise FileNotFoundError(
            "Model package is missing artifacts: " + ", ".join(missing_artifacts)
        )

    thresholds = json.loads((package_path / "thresholds.json").read_text())
    lower_threshold = float(thresholds["lower_threshold"])
    upper_threshold = float(thresholds["upper_threshold"])
    if not 0 <= lower_threshold < upper_threshold <= 1:
        raise ValueError("Model package thresholds are invalid.")

    feature_schema = json.loads((package_path / "feature_schema.json").read_text())
    for schema_key in ("raw_features", "model_features", "target"):
        if not feature_schema.get(schema_key):
            raise ValueError(f"Model package schema is missing {schema_key}.")

    split_manifest = json.loads((package_path / "split_manifest.json").read_text())
    if split_manifest.get("manifest_version") != SPLIT_MANIFEST_VERSION:
        raise ValueError("Model package split manifest version is unsupported.")
    split_row_ids = []
    for split_name in ("train", "validation", "test"):
        row_ids = split_manifest.get(f"{split_name}_row_ids")
        if not isinstance(row_ids, list):
            raise ValueError("Model package split manifest is invalid.")
        split_row_ids.extend(row_ids)
    if len(split_row_ids) != len(set(split_row_ids)):
        raise ValueError("Model package split manifest contains duplicate row IDs.")

    try:
        calibrated_model = joblib.load(package_path / "calibrated_model.joblib")
        selected_estimator = joblib.load(package_path / "selected_estimator.joblib")
        feature_engineering = joblib.load(package_path / "feature_engineering.joblib")
    except Exception as exc:
        raise ValueError(
            "Model package contains an unreadable serialized artifact."
        ) from exc

    if not hasattr(calibrated_model, "predict_proba"):
        raise ValueError("Calibrated model does not provide predict_proba.")
    if not hasattr(selected_estimator, "predict_proba"):
        raise ValueError("Selected estimator does not provide predict_proba.")
    if not hasattr(feature_engineering, "transform"):
        raise ValueError("Feature-engineering artifact does not provide transform.")
    return metadata


def load_model_package(package_directory: str | Path) -> dict[str, Any]:
    """Validate and load all runtime objects from a model package."""
    package_path = Path(package_directory)
    metadata = validate_model_package(package_path)
    return {
        "metadata": metadata,
        "calibrated_model": joblib.load(package_path / "calibrated_model.joblib"),
        "selected_estimator": joblib.load(package_path / "selected_estimator.joblib"),
        "feature_engineering": joblib.load(package_path / "feature_engineering.joblib"),
        "thresholds": json.loads((package_path / "thresholds.json").read_text()),
        "feature_schema": json.loads(
            (package_path / "feature_schema.json").read_text()
        ),
    }
