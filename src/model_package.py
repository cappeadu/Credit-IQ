"""Build an immutable, self-contained model package for local deployment."""

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PACKAGE_VERSION = 1


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
