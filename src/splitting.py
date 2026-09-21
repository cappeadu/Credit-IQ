"""Dataset split persistence for reproducible training and retraining."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import validate_numeric_values, validate_schema

SPLIT_MANIFEST_VERSION = 1
SPLIT_NAMES = ("train", "validation", "test")


def build_stable_row_ids(dataset: pd.DataFrame) -> list[str]:
    """Build deterministic row IDs from row content and duplicate occurrence."""
    identity_columns = dataset.drop(columns=["target"], errors="ignore")
    row_hashes = pd.util.hash_pandas_object(identity_columns, index=False)
    occurrence_counts = row_hashes.groupby(row_hashes).cumcount()
    return [
        f"{int(row_hash):016x}-{int(occurrence)}"
        for row_hash, occurrence in zip(row_hashes, occurrence_counts)
    ]


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("manifest_version") != SPLIT_MANIFEST_VERSION:
        raise ValueError("Unsupported split manifest version.")
    for split_name in SPLIT_NAMES:
        row_ids = manifest.get(f"{split_name}_row_ids")
        if not isinstance(row_ids, list):
            raise TypeError(f"Split manifest is missing {split_name}_row_ids.")

    all_row_ids = [
        row_id
        for split_name in SPLIT_NAMES
        for row_id in manifest[f"{split_name}_row_ids"]
    ]
    if len(all_row_ids) != len(set(all_row_ids)):
        raise ValueError("Split manifest contains duplicate row IDs.")


def _save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _new_split_manifest(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    row_ids: pd.Series,
    *,
    dataset_fingerprint: str,
    random_state: int,
) -> dict[str, Any]:
    return {
        "manifest_version": SPLIT_MANIFEST_VERSION,
        "split_strategy": "stratified_random_80_10_10",
        "random_state": random_state,
        "dataset_fingerprint": dataset_fingerprint,
        "row_count": len(row_ids),
        "train_row_ids": row_ids.loc[training_data.index].tolist(),
        "validation_row_ids": row_ids.loc[validation_data.index].tolist(),
        "test_row_ids": row_ids.loc[test_data.index].tolist(),
    }


def create_or_load_splits(
    dataset: pd.DataFrame,
    *,
    manifest_path: str | Path,
    dataset_fingerprint: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create stable splits or reuse a manifest, assigning new rows to train."""
    validate_schema(dataset, require_target=True)
    validate_numeric_values(dataset, include_target=True)
    if dataset.empty:
        raise ValueError("Cannot split an empty dataset.")

    manifest_file = Path(manifest_path)
    row_ids = pd.Series(build_stable_row_ids(dataset), index=dataset.index)
    if row_ids.duplicated().any():
        raise ValueError("Stable row IDs must be unique within a dataset.")

    if not manifest_file.exists():
        training_data, holdout_data = train_test_split(
            dataset,
            test_size=0.2,
            stratify=dataset["target"],
            random_state=random_state,
        )
        validation_data, test_data = train_test_split(
            holdout_data,
            test_size=0.5,
            stratify=holdout_data["target"],
            random_state=random_state,
        )
        manifest = _new_split_manifest(
            training_data,
            validation_data,
            test_data,
            row_ids,
            dataset_fingerprint=dataset_fingerprint,
            random_state=random_state,
        )
        _save_manifest(manifest_file, manifest)
        return training_data, validation_data, test_data, manifest

    manifest = json.loads(manifest_file.read_text())
    _validate_manifest(manifest)
    current_row_ids = set(row_ids)
    manifest_row_ids = {
        row_id
        for split_name in SPLIT_NAMES
        for row_id in manifest[f"{split_name}_row_ids"]
    }
    missing_row_ids = manifest_row_ids - current_row_ids
    if missing_row_ids:
        raise ValueError(
            "Existing split records are missing from the new dataset; refusing "
            "to change frozen split membership."
        )

    split_by_row_id = {
        row_id: split_name
        for split_name in SPLIT_NAMES
        for row_id in manifest[f"{split_name}_row_ids"]
    }
    split_labels = row_ids.map(lambda row_id: split_by_row_id.get(row_id, "train"))
    training_data = dataset.loc[split_labels == "train"]
    validation_data = dataset.loc[split_labels == "validation"]
    test_data = dataset.loc[split_labels == "test"]

    manifest["dataset_fingerprint"] = dataset_fingerprint
    manifest["row_count"] = len(dataset)
    manifest["train_row_ids"] = row_ids.loc[training_data.index].tolist()
    manifest["validation_row_ids"] = row_ids.loc[validation_data.index].tolist()
    manifest["test_row_ids"] = row_ids.loc[test_data.index].tolist()
    _save_manifest(manifest_file, manifest)
    return training_data, validation_data, test_data, manifest
