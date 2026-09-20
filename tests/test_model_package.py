import json
import shutil
import unittest
from pathlib import Path

import joblib

from src.model_package import (
    create_model_package,
    load_model_package,
    validate_model_package,
)


class DummyModel:
    def predict_proba(self, features):
        return [[0.5, 0.5]] * len(features)


class DummyFeatureEngineering:
    def transform(self, data):
        return data


class ModelPackageTests(unittest.TestCase):
    def test_model_package_contains_runtime_artifacts_and_metadata(self):
        test_directory = Path(__file__).parent / "_model_package_test"
        source_directory = test_directory / "source"
        package_directory = test_directory / "package"
        shutil.rmtree(test_directory, ignore_errors=True)
        source_directory.mkdir(parents=True)
        joblib.dump(DummyModel(), source_directory / "calibrated_model.joblib")
        joblib.dump(DummyModel(), source_directory / "selected_estimator.joblib")
        joblib.dump(
            DummyFeatureEngineering(),
            source_directory / "feature_engineering.joblib",
        )
        (source_directory / "thresholds.json").write_text(
            json.dumps({"lower_threshold": 0.1, "upper_threshold": 0.3})
        )
        (source_directory / "feature_schema.json").write_text(
            json.dumps(
                {
                    "raw_features": ["feature"],
                    "model_features": ["feature"],
                    "target": "target",
                }
            )
        )
        (source_directory / "split_manifest.json").write_text(
            json.dumps(
                {
                    "manifest_version": 1,
                    "train_row_ids": ["train"],
                    "validation_row_ids": ["validation"],
                    "test_row_ids": ["test"],
                }
            )
        )

        try:
            created_package = create_model_package(
                package_directory=package_directory,
                calibrated_model_path=source_directory / "calibrated_model.joblib",
                selected_estimator_path=source_directory / "selected_estimator.joblib",
                feature_engineering_path=source_directory / "feature_engineering.joblib",
                thresholds_path=source_directory / "thresholds.json",
                feature_schema_path=source_directory / "feature_schema.json",
                split_manifest_path=source_directory / "split_manifest.json",
                metadata={"mlflow_run_id": "run-123"},
            )

            self.assertTrue((created_package / "package_metadata.json").exists())
            package_metadata = json.loads(
                (created_package / "package_metadata.json").read_text()
            )
            self.assertEqual(package_metadata["package_version"], 1)
            self.assertEqual(package_metadata["mlflow_run_id"], "run-123")
            self.assertTrue(
                (created_package / "feature_engineering.joblib").exists()
            )
            self.assertEqual(
                validate_model_package(created_package)["mlflow_run_id"],
                "run-123",
            )
            loaded_package = load_model_package(created_package)
            self.assertIn("calibrated_model", loaded_package)
        finally:
            shutil.rmtree(test_directory, ignore_errors=True)
