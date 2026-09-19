import json
import shutil
import unittest
from pathlib import Path

from src.model_package import create_model_package


class ModelPackageTests(unittest.TestCase):
    def test_model_package_contains_runtime_artifacts_and_metadata(self):
        test_directory = Path(__file__).parent / "_model_package_test"
        source_directory = test_directory / "source"
        package_directory = test_directory / "package"
        shutil.rmtree(test_directory, ignore_errors=True)
        source_directory.mkdir(parents=True)
        for artifact_name in (
            "calibrated_model.joblib",
            "selected_estimator.joblib",
            "feature_engineering.joblib",
            "thresholds.json",
            "feature_schema.json",
            "split_manifest.json",
        ):
            (source_directory / artifact_name).write_text(artifact_name)

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
        finally:
            shutil.rmtree(test_directory, ignore_errors=True)
