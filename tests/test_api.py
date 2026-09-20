import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import create_app, resolve_model_package_path


class ApiStartupTests(unittest.TestCase):
    def test_explicit_relative_package_path_is_resolved_from_repository_root(self):
        resolved_path = resolve_model_package_path("artifacts")

        self.assertEqual(
            resolved_path,
            Path(__file__).resolve().parent.parent / "artifacts",
        )

    def test_api_loads_the_validated_package_during_startup(self):
        fake_package = {"metadata": {"mlflow_run_id": "run-123"}}
        application = create_app("artifacts")

        with patch("api.main.load_model_package", return_value=fake_package):
            with TestClient(application):
                self.assertEqual(application.state.model_package, fake_package)

    def test_api_startup_fails_without_a_package_path(self):
        application = create_app()

        with self.assertRaises(RuntimeError):
            with TestClient(application):
                pass


if __name__ == "__main__":
    unittest.main()
