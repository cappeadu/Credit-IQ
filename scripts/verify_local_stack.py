"""Verify the locally running packaged-model API and one Streamlit input path."""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.api_client import CreditRiskApiClient, CreditRiskApiError
from src.model_package import load_model_package

MODEL_PACKAGE_ENVIRONMENT_VARIABLE = "CREDIT_CARD_MODEL_PACKAGE"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the local Credit Card Risk API and model package."
    )
    parser.add_argument(
        "--dataset-path",
        default="data/test_only/test_only.csv",
        help="Raw labelled CSV used for one prediction request.",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("CREDIT_CARD_API_URL", "http://127.0.0.1:8000"),
        help="Base URL of the running FastAPI service.",
    )
    return parser


def resolve_package_path() -> Path:
    configured_path = os.getenv(MODEL_PACKAGE_ENVIRONMENT_VARIABLE)
    if not configured_path:
        raise RuntimeError(
            f"Set {MODEL_PACKAGE_ENVIRONMENT_VARIABLE} before running this check."
        )
    package_path = Path(configured_path)
    if not package_path.is_absolute():
        package_path = REPOSITORY_ROOT / package_path
    return package_path


def main() -> None:
    arguments = build_parser().parse_args()
    package_path = resolve_package_path()
    package_metadata = load_model_package(package_path)["metadata"]

    api_client = CreditRiskApiClient(arguments.api_url)
    health = api_client.health()
    model_info = api_client.model_info()

    dataset = pd.read_csv(REPOSITORY_ROOT / arguments.dataset_path)
    prediction = api_client.predict_dataframe(dataset.head(1))[0]
    explanation = prediction.get("explanation") or {}

    if not health.get("model_loaded"):
        raise RuntimeError("The API is reachable but no model package is loaded.")
    if model_info["mlflow_run_id"] != package_metadata["mlflow_run_id"]:
        raise RuntimeError("The API package does not match the configured package.")
    if not explanation.get("contributions"):
        raise RuntimeError("The API response did not include SHAP contributions.")

    print(
        json.dumps(
            {
                "status": "ok",
                "api_url": arguments.api_url,
                "package_path": str(package_path),
                "mlflow_run_id": model_info["mlflow_run_id"],
                "selected_model": model_info["selected_model_name"],
                "prediction": prediction,
                "shap_contribution_count": len(explanation["contributions"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (CreditRiskApiError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Local stack verification failed: {exc}") from exc
