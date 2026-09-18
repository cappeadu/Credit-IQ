"""Print a concise verification report for one completed MLflow comparison run."""

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mlflow.tracking import MlflowClient

from src.config import MLFLOW_TRACKING_URI
from src.tracking import verify_baseline_comparison_run

DEFAULT_CANDIDATE_MODELS = (
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and summarize a Credit Card MLflow comparison run."
    )
    parser.add_argument(
        "run_id",
        help="Parent MLflow run ID printed by src.train after training.",
    )
    parser.add_argument(
        "--candidate",
        dest="candidate_models",
        action="append",
        help="Expected candidate model name; repeat for additional models.",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    expected_models = arguments.candidate_models or list(DEFAULT_CANDIDATE_MODELS)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    parent_run = client.get_run(arguments.run_id)

    verify_baseline_comparison_run(arguments.run_id, expected_models)
    child_runs = client.search_runs(
        [parent_run.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{arguments.run_id}'",
    )

    report = {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "run_id": arguments.run_id,
        "status": parent_run.info.status,
        "selected_model": parent_run.data.tags.get("selected_model"),
        "feature_version": parent_run.data.tags.get("feature_version"),
        "dataset_fingerprint": parent_run.data.tags.get("dataset_fingerprint"),
        "parent_artifacts": [
            artifact.path for artifact in client.list_artifacts(arguments.run_id)
        ],
        "candidate_runs": [
            {
                "run_id": child.info.run_id,
                "model": child.data.tags.get("candidate_model"),
                "status": child.info.status,
                "metric_count": len(child.data.metrics),
                "artifact_paths": [
                    artifact.path
                    for artifact in client.list_artifacts(child.info.run_id)
                ],
            }
            for child in child_runs
            if child.data.tags.get("run_type") == "candidate-model"
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
