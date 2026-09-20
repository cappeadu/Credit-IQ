"""Small client for the local Credit Card Risk FastAPI service."""

import json
import os
from collections.abc import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from src.data import RAW_FEATURE_COLUMNS

DEFAULT_API_URL = "http://127.0.0.1:8000"
API_URL_ENVIRONMENT_VARIABLE = "CREDIT_CARD_API_URL"
MAX_BATCH_SIZE = 1000


class CreditRiskApiError(RuntimeError):
    """Raised when the prediction API cannot fulfil a request."""


class CreditRiskApiClient:
    """Client for health, model metadata, and prediction API calls."""

    def __init__(self, base_url: str | None = None, timeout_seconds: float = 10.0):
        configured_url = base_url or os.getenv(
            API_URL_ENVIRONMENT_VARIABLE,
            DEFAULT_API_URL,
        )
        self.base_url = configured_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request(self, method: str, path: str, payload: Mapping | None = None):
        request_body = None
        headers = {}
        if payload is not None:
            request_body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            f"{self.base_url}{path}",
            data=request_body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                error_body = json.loads(exc.read().decode("utf-8"))
                detail = error_body.get("detail", str(exc))
            except (json.JSONDecodeError, UnicodeDecodeError):
                detail = str(exc)
            raise CreditRiskApiError(f"API request failed ({exc.code}): {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise CreditRiskApiError(
                f"Could not connect to the prediction API at {self.base_url}."
            ) from exc

    def health(self) -> dict:
        """Return the API readiness response."""
        return self._request("GET", "/health")

    def model_info(self) -> dict:
        """Return metadata for the loaded packaged model."""
        return self._request("GET", "/model-info")

    def predict_records(self, records: Sequence[Mapping]) -> list[dict]:
        """Score a non-empty sequence of raw customer records."""
        if not records:
            raise ValueError("At least one customer record is required.")
        response = self._request("POST", "/predict/batch", {"customers": list(records)})
        return response["predictions"]

    def predict_dataframe(self, raw_data: pd.DataFrame) -> list[dict]:
        """Score raw customer data through the API batch endpoint."""
        missing_columns = [
            column for column in RAW_FEATURE_COLUMNS if column not in raw_data.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required customer columns: {missing_columns}")
        records = raw_data[RAW_FEATURE_COLUMNS].to_dict(orient="records")
        predictions = []
        for start_index in range(0, len(records), MAX_BATCH_SIZE):
            predictions.extend(
                self.predict_records(
                    records[start_index : start_index + MAX_BATCH_SIZE]
                )
            )
        return predictions
