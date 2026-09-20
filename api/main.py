"""FastAPI application skeleton with packaged-model startup loading."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerRecord,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.config import ROOT
from src.model_package import load_model_package
from src.predict import score_dataframe, transform_for_prediction

MODEL_PACKAGE_ENVIRONMENT_VARIABLE = "CREDIT_CARD_MODEL_PACKAGE"


def _score_customer_records(
    records: list[CustomerRecord],
    model_package: dict[str, Any],
) -> list[PredictionResponse]:
    """Transform and score API records with one loaded model package."""
    raw_data = pd.DataFrame([record.model_dump() for record in records])
    transformed_data = transform_for_prediction(
        raw_data,
        model_package["feature_engineering"],
        require_target=False,
    )
    thresholds = model_package["thresholds"]
    scored_data = score_dataframe(
        model_package["calibrated_model"],
        transformed_data,
        threshold=thresholds["upper_threshold"],
        lower_threshold=thresholds["lower_threshold"],
        upper_threshold=thresholds["upper_threshold"],
    )
    return [
        PredictionResponse(
            probability=float(row.probability),
            decision=row.decision,
        )
        for row in scored_data.itertuples()
    ]


def _require_model_package(request: Request) -> dict[str, Any]:
    """Return the loaded package or a clear service-unavailable response."""
    model_package = getattr(request.app.state, "model_package", None)
    if model_package is None:
        raise HTTPException(
            status_code=503,
            detail="The model package is not loaded.",
        )
    return model_package


def resolve_model_package_path(
    model_package_path: str | Path | None = None,
) -> Path:
    """Resolve an explicit package path or the configured environment value."""
    configured_path = model_package_path or os.getenv(
        MODEL_PACKAGE_ENVIRONMENT_VARIABLE
    )
    if not configured_path:
        raise RuntimeError(
            "A packaged model path is required. Set "
            f"{MODEL_PACKAGE_ENVIRONMENT_VARIABLE} before starting the API."
        )

    package_path = Path(configured_path)
    if not package_path.is_absolute():
        package_path = ROOT / package_path
    if not package_path.is_dir():
        raise FileNotFoundError(
            f"Configured model package directory was not found: {package_path}"
        )
    return package_path


def create_app(model_package_path: str | Path | None = None) -> FastAPI:
    """Create the API application and load one validated package at startup."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        package_path = resolve_model_package_path(model_package_path)
        application.state.model_package_path = package_path
        application.state.model_package = load_model_package(package_path)
        yield
        application.state.model_package = None

    application = FastAPI(
        title="Credit Card Risk API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        """Report whether the application has a loaded model package."""
        return HealthResponse(
            status="ok",
            model_loaded=request.app.state.model_package is not None,
        )

    @application.get("/model-info", response_model=ModelInfoResponse)
    async def model_info(request: Request) -> ModelInfoResponse:
        """Expose metadata and frozen thresholds for the loaded package."""
        package = _require_model_package(request)

        metadata = package["metadata"]
        thresholds = package["thresholds"]
        return ModelInfoResponse(
            package_version=metadata["package_version"],
            mlflow_run_id=metadata["mlflow_run_id"],
            selected_model_name=metadata["selected_model_name"],
            feature_version=metadata["feature_version"],
            lower_threshold=thresholds["lower_threshold"],
            upper_threshold=thresholds["upper_threshold"],
        )

    @application.post("/predict", response_model=PredictionResponse)
    async def predict_customer(
        customer: CustomerRecord,
        request: Request,
    ) -> PredictionResponse:
        """Return a prediction for one cleaned customer record."""
        model_package = _require_model_package(request)
        try:
            predictions = _score_customer_records([customer], model_package)
        except (TypeError, ValueError, KeyError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Customer could not be scored: {exc}",
            ) from exc
        return predictions[0]

    @application.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(
        prediction_request: BatchPredictionRequest,
        request: Request,
    ) -> BatchPredictionResponse:
        """Return predictions for a bounded batch of customer records."""
        model_package = _require_model_package(request)
        try:
            predictions = _score_customer_records(
                prediction_request.customers,
                model_package,
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Customers could not be scored: {exc}",
            ) from exc
        return BatchPredictionResponse(predictions=predictions)

    return application


app = create_app()
