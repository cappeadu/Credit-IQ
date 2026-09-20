"""FastAPI application skeleton with packaged-model startup loading."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from src.config import ROOT
from src.model_package import load_model_package

MODEL_PACKAGE_ENVIRONMENT_VARIABLE = "CREDIT_CARD_MODEL_PACKAGE"


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

    return FastAPI(
        title="Credit Card Risk API",
        version="1.0.0",
        lifespan=lifespan,
    )


app = create_app()
