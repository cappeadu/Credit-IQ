"""Pydantic request and response contracts for the prediction API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

DECISIONS = Literal["APPROVE", "REVIEW", "REJECT"]


class CustomerRecord(BaseModel):
    """One cleaned customer record before feature engineering."""

    model_config = ConfigDict(extra="forbid")

    limit: FiniteFloat = Field(gt=0)
    gender: FiniteFloat
    edu: FiniteFloat
    marital_status: FiniteFloat
    age: FiniteFloat
    rep_status_mth_6: FiniteFloat
    rep_status_mth_5: FiniteFloat
    rep_status_mth_4: FiniteFloat
    rep_status_mth_3: FiniteFloat
    rep_status_mth_2: FiniteFloat
    rep_status_mth_1: FiniteFloat
    bill_amt_mth_6: FiniteFloat
    bill_amt_mth_5: FiniteFloat
    bill_amt_mth_4: FiniteFloat
    bill_amt_mth_3: FiniteFloat
    bill_amt_mth_2: FiniteFloat
    bill_amt_mth_1: FiniteFloat
    pmt_amt_mth_6: FiniteFloat
    pmt_amt_mth_5: FiniteFloat
    pmt_amt_mth_4: FiniteFloat
    pmt_amt_mth_3: FiniteFloat
    pmt_amt_mth_2: FiniteFloat
    pmt_amt_mth_1: FiniteFloat


class BatchPredictionRequest(BaseModel):
    """A bounded batch of cleaned customer records."""

    model_config = ConfigDict(extra="forbid")

    customers: list[CustomerRecord] = Field(min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    """Prediction for one customer."""

    probability: FiniteFloat = Field(ge=0, le=1)
    decision: DECISIONS


class BatchPredictionResponse(BaseModel):
    """Predictions for a submitted batch."""

    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Service readiness response."""

    status: Literal["ok"]
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Metadata exposed for the loaded model package."""

    package_version: int
    mlflow_run_id: str
    selected_model_name: str
    feature_version: str
    lower_threshold: FiniteFloat = Field(ge=0, le=1)
    upper_threshold: FiniteFloat = Field(ge=0, le=1)
