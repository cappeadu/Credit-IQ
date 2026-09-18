# Credit Card Risk Project Guidelines

## Project purpose

This repository contains a personal credit-risk modelling project. It trains models to estimate the probability of credit-card default and assigns one of three operational decisions:

- `APPROVE`
- `REVIEW`
- `REJECT`

The project should be solid, reproducible, understandable, and explainable. It is not intended to implement enterprise-scale MLOps, CI/CD, cloud infrastructure, or a regulated production credit-decisioning platform at this stage.

## Repository structure

- `src/`: production code used by training, preprocessing, prediction, model packaging, and API functionality.
- `scripts/`: one-off or operational Python files and commands, such as local training runners, evaluation utilities, artifact-generation commands, and migration helpers. Reusable production logic belongs in `src/`, not in `scripts/`.
- `api/`: FastAPI application code and API schemas, if introduced during the implementation phases.
- `artifacts/`: generated model packages and related runtime artifacts. Do not edit generated artifacts manually.
- `data/`: local datasets and generated data splits. Sensitive or large data should not be committed.
- `metrics/`: generated evaluation outputs and reports.
- `notebooks/`: exploratory analysis only. Notebook code must not be the only place where production behaviour is defined.
- `tests/`: automated tests, when introduced.
- `app.py`: Streamlit user interface. It should call the FastAPI service for prediction once the API phase is implemented.

The current repository may not contain every directory listed above. Directories should be added only when needed by an implementation phase.

## Implementation phases

Work must be implemented phase by phase. Do not begin a later phase unless the user explicitly requests it.

Planned phases:

1. Scope and documentation.
2. Standardise data preparation and feature engineering.
3. Compare and tune multiple models.
4. Track experiments and reproducibility with MLflow.
5. Correct calibration, threshold selection, and evaluation.
6. Package the selected model and metadata.
7. Add the FastAPI prediction service.
8. Update Streamlit to consume FastAPI.
9. Add structured SHAP context for AI explanations.
10. Add practical tests and manual workflow documentation.

When implementing a phase:

- keep changes within that phase's scope;
- preserve existing user changes;
- explain changed files and behaviour after implementation;
- run relevant checks before reporting completion;
- do not silently implement later-phase features.

## Data and model conventions

- Use one canonical preprocessing and feature-engineering path for training, batch prediction, FastAPI, and Streamlit.
- Validate input columns, data types, missing values, impossible values, and feature order.
- Keep target columns out of live prediction input.
- Treat thresholds as configuration, not duplicated constants.
- Select thresholds using validation data and evaluate the frozen policy on an untouched test set.
- Do not assume Random Forest is the best model. Compare documented baseline models and optionally tune them with Optuna.
- Track training configuration, parameters, metrics, data identity, and selected model information with MLflow.
- Include model version and MLflow run information in the final model package when model packaging is implemented.

## Evaluation expectations

Evaluation should cover both ranking quality and operational decisions.

At minimum, record:

- ROC-AUC;
- PR-AUC;
- precision;
- recall;
- F1;
- specificity;
- confusion matrix;
- Brier score and calibration information;
- approval, review, and rejection rates;
- false-approval and false-rejection rates.

Baseline model selection should use a documented lexicographic policy:

1. PR-AUC;
2. recall;
3. F1;
4. specificity;
5. ROC-AUC.

PR-AUC is primary because default prediction is an imbalanced classification
problem. This policy is a baseline and may be revised after threshold and
calibration analysis, but model selection must not silently rely on one metric.

The operational policy is:

```text
probability < lower_threshold        APPROVE
lower_threshold <= probability <
upper_threshold                      REVIEW
probability >= upper_threshold       REJECT
```

Threshold boundaries must be tested explicitly. Metrics must describe the same thresholds used by the application.

## Explainability and AI context

SHAP values should be returned as structured explanation context containing feature names, feature values, SHAP values, contribution direction, and model metadata.

Do not hard-code financial advice or rule-based recommendations from SHAP values. A later OpenAI explanation layer may use the structured SHAP context, but it must not invent facts, claim causation, guarantee an outcome, or override the model decision.

## Local development expectations

The intended local workflow is:

1. prepare and validate data;
2. train and evaluate candidate models;
3. track runs in local MLflow;
4. package the selected model;
5. run FastAPI separately with Uvicorn;
6. run Streamlit separately;
7. have Streamlit call FastAPI over HTTP.

Use project-relative paths based on the repository location. Do not rely on the current working directory when loading artifacts.

## Scope boundaries

Currently out of scope:

- CI/CD;
- cloud deployment;
- enterprise model registries;
- authentication and authorisation platforms;
- distributed processing;
- automated retraining;
- production database infrastructure;
- regulated credit-decisioning compliance processes.

These may be considered later, but should not be added implicitly to a requested phase.
