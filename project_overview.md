# Credit Card Risk Project Overview

## Purpose

This project estimates the probability that a credit-card customer will default and presents an operational risk decision:

- `APPROVE`: lower predicted risk;
- `REVIEW`: uncertain or intermediate risk requiring manual consideration;
- `REJECT`: higher predicted risk.

It is a personal project intended to demonstrate a clear machine-learning workflow, reproducible experiments, explainable predictions, and a small local application. It is not currently intended to be an enterprise or regulated production credit-decisioning system.

## Current repository roles

The repository currently contains:

- `src/`: reusable and production-oriented Python code;
- `scripts/`: reserved for one-off commands and utilities that will be added in later phases;
- `app.py`: Streamlit interface;
- `artifacts/`: generated model and explanation artifacts;
- `data/`: local source data and generated splits;
- `metrics/`: generated validation and test metrics;
- `notebooks/`: exploratory analysis.

The existing implementation trains models from the credit-card dataset, creates engineered payment and utilisation features, calibrates a selected classifier, generates scored customers, and displays portfolio and customer-level information in Streamlit. The workflow will be progressively reorganised so the same preprocessing, model, threshold, and explanation behaviour is used everywhere.

Saved train, validation, and test datasets are kept in the cleaned raw schema.
The feature-engineering artifact is applied after loading a dataset and before
model scoring. This keeps evaluation and future API prediction on the same
transformation path and allows learned preprocessing to be fitted on training
data only if it is introduced later.

Training also persists a split manifest. Existing records retain their original
train, validation, or test membership on retraining; newly added records are
assigned to training. If records from a frozen split are removed, training stops
instead of silently changing the evaluation population. The manifest is stored
alongside the split data as `split_manifest.json`.

After a successful training run, an immutable package is created at
`artifacts/model_package_<MLFLOW_RUN_ID>/`. It contains the calibrated model,
selected estimator, feature-engineering artifact, thresholds, feature schema,
split manifest, and package metadata. The MLflow run ID in the package metadata
links the runtime files back to the tracked training and validation results.
Before scoring, a package is validated for required files, metadata, thresholds,
feature schema, split manifest, and readable serialized objects. Prediction can
only use a packaged model directory; the loose `artifacts/` directory is not a
valid runtime model input.

## Planned architecture

```text
Raw dataset
   ↓
Data validation and cleaning
   ↓
Canonical feature engineering
   ↓
Candidate model training and tuning
   ↓
MLflow experiment tracking
   ↓
Calibration and threshold selection
   ↓
Untouched test evaluation
   ↓
Versioned model package
   ↓
FastAPI prediction service
   ↓
Streamlit user interface
```

The FastAPI service will own model loading, input validation, prediction, threshold decisions, and SHAP context generation. Streamlit will act as the user interface and will call FastAPI over HTTP.

## Model experimentation

Random Forest must not be assumed to be the final model. Candidate models should be compared using the same data splits, feature preparation, calibration approach, and operational evaluation.

Initial candidates may include:

- Logistic Regression;
- Random Forest;
- XGBoost;
- another suitable scikit-learn gradient-boosting model if useful.

Optuna may be used to tune a limited, documented parameter space after baseline comparisons work. Tuning must use training and validation data only and must not use the final test set.

## Reproducibility with MLflow

MLflow will be used locally to track experiments and make model comparisons reproducible. Runs should record:

- model family;
- hyperparameters;
- random seed;
- dataset identity or fingerprint;
- feature-engineering version;
- split configuration;
- calibration method;
- threshold values;
- validation and test metrics;
- Optuna trial information where relevant;
- generated plots and model artifacts.

The selected model package should retain the MLflow run ID and enough metadata to identify how it was produced. A local MLflow store is sufficient for this project; cloud tracking and enterprise model registries are out of scope.

MLflow uses SQLite metadata in mlflow.db and stores artifacts in mlruns/.
Start the tracking server from the repository root with:

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000

Training connects to http://127.0.0.1:5000 and should only be started after the
server is running.

Each comparison run records a SHA-256 dataset fingerprint, feature-engineering
version, Git revision when available, split row counts, and a JSON feature schema.
After logging, training verifies that the parent run finished and that each
candidate model has a nested child run. This makes missing or incomplete tracking
fail clearly instead of silently producing an incomplete experiment record.

After training, the run can be reviewed without retraining:

```powershell
python scripts/verify_mlflow_run.py <PARENT_RUN_ID>
```

The script prints the selected model, reproducibility metadata, parent artifacts,
and each candidate run's metrics and artifacts. It exits with an error if the
expected candidate runs are missing. If the candidate registry changes, provide
the expected names explicitly with repeated `--candidate` options.

## Evaluation approach

Use three data roles:

- training data for fitting models and internal calibration;
- validation data for model comparison and threshold selection;
- an untouched test set for the final report.

Evaluation must include both general model quality and the actual three-way decision policy.

Important metrics include:

- ROC-AUC for ranking;
- PR-AUC for imbalanced default detection;
- precision, recall, F1, and specificity;
- confusion matrix;
- Brier score and calibration curve;
- approval, review, and rejection rates;
- false approvals and false rejections;
- default rate within each decision group.

For baseline candidate-model selection, use the following lexicographic policy:
PR-AUC first, then recall, F1, specificity, and ROC-AUC. PR-AUC is the primary
metric because the default target is imbalanced. This policy can be revised
after calibration and threshold analysis, but the selection rule must remain
explicit and reproducible.

The operational decision policy is:

```text
probability < lower_threshold        APPROVE
lower_threshold <= probability <
upper_threshold                      REVIEW
probability >= upper_threshold       REJECT
```

Thresholds should be selected on validation data using documented business assumptions or cost trade-offs, then frozen before final test evaluation. The current threshold values are candidates to evaluate, not permanent truths.

Prediction and evaluation are dataset-role independent. The scoring command can
score labelled or unlabelled feature-engineered data, while the evaluation
command requires labels and can evaluate validation, test, or another labelled
dataset. The caller supplies the dataset name and output path; the code does
not assume that every labelled dataset is the final test set.

Evaluation runs are also tracked in MLflow with the supplied dataset role and a
dataset fingerprint. The training parent run contains validation evaluation;
later evaluation commands create separate runs for roles such as `test` or
`external_evaluation` and can link them to the source training run.

```powershell
python -m src.predict score --data-path data/new_customers.csv --model-path artifacts/model_package_<MLFLOW_RUN_ID>
python -m src.predict evaluate --data-path data/test_only/test_only.csv --model-path artifacts/model_package_<MLFLOW_RUN_ID> --dataset-name test --dataset-role test
```

## PowerShell command reference

Run these commands from the repository root. Use separate PowerShell terminals
for the MLflow server and training/evaluation commands.

```powershell
# Move to the project root
Set-Location "C:\Users\Lenovo\code\Credit Card"

# Run the automated tests
python -B -m unittest discover -s tests -v
```

Start MLflow in the first terminal:

```powershell
mlflow server `
  --backend-store-uri sqlite:///mlflow.db `
  --default-artifact-root ./mlruns `
  --host 127.0.0.1 `
  --port 5000
```

Run training in a second terminal:

```powershell
python -m src.train `
  --data-path data/credit_card.xls `
  --path-to-save-val-test data/split `
  --path-to-save-test-only data/test_only
```

After training, save the printed MLflow parent run ID and inspect the run:

```powershell
$PARENT_RUN_ID = "paste-parent-run-id-here"
python -B scripts/verify_mlflow_run.py $PARENT_RUN_ID
```

Evaluate the frozen model and thresholds on a labelled dataset:

```powershell
python -m src.predict evaluate `
  --data-path data/test_only/test_only.csv `
  --model-path artifacts/model_package_$PARENT_RUN_ID `
  --dataset-name test `
  --dataset-role test `
  --source-training-run-id $PARENT_RUN_ID
```

Score another cleaned, unlabelled dataset:

```powershell
python -m src.predict score `
  --data-path data/new_customers.csv `
  --model-path artifacts `
  --dataset-name new_customers
```

Review generated reports and thresholds:

```powershell
Get-Content artifacts/thresholds.json
Get-Content metrics/test.json
Get-Content metrics/val_set.json
```

## FastAPI interface

The API loads one validated packaged model at startup. Set the package path
before starting Uvicorn:

```powershell
$env:CREDIT_CARD_MODEL_PACKAGE = "artifacts/model_package_<MLFLOW_RUN_ID>"
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Startup fails if the environment variable is missing or the package fails
validation. Loose files in `artifacts/` are not accepted by the API.

The likely API endpoints are:

- `GET /health`: confirms that the service and model package loaded;
- `GET /model-info`: returns model version, model family, MLflow run ID, feature version, and thresholds;
- `POST /predict`: scores one customer;
- `POST /predict/batch`: scores multiple customers;
- `POST /explain`: returns prediction details and structured SHAP context.

An optional later endpoint, `POST /explain/ai`, may send structured SHAP context to OpenAI. The core prediction endpoint should not depend on an AI explanation being available.

## SHAP and AI explanations

The explanation flow will be:

```text
Customer input
   ↓
FastAPI prediction
   ↓
Probability and decision
   ↓
SHAP values and feature context
   ↓
OpenAI explanation layer
```

The project will not hard-code recommendations based on SHAP values. The API should return feature names, feature values, SHAP values, contribution direction, and model metadata. The later AI layer should explain the supplied evidence without inventing facts, claiming causation, guaranteeing approval, or overriding the model decision.

## Scope and limitations

This project focuses on a clear local workflow. It does not currently include:

- CI/CD;
- cloud deployment;
- enterprise MLOps;
- automated retraining;
- production databases;
- enterprise authentication;
- regulated credit-decisioning controls;
- real-time monitoring infrastructure.

The model should therefore be presented as an educational and analytical tool until those concerns are addressed.

## Implementation rule

The work will be implemented in explicit phases. No later phase should be implemented unless it is explicitly requested. Phase 1 establishes this scope and documentation only; it does not change the training pipeline, API, model artifacts, or Streamlit behaviour.

API request schemas accept cleaned customer fields only. The target column and
unexpected fields are rejected, and batch requests are limited to 1,000
customers per request in the initial local implementation.

### FastAPI endpoint checks

Start the API with a packaged model selected:

```powershell
$env:CREDIT_CARD_MODEL_PACKAGE = "artifacts/model_package_<MLFLOW_RUN_ID>"
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Useful local checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/model-info
Start-Process http://127.0.0.1:8000/docs
```

The prediction endpoints are `POST /predict` for one customer and
`POST /predict/batch` for a bounded batch. FastAPI returns `422` for invalid
request data or scoring input and `503` when no model package is loaded.

### Phase 8 — Streamlit integration stages

1. **API client and prediction migration:** Streamlit uses FastAPI for model
   metadata and uploaded-customer scoring instead of loading model artifacts.
2. **Dashboard and batch workflow migration:** completed for the local test
   dataset and uploads. Streamlit loads raw `data/test_only/test_only.csv`,
   sends raw customer fields to FastAPI in batches of at most 1,000, and uses
   the returned probabilities and decisions for the dashboard.
3. **Explanation integration:** connect the packaged/API explanation data to
   the planned SHAP context and AI explanation layer without hard-coded claims.
4. **End-to-end review:** verify the API and Streamlit workflow together,
   document startup order, and test unavailable-service behaviour.

The end-to-end check can be run after MLflow, FastAPI, and the packaged model
are ready:

```powershell
python -B scripts/verify_local_stack.py
```

It validates the configured package locally, checks API health and metadata,
sends one raw test customer through the API, and confirms that SHAP
contributions are present. Start Streamlit separately to review the visual
dashboard and customer breakdown:

```powershell
$env:CREDIT_CARD_API_URL = "http://127.0.0.1:8000"
python -m streamlit run app.py
```

### Phase 9 — AI explanation stages

1. **Explanation contract:** define the structured prediction, model metadata,
   SHAP context, and grounded AI response shape. This stage is complete; it
   does not call OpenAI.
2. **OpenAI configuration and client:** add server-side configuration and a
   small client with timeout and unavailable-service handling.
3. **Explanation service and endpoint:** implement `POST /explain` using the
   prediction context and return the structured AI response.
4. **Streamlit integration and review:** add an optional explanation action,
   keep raw SHAP values visible, and test the full workflow without allowing
   the AI to change the model decision.
