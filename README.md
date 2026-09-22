# CreditIQ — Credit Card Risk Demonstration

CreditIQ is a personal machine-learning project that predicts credit-card default risk and presents results through FastAPI and Streamlit.

It demonstrates an explainable workflow:

- data validation and reproducible splitting before feature engineering;
- comparison of multiple candidate models;
- calibration and validation-based threshold selection;
- MLflow experiment tracking;
- packaging of the selected model and runtime metadata;
- FastAPI model serving;
- Streamlit presentation of predictions, SHAP values, and deterministic explanations.

This is a demo and learning project, not a regulated credit-decisioning system or financial-advice tool.

## Architecture

```text
Training data → src/train.py → model package + metrics + MLflow run
                                      │
                                      ▼
                             FastAPI (api/main.py)
                                      │
                                      ▼
                              Streamlit (app.py)
```

Streamlit does not load the model directly. It sends raw customer records to FastAPI through `CREDIT_CARD_API_URL`.

## Repository structure

```text
api/                    FastAPI application and schemas
src/                    Production training, prediction, evaluation, and packaging code
tests/                  Tests
scripts/                Manual verification scripts
data/                   Source and split data
artifacts/              Generated model packages and local outputs
metrics/                Validation and test reports
app.py                  Streamlit application
Dockerfile              FastAPI container definition
requirements-api.txt    FastAPI container dependencies
environment.yml         Full local training/development environment
```

## Local setup

From the repository root in PowerShell:

```powershell
conda env create -f environment.yml
conda activate credit-card-risk
```

For an existing environment:

```powershell
conda activate credit-card-risk
conda env update -f environment.yml --prune
```

Run the tests:

```powershell
python -B -m unittest discover -s tests -q
```

## Data

The training pipeline validates the source schema and creates reproducible train, validation, and test splits before applying feature engineering.

Users can download the validation data used by the project here:

[Download `validation_set.csv`](data/split/validation_set.csv)

The Streamlit upload workflow expects compatible raw customer columns. The target column is not required for prediction uploads.

## Training and MLflow

Start MLflow in one PowerShell terminal:

```powershell
conda activate credit-card-risk
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`, then train in another terminal:

```powershell
conda activate credit-card-risk
python -m src.train train
```

Training validates data, creates or reuses the split manifest, fits feature engineering using training data, compares Logistic Regression, Random Forest, and XGBoost, calibrates the selected model, selects validation thresholds, tracks the run in MLflow, and creates a model package.

The command prints a path similar to:

```text
artifacts/model_package_<MLFLOW_RUN_ID>
```

The package contains the calibrated model, selected estimator, feature-engineering object, thresholds, feature schema, split manifest, and package metadata.

Validation results are written to `metrics/val_set.json`.

Evaluate the untouched test set only after training has completed:

```powershell
python -m src.predict evaluate `
  --data-path data/test_only/test_only.csv `
  --model-path artifacts/model_package_<MLFLOW_RUN_ID> `
  --dataset-name test `
  --dataset-role test `
  --source-training-run-id <MLFLOW_RUN_ID>
```

The test report includes ranking, classification, calibration, decision-rate, false-approval, and false-rejection metrics.

## FastAPI

Set the model package and start the service:

```powershell
$env:CREDIT_CARD_MODEL_PACKAGE = "artifacts\model_package_<MLFLOW_RUN_ID>"
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Swagger is available at `http://127.0.0.1:8000/docs`.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Check service and model readiness |
| `GET` | `/model-info` | Return model and threshold metadata |
| `POST` | `/predict` | Score one customer |
| `POST` | `/predict/batch` | Score multiple customers |
| `POST` | `/explain` | Return a deterministic explanation from SHAP context |

The API applies packaged feature engineering and frozen thresholds. SHAP values describe the underlying selected estimator; probability and decision come from the calibrated runtime model.

## Streamlit

Start FastAPI first, then run Streamlit in another terminal:

```powershell
conda activate credit-card-risk
$env:CREDIT_CARD_API_URL = "http://127.0.0.1:8000"
python -m streamlit run app.py
```

The app supports:

- built-in test-dataset scoring;
- compatible CSV upload and batch scoring;
- customer selection by original row;
- probability and decision breakdowns;
- SHAP charts and contribution tables;
- deterministic explanations through `/explain`;
- portfolio summaries and validation model comparison.

The API client uses a 60-second request timeout to tolerate cold starts on free hosting services.

## Docker API

The Dockerfile builds the FastAPI service only. Streamlit can run locally or through Streamlit Community Cloud.

Build and run the API image:

```powershell
docker build -t credit-card-risk-api:local .
docker run --rm --name credit-card-risk-api -p 8000:8000 credit-card-risk-api:local
```

Test it:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/model-info
```

Then point Streamlit to the container:

```powershell
$env:CREDIT_CARD_API_URL = "http://127.0.0.1:8000"
python -m streamlit run app.py
```

If a new package is selected, update the package path in the deployment Dockerfile before rebuilding.

## Deployment

The intended demo deployment uses separate services:

- FastAPI is deployed as a Docker service, such as Render.
- Streamlit is deployed separately from GitHub.
- Streamlit receives the API address through `CREDIT_CARD_API_URL`, for example:

```text
CREDIT_CARD_API_URL=https://your-api-service.onrender.com
```

The deployment branch may use a pip-based `requirements.txt` for Streamlit, while the main development branch uses `environment.yml` for local training.

## Troubleshooting

### Matplotlib is missing in Streamlit

Ensure the deployed Streamlit branch contains a recognized dependency file with Streamlit, Matplotlib, NumPy, and Pandas, then force a clean redeploy.

### Streamlit cannot connect to FastAPI

Check the API directly:

```powershell
Invoke-RestMethod https://your-api-service.onrender.com/health
```

`CREDIT_CARD_API_URL` must contain only the API base URL, not `/docs` or `/predict`. Free Render services may sleep; the first request can be slow.

### FastAPI starts without a model

Confirm that `CREDIT_CARD_MODEL_PACKAGE` points to a complete model-package directory.

### MLflow training fails

Start the MLflow server before running training and confirm it is available at `http://127.0.0.1:5000`.

## Scope and limitations

This project intentionally does not include CI/CD, automated retraining, cloud model registries, authentication, distributed processing, enterprise monitoring, or regulated credit-decisioning controls.

Model output is for education and demonstration only and should not be used as the sole basis for real financial decisions.
