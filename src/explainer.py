from typing import Annotated

import joblib
import shap
import typer

from src.config import ROOT

app = typer.Typer()


@app.command()
def test_predictions(
    model_path: Annotated[str, typer.Option(help="dataset path or link")] = None,
):

    model = joblib.load(f"{model_path}/Random Forest.joblib")
    explainer = shap.TreeExplainer(model)
    explainer_file_name = ROOT / "artifacts/shap_explainer.joblib"
    joblib.dump(explainer, explainer_file_name)
    print("Explainer Saved.")
    return explainer


if __name__ == "__main__":
    app()
