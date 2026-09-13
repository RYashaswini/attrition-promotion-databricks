"""
Batch scoring job for the attrition model.

Run as a Databricks Job/notebook. Accepts an input_path widget/parameter
so it can score whatever CSV was uploaded (e.g. via the future FastAPI
upload endpoint), not just a fixed file.

Reads: any CSV matching the attrition feature schema (employee_id + features,
       target column optional/ignored if present).
Writes: predictions CSV to the outputs Volume, named after the run.
"""

import sys

import mlflow
import pandas as pd

CATALOG = "ml_dev"
SCHEMA = "attrition_promotion"
MODEL_URI = f"models:/{CATALOG}.{SCHEMA}.attrition_model@champion"

# --- get input_path: Databricks widget if run as a notebook, else CLI arg ---
try:
    dbutils  # noqa: F821  -- only exists inside a Databricks notebook context
    input_path = dbutils.widgets.get("input_path")  # noqa: F821
    run_id = dbutils.widgets.get("run_id")  # noqa: F821
except NameError:
    input_path = sys.argv[1]
    run_id = sys.argv[2] if len(sys.argv) > 2 else "manual"

output_path = f"/Volumes/{CATALOG}/{SCHEMA}/outputs/attrition_predictions_{run_id}.csv"

mlflow.set_registry_uri("databricks-uc")

print(f"Loading champion model from {MODEL_URI}")
model = mlflow.pyfunc.load_model(MODEL_URI)

print(f"Reading input from {input_path}")
df = pd.read_csv(input_path)

# keep employee_id for the output, drop it (and the label if present) before scoring
employee_ids = df["employee_id"] if "employee_id" in df.columns else None
feature_df = df.drop(columns=[c for c in ["employee_id", "attrition"] if c in df.columns])

predictions = model.predict(feature_df)

result = pd.DataFrame({
    "employee_id": employee_ids if employee_ids is not None else range(len(df)),
    "attrition_prediction": predictions,
})

result.to_csv(output_path, index=False)
print(f"Wrote {len(result)} predictions to {output_path}")