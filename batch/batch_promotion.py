"""
Batch scoring job for the promotion-readiness model.

Run as a Databricks Job/notebook. Accepts an input_path widget/parameter
so it can score whatever CSV was uploaded, not just a fixed file.

Reads: any CSV matching the promotion feature schema (employee_id + features,
       target column optional/ignored if present).
Writes: predictions CSV to the outputs Volume, named after the run.
"""

import sys
from datetime import datetime

import mlflow
import pandas as pd

CATALOG = "ml_dev"
SCHEMA = "attrition_promotion"
MODEL_URI = f"models:/{CATALOG}.{SCHEMA}.promotion_model@champion"

try:
    input_path = dbutils.widgets.get("input_path")  # noqa: F821
except Exception:
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        raise ValueError(
            "input_path not provided - pass it as a job parameter "
            "(file task) or set the notebook widget."
        )

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

output_path = f"/Volumes/{CATALOG}/{SCHEMA}/outputs/promotion_predictions_{run_id}.csv"

mlflow.set_registry_uri("databricks-uc")

print(f"Loading champion model from {MODEL_URI}")
model = mlflow.pyfunc.load_model(MODEL_URI)

print(f"Reading input from {input_path}")
df = pd.read_csv(input_path)

employee_ids = df["employee_id"] if "employee_id" in df.columns else None
feature_df = df.drop(columns=[c for c in ["employee_id", "promoted"] if c in df.columns])

predictions = model.predict(feature_df)

result = pd.DataFrame({
    "employee_id": employee_ids if employee_ids is not None else range(len(df)),
    "promotion_prediction": predictions,
})

result.to_csv(output_path, index=False)
print(f"Wrote {len(result)} predictions to {output_path}")