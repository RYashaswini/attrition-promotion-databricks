"""
Trains the promotion-readiness model on Databricks, logs to MLflow,
registers to Unity Catalog, and promotes to @champion only if it beats
the current champion's test ROC-AUC (or if there is no champion yet).

Run this as a Databricks notebook/job with a cluster attached.
Reads from the Volume created earlier; writes nothing back to git.
"""

import sys

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --- config -----------------------------------------------------------
CATALOG = sys.argv[1] if len(sys.argv) > 1 else "ml_dev"  # passed by databricks.yml per target
SCHEMA = "attrition_promotion"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.promotion_model"
DATA_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/uploads/promotion_raw.csv"
CAT_COLS = ["department", "education_level"]
TARGET = "promoted"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Shared/promotion_training")

# --- load data ----------------------------------------------------------
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["employee_id", TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)],
    remainder="passthrough",
)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)),
])

# --- train + log --------------------------------------------------------
with mlflow.start_run(run_name="promotion_training") as run:
    pipeline.fit(X_train, y_train)
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)

    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", 8)
    mlflow.log_metric("test_roc_auc", test_auc)

    logged_model = mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
        input_example=X_test.head(3),
    )
    new_version = logged_model.registered_model_version

print(f"Trained. Test ROC-AUC: {test_auc:.4f}. Registered as version {new_version}.")

# --- champion promotion gate ---------------------------------------------
client = MlflowClient()

current_champion_auc = None
try:
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    champion_run = client.get_run(champion.run_id)
    current_champion_auc = champion_run.data.metrics.get("test_roc_auc")
    print(f"Current champion: version {champion.version}, test_roc_auc={current_champion_auc}")
except Exception:
    print("No existing champion found - this will become the first champion.")

if current_champion_auc is None or test_auc > current_champion_auc:
    client.set_registered_model_alias(MODEL_NAME, "champion", new_version)
    print(f"PROMOTED version {new_version} to @champion (test_roc_auc={test_auc:.4f}).")
else:
    print(
        f"NOT promoted. New version {new_version} (AUC={test_auc:.4f}) "
        f"did not beat current champion (AUC={current_champion_auc:.4f})."
    )