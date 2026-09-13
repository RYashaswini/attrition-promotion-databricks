"""
All Databricks connection details come from environment variables.
Never hardcode the token or paste it into code - set these in a local
.env file (gitignored) for now, and as real secrets in CI/CD later.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # reads serving/.env into the process environment, if present


class Settings:
    DATABRICKS_HOST = os.environ["DATABRICKS_HOST"].rstrip("/")
    DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]

    ATTRITION_ENDPOINT_URL = f"{DATABRICKS_HOST}/serving-endpoints/attrition-endpoint/invocations"
    PROMOTION_ENDPOINT_URL = f"{DATABRICKS_HOST}/serving-endpoints/promotion-endpoint/invocations"

    ATTRITION_JOB_ID = "99409234083982"
    PROMOTION_JOB_ID = "638959830126724"

    CATALOG = "ml_dev"
    SCHEMA = "attrition_promotion"
    UPLOADS_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/uploads"
    OUTPUTS_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/outputs"

    # simple shared-secret auth for our own API, separate from the Databricks token
    API_KEY = os.environ.get("SERVING_API_KEY", "dev-only-key-change-me")


settings = Settings()