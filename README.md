# attrition-promotion-databricks

Two models — Employee Attrition Risk and Promotion Readiness — trained,
registered, and served entirely on Databricks, orchestrated through a single
FastAPI layer, with CI/CD via GitHub Actions.



## Repo layout

```
data_generation/   synthetic data generators for both models
models/
  attrition/       training script + MLflow/UC registration
  promotion/       training script + MLflow/UC registration
batch/             batch scoring jobs (read from a Volume, write to a Volume)
serving/           FastAPI orchestration layer (real-time + batch trigger)
tests/
  unit/            no external calls, mocked
  integration/     runs against a real staging workspace
  smoke/           post-deploy sanity checks
databricks.yml     Asset Bundle definition (jobs, endpoints, targets)
.github/workflows/ CI + staging deploy + prod promotion
```
