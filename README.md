# attrition-promotion-databricks

Two models — Employee Attrition Risk and Promotion Readiness — trained,
registered, and served entirely on Databricks, orchestrated through a single
FastAPI layer, with CI/CD via GitHub Actions.

## Design decisions locked in

- Data: synthetic, generated fresh (`data_generation/`) — **not committed to
  git**. Data generation runs as its own step and writes to a Unity Catalog
  Volume, so training always reads from the Volume, never from a local CSV
  baked into the repo. (Avoids the classic .gitignore-excludes-the-data
  gotcha from earlier projects.)
- Training: on Databricks (Jobs), MLflow tracking, registered to Unity
  Catalog. Champion-alias promotion only after beating the current champion.
- Serving: **two separate** Databricks Model Serving endpoints
  (`attrition-endpoint`, `promotion-endpoint`) — not one endpoint with two
  served entities.
- Batch inference: triggered **on-demand via FastAPI** calling the Databricks
  Jobs API (not schedule-only). Supports CSV upload from a frontend.
- Environments: `dev` / `staging` / `prod` as separate Unity Catalog catalogs
  via Databricks Asset Bundle targets. Prod promotion requires manual
  approval in GitHub.

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
