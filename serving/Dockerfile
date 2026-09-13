# Multi-stage build: keep the final image small
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

# bring in the installed packages from the builder stage
COPY --from=builder /install /usr/local

COPY app/ ./app/

# runtime env vars (DATABRICKS_HOST, DATABRICKS_TOKEN, SERVING_API_KEY) are
# injected at container run time (docker run -e ... / k8s secret), never
# baked into the image
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=3).raise_for_status()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]