# Stage 1: Builder
FROM python:3.11-slim as builder
WORKDIR /app
RUN pip install --no-cache-dir poetry poetry-plugin-export
COPY pyproject.toml poetry.lock* ./
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /app/requirements.txt .

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# EXPLICIT INSTALL: Guaranteeing the frontend dependencies exist
RUN pip install --no-cache-dir -r requirements.txt streamlit requests pandas streamlit-authenticator PyYAML openpyxl

COPY src ./src
COPY config ./config
COPY artifacts ./artifacts
COPY data ./data

EXPOSE 8501
# Bypass Linux $PATH issues by running as a Python module
CMD ["python", "-m", "streamlit", "run", "src/app/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]