# NeuralRetail Enterprise Platform

Welcome to the **NeuralRetail** project! This repository contains the source code for our next-generation Enterprise Sales Intelligence and MLOps platform.

## 🚀 Current Project Status

**Status:** Infrastructure Initialized. Entering Data Engineering Phase.

We have successfully completed the foundation and local infrastructure phases. The project is currently configured as a monorepo with strict dependency management and a fully containerized local development sandbox.

### Completed Phases
- **Phase 0: Repository Foundation** - Monorepo setup, Poetry dependency locking, Ruff/Black pre-commit hooks. *(Completed by Sesha Sai)*
- **Phase 1: Local Infrastructure** - Profile-driven Docker architecture (Airflow, Marquez, Postgres, Redis, isolated MLflow on port 5002) operating under a strict 6GB WSL constraint. *(Completed by Sesha Sai)*

---

## 💻 Developer Setup & Installation

To run the current sandbox on your local machine, please follow these instructions carefully.

### Prerequisites
1. **Python 3.12+**: Required for our core ML and data engineering libraries.
2. **Poetry**: We use Poetry for dependency management. (`pip install poetry`)
3. **Docker Desktop**: Ensure you have Docker running. 
   * **Crucial Note for Windows Users:** You must allocate a strict **6GB RAM limit** in your WSL2 `.wslconfig` file. The infrastructure has been specifically tuned for this constraint.

### 1. Repository Setup

Clone the repository and install the Python dependencies:

```bash
git clone <repository_url>
cd NeuralRetail

# Install dependencies using Poetry
poetry install
```

### 2. Environment Configuration

You need a `.env` file to securely store your local configuration.

```bash
# Copy the example environment file
cp .env.example .env
```
*(Note: Do not commit your `.env` file!)*

### 3. Launching the Local Sandbox

We utilize a **Profile-Driven Docker Architecture** to stay within the 6GB memory constraint. **Do NOT run `docker compose up -d`.** Instead, use the provided `Makefile` to spin up only the environment you need for your phase.

```bash
# For Integration & MLOps (Boots Airflow, Marquez, Postgres, Redis)
make up-ops

# For Data Engineering (Boots Spark/Data Infrastructure)
make up-data

# For ML Engineering (Boots MLflow)
make up-ml
```

*To shut down your active sandbox and free up memory, run:*
```bash
make down
```

---

## ✅ How to Verify the Setup (Phases 0 & 1)

Once you have completed the installation, verify that the initial phases are working perfectly:

### Verify Phase 0 (Code Quality & Dependencies)
Run the following commands to ensure your local environment is correctly locked and formatted:
- **Check Dependencies:** `poetry check` and `poetry env info`
- **Check Linting:** `poetry run ruff check .`
- **Check Formatting:** `poetry run black --check .`

### Verify Phase 1 (Local Infrastructure)
Run `docker compose ps` to verify container health. You should see `postgres`, `redis`, and `airflow-webserver` listed as `Up (healthy)`.

Verify the UIs are accessible in your browser:
- **Apache Airflow:** [http://localhost:8080](http://localhost:8080)
- **MLflow Model Registry:** [http://localhost:5002](http://localhost:5002) *(Note: Port 5002 is intentional to avoid collisions)*
- **Marquez (Data Lineage):** [http://localhost:3000](http://localhost:3000)

---

## 🎯 Remaining Phases & Assignments

The following phases outline the roadmap for the rest of the project. Please coordinate with your respective team members.

- **Phase 2 & 3: Data Ingestion & Feast Feature Store** - PySpark transformations, Great Expectations data quality gates, and centralized feature serving. 
  - 👤 **Assignee:** Vaidehi
- **Phase 4a: Demand Forecasting Modeling** - PyTorch/Prophet time-series intelligence engines. 
  - 👤 **Assignee:** Karthikeyan
- **Phase 4b: Classification & Churn Modeling** - XGBoost/LightGBM with SHAP segmentation. 
  - 👤 **Assignee:** Ayush
- **Phase 5: Serving Layer** - Secure FastAPI endpoints with Redis caching for low-latency inference. 
  - 👤 **Assignee:** Pawan
- **Phase 6: Intelligence Dashboard** - Streamlit interactive UI for business stakeholders. 
  - 👤 **Assignee:** Nithish
- **Phases 7-12: Production MLOps** - CI/CD pipelines, Kubernetes deployment, model drift monitoring, and automated retraining. 
  - 👤 **Assignee:** Sesha Sai