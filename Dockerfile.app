FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY config ./config
COPY artifacts ./artifacts
COPY data ./data

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/app/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
