FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

COPY . .

ENV PYTHONPATH=/app
ENV PORT=8501

EXPOSE 8501

CMD streamlit run src/app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
