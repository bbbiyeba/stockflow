# Reproducible image for the batch pipeline and the dashboard. Results live in named volumes,
# so the pipeline container and the dashboard container share them:
#   docker build -t stockflow .
#   VOLS="-v sf-raw:/app/data/raw -v sf-processed:/app/data/processed -v sf-outputs:/app/outputs
#         -v sf-reports:/app/reports -v sf-decisions:/app/decisions"
#   docker run --rm $VOLS stockflow python run_pipeline.py              # ~9 min
#   docker run --rm -p 8501:8501 $VOLS --env-file .env stockflow       # dashboard
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# mount points exist in the image so named volumes inherit the non-root owner
RUN mkdir -p data/raw data/processed outputs reports decisions \
    && useradd --create-home --uid 10001 stockflow && chown -R stockflow /app
USER stockflow

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
