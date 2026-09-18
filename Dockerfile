# One image, two entrypoints (API and UI); docker-compose.yml runs both.
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so code changes don't invalidate the pip layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --create-home --uid 1000 appuser
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser ui ./ui
COPY --chown=appuser:appuser config ./config
COPY --chown=appuser:appuser samples ./samples
COPY --chown=appuser:appuser .streamlit ./.streamlit
RUN mkdir -p /app/data && chown appuser:appuser /app/data

USER appuser
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

# Default: the API. Two workers so one slow LLM call doesn't block health checks and short requests.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
