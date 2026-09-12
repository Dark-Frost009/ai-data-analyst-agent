# syntax=docker/dockerfile:1

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies are copied separately so Docker can reuse this layer when
# application code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Supply GROQ_API_KEY and APP_ACCESS_PASSWORD privately at runtime.
# Never copy a local .env or secrets file into the image.
COPY app ./app

# Include the non-secret Streamlit theme and server configuration.
# .dockerignore continues to exclude .streamlit/secrets.toml.
COPY .streamlit ./.streamlit

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
