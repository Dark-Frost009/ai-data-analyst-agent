# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies are copied separately so Docker can reuse this layer when
# application code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The application reads credentials from the standard AWS provider chain at
# runtime. Do not copy a local .env file or AWS credential files into image.
COPY app ./app

# Include the non-secret Streamlit theme and server configuration.
# .dockerignore continues to exclude .streamlit/secrets.toml.
COPY .streamlit ./.streamlit

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
