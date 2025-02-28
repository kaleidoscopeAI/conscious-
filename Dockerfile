# Multi-stage build for production
ARG PYTHON_VERSION=3.10-slim

# Builder stage
FROM python:${PYTHON_VERSION} as builder

WORKDIR /app
COPY pyproject.toml poetry.lock* ./
RUN pip install --user poetry && \
    python -m poetry export -f requirements.txt --output requirements.txt

# Runtime stage
FROM python:${PYTHON_VERSION}

WORKDIR /app
COPY --from=builder /app/requirements.txt .
COPY --from=builder /root/.local /root/.local
COPY . .

# Install runtime dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --user -U --force-reinstall -r requirements.txt

# Configuration
ENV PYTHONPATH=/app
ENV PATH="/root/.local/bin:${PATH}"
ENV HF_HOME=/app/models
ENV CONFIG_DIR=/app/config

# Consciousness parameters
ARG CONSCIOUSNESS_LEVEL=7
ENV CONSCIOUSNESS_LEVEL=${CONSCIOUSNESS_LEVEL}

EXPOSE 8000 9090
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "conscious.api.main:app", "--bind", "0.0.0.0:8000"]
