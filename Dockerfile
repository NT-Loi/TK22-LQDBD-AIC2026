FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install system dependencies (OpenCV libraries and curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Enable bytecode compilation and use copy mode for uv cache linking
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy dependency configuration files first for Docker layer caching
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv sync without installing the project root yet
RUN uv sync --frozen --no-install-project

# Copy application source code
COPY . .

# Final project sync (registers editable package if needed)
RUN uv sync --frozen

# Ensure virtualenv binaries are directly accessible in PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose default FastAPI port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Launch FastAPI via uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
