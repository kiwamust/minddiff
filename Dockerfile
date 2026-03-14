FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies
COPY pyproject.toml .
RUN uv sync --no-dev --no-install-project

# Copy source
COPY src/ src/
COPY scripts/ scripts/
RUN uv sync --no-dev

# Create data directory for SQLite
RUN mkdir -p /data

ENV DATABASE_URL=sqlite:////data/minddiff.db

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "minddiff.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
