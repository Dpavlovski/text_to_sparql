FROM python:3.13-slim

# Standard Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    POETRY_VERSION=1.7.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# Install system dependencies
# Added 'git' just in case some dependencies come from GitHub
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copy only dependency files first to leverage Docker layer caching
COPY pyproject.toml poetry.lock ./

# Install dependencies (skipping root package install)
RUN poetry install --no-ansi --no-root

# Copy the rest of the project files
COPY . .

# Ensure the results folders exist so Streamlit doesn't crash 
# (Even if they are empty in your repo)
RUN mkdir -p results/benchmark/raw results/benchmark/processed

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "src/streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]