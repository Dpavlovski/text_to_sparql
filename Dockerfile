#FROM python:3.13-slim
#
#ENV PYTHONDONTWRITEBYTECODE=1 \
#    PYTHONUNBUFFERED=1 \
#    PYTHONPATH=/app \
#    POETRY_VERSION=1.7.1 \
#    POETRY_VIRTUALENVS_CREATE=false \
#    POETRY_NO_INTERACTION=1
#
#WORKDIR /app
#
#
#RUN apt-get update && apt-get install -y \
#    curl \
#    build-essential \
#    git \
#    && rm -rf /var/lib/apt/lists/*
#
#RUN pip install "poetry==$POETRY_VERSION"
#
#COPY pyproject.toml poetry.lock ./
#
#RUN poetry install --no-ansi --no-root
#
#COPY . .
#
#RUN mkdir -p results/benchmark/raw results/benchmark/processed

FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN pip install --no-cache-dir streamlit pandas plotly

# Copy the specific CSV file
COPY results/benchmark/with_neighbors/processed/en_gpt-_4.1-mini_v2.csv ./results/benchmark/with_neighbors/processed/en_gpt-_4.1-mini_v2.csv

# Copy the source code for the Streamlit app
COPY src/ ./src/
