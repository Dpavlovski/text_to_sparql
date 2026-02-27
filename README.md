# Text-to-SPARQL Project

A system that translates natural language questions into SPARQL queries and evaluates them against the QALD (Question
Answering over Linked Data) dataset. The project integrates LLMs (via LangChain and LangGraph), vector databases (
Qdrant), and Wikidata to provide accurate SPARQL generation and entity/relation linking.

## Features

- **Multi-step SPARQL Generation**: Uses a LangGraph-based agent to rephrase questions, extract entities/relations, and
  generate queries.
- **Entity and Relation Linking**: Identifies entities and relations using NER and links them to Wikidata items and
  properties using vector search.
- **QALD Dataset Support**: Built-in support for QALD-10 benchmarks (English, Chinese, German, Russian, etc.).
- **Vector Database Integration**: Uses Qdrant for storing and searching Wikidata labels and descriptions.
- **Streamlit Dashboard**: Interactive UI for testing the agent, analyzing outputs, and viewing benchmarks.
- **Evaluation Pipeline**: Compares generated SPARQL results with ground truth to assess correctness.
- **Dockerized Infrastructure**: Easily deployable with Docker and Docker Compose.

## Tech Stack

- **Language**: Python 3.13+
- **Frameworks**: LangChain, LangGraph, Streamlit
- **Package Manager**: Poetry
- **Databases**:
  - **Qdrant**: Vector database for entity/relation linking.
- **External APIs**: Wikidata SPARQL Endpoint, OpenRouter/OpenAI/Ollama for LLMs.

## Requirements

- Python 3.13+
- [Poetry](https://python-poetry.org/docs/#installation)
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)

## Setup

### 1. Environment Variables

Create a `.env` file in the root directory based on the following template (see `.env` for examples):

```env
# LLM Configuration
CHAT_MODEL=openai # or ollama, openrouter
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4.1-mini

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Other APIs
=OPENROUTER_API_KEY=your_openrouter_key
```

### 2. Installation

```bash
# Install dependencies
poetry install
```

### 3. Run Infrastructure

```bash
# Start Qdrant and the application using Docker Compose
docker-compose up -d
```

## Usage

### Streamlit Dashboard

The main interface for the project is a Streamlit app:

```bash
poetry run streamlit run src/streamlit/app.py
```

This provides:

- **Chat Agent**: Interactive natural language to SPARQL interface.
- **Output Analysis**: Tools for analyzing generated queries.
- **Benchmarks**: Visualization of performance on datasets.

### Running Benchmarks

To run the benchmark script directly:

```bash
poetry run python src/main.py
```

*Note: Ensure Qdrant is running and populated before running benchmarks.*

### Data Population

To insert Wikidata labels into Qdrant:

```bash
poetry run python src/databases/qdrant/insert_wikidata_labels.py
```

## Project Structure

- `src/`
  - `agent/`: LangGraph agent definition, prompts, and state management.
  - `databases/`: Qdrant interaction logic.
  - `dataset/`: Parsers for QALD and LC-QuAD datasets.
  - `llm/`: LLM provider wrappers and embedding logic.
  - `streamlit/`: Streamlit multipage application.
  - `tools/`: Tools used by the SPARQL agent (NER, SPARQL execution, etc.).
  - `wikidata/`: Wikidata API clients and dump processing scripts.
- `results/`: Benchmark outputs, analysis files, and GERBIL evaluation results.
- `qdrant_storage/`: Local storage for Qdrant data.

## Scripts

- `src/main.py`: Main entry point for running benchmarks.
- `src/databases/qdrant/insert_wikidata_labels.py`: Populates Qdrant with Wikidata labels.
- `src/wikidata/dump_processing/preprocess_dump.py`: Scripts for processing Wikidata JSON dumps.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **QALD Dataset**: [https://qald.aksw.org/](https://qald.aksw.org/)
- **Wikidata**: [https://www.wikidata.org/](https://www.wikidata.org/)
- **LangChain/LangGraph**: [https://www.langchain.com/](https://www.langchain.com/)

