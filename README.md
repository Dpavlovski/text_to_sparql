# Text-to-SPARQL Project

This project aims to build a system that translates natural language questions into SPARQL queries and evaluates their correctness using the QALD (Question Answering over Linked Data) dataset. The system supports querying Wikidata and integrates various modules for natural language processing, SPARQL generation, and result evaluation.

## Features

- **QALD Data**cted results from the QALD JSON dataset.**set Parsing**: Extracts questions, SPARQL queries, and expe
- **Entity and Relation Linking**: Identifies entities and relations in the question using Named Entity Recognition (NER) and links them to Wikidata items and properties.
- **Enititis and Relations**: are searched on Wikidata endpoint for finding the most similar.
- **SPARQL Query Generation**: Generates SPARQL queries based on extracted entities, relations, and question templates.
- **Result Evaluation**: Compares generated results with ground truth results to assess the correctness of the query.
- **Database Integration**: Saves and retrieves benchmark results to/from a MongoDB database.

## Requirements

- **Python 3.8+**
- Poetry
- Docker

## Usage

### 1. Load QALD Dataset

The `load_qald_json()` function extracts questions, SPARQL queries, and expected results from the QALD dataset:

```python
from src.dataset.qald_10 import load_qald_json

benchmark_data = load_qald_json()
print(benchmark_data)
```

### 2. Run Evaluation

The `test_on_qald_10()` function processes the benchmark data, generates SPARQL queries, and evaluates their correctness:

```python
from src.main import test_on_qald_10

benchmark_data = load_qald_json()
test_on_qald_10(benchmark_data)
```

### 3. MongoDB Integration

Results are automatically saved to MongoDB in the `text-to-sparql` collection. Verify entries using a MongoDB client:

```bash
mongo
use text_to_sparql
db.getCollection('text-to-sparql').find()
```

## Project Structure

- `src/`
  - `databases/`: Handles MongoDB interactions and entity/relation searches.
  - `dataset/`: Contains dataset parsing logic.
  - `main.py`: Entry point for testing and integration.
  - `utils/`: Contains utility functions for formatting and processing results.
  - `wikidata/`: Interfaces with the Wikidata API for entity and relation linking.

## Example Workflow

1. Load the QALD dataset.
2. Perform entity and relation linking using NER.
3. Generate SPARQL queries and fetch results from Wikidata.
4. Compare the generated results with ground truth results.
5. Save results and evaluation metrics to MongoDB.

## Troubleshooting

### MongoDB Connection Error

- Ensure MongoDB is running and accessible.
- Verify the connection string in `MongoDBDatabase`.

### Invalid SPARQL Queries

- Check the entity and relation linking results.
- Verify SPARQL query templates in `perform_multi_querying_with_ranking()`.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## Acknowledgments

- QALD Dataset: [https://qald.aksw.org/](https://qald.aksw.org/)
- Wikidata: [https://www.wikidata.org/](https://www.wikidata.org/)

