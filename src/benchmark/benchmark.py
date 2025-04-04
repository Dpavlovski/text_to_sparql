from typing import Any

from matplotlib import pyplot as plt

from src.databases.mongo.models.Embedding_QALD_10 import EmbeddingQALD10
from src.databases.mongo.mongo import MongoDBDatabase
from src.dataset.qald_10 import load_qald_json
from src.main import text_to_sparql


def test_on_qald_10(benchmark_data: list[dict[str, Any]]):
    db = MongoDBDatabase()
    for data in benchmark_data:
        question = data['question']
        ground_truth_sparql = data['ground_truth_sparql']
        expected_result = data['expected_result']

        generated_sparql, result = text_to_sparql(question)

        entry = EmbeddingQALD10(
            question=question,
            ground_truth_sparql=ground_truth_sparql,
            expected_result=expected_result,
            generated_sparql=generated_sparql,
            result=result
        )

        db.add_entry(entity=entry)


def evaluate_correctness():
    db = MongoDBDatabase()
    collection_name = "text-to-sparql"

    try:
        entries = db.get_entries_dict(collection_name=collection_name)
        count = sum(1 for entry in entries if entry.get("correct_result") is True)
        return count

    except Exception as e:
        print(f"An error occurred while evaluating correctness: {e}")
        return 0


def calculate_accuracy():
    db = MongoDBDatabase()
    collection_name = "text-to-sparql"

    try:
        entries = db.get_entries_dict(collection_name=collection_name)
        total = len(entries)
        correct = sum(1 for entry in entries if entry.get("correct_result") is True)
        accuracy = correct / total if total > 0 else 0
        return accuracy

    except Exception as e:
        print(f"An error occurred while calculating accuracy: {e}")
        return 0


def plot_results():
    db = MongoDBDatabase()
    collection_name = "text-to-sparql"

    try:
        entries = db.get_entries_dict(collection_name=collection_name)
        correct = sum(1 for entry in entries if entry.get("correct_result") is True)
        incorrect = len(entries) - correct

        labels = ['Correct', 'Incorrect']
        sizes = [correct, incorrect]
        colors = ['green', 'red']

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title('Query Evaluation Results')
        plt.show()

    except Exception as e:
        print(f"An error occurred while plotting results: {e}")


test_on_qald_10(load_qald_json())
# print("Correct queries: " + str(evaluate_correctness()) + "/" + str(len(load_qald_json())))
# print("Accuracy: " + str(calculate_accuracy()))
# plot_results()
