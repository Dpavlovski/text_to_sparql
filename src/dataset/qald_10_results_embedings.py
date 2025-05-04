import json
import re
from typing import List, Optional


def extract_qald_query_ids() -> Optional[List[str]]:
    file_path = r"C:\Users\User\PycharmProjects\text_to_sparql\src\dataset\qald_10.json"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in the file.")
        return None

    questions = data.get("questions", [])
    extracted_items = set()

    for question in questions:
        sparql_query = question.get("query", {}).get("sparql", "")
        if sparql_query:
            extracted_items.update(re.findall(r"(?:wd:|wdt:)([QP]\d+)", sparql_query))

    return extracted_items if extracted_items else None


print(extract_qald_query_ids())
