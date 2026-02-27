import json
from typing import Any, Optional

# Mapping based on your specific file structure
# 0=en, 1=zh, 2=de.
# We assume 'mk' might be explicitly labeled or at a specific index.
# If your JSON has "language": "mk" tags, the loop below handles it automatically.
INDEX_FALLBACK = {
    "en": 0,
    "zh": 1,
    "de": 2,
    # Add others if known, e.g., "ru": 3
}


def get_question_string(question_list: list[dict], target_lang: str) -> Optional[str]:
    """
    Tries to find the question string for the target language.
    1. Looks for explicit 'language' key match (Standard QALD).
    2. Falls back to hardcoded indices if provided.
    3. Defaults to English (index 0) if target not found (optional).
    """
    # 1. Try to find by ISO code tag (Robust method)
    for entry in question_list:
        if entry.get("language") == target_lang:
            return entry["string"]

    # 2. Fallback to specific indices (Your requested method)
    if target_lang in INDEX_FALLBACK:
        idx = INDEX_FALLBACK[target_lang]
        if idx < len(question_list):
            return question_list[idx]["string"]

    # 3. Special handling for 'mk' if it was appended manually without a tag
    # Assuming 'mk' might be the last element if not found above
    # Uncomment if needed:
    # if target_lang == "mk" and len(question_list) > 0:
    #     return question_list[-1]["string"]

    return None


def load_qald_json(lang: str = "en") -> list[dict[str, Any]]:
    """
    Loads QALD data and filters for the specific language.
    """
    # Adjust path if necessary
    file_path = "../qald_10_with_mk.json"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    rows = []

    for question in questions:
        # Extract the question string for the requested language
        q_string = get_question_string(question['question'], lang)

        # If translation is missing for this specific question, skip it or fallback
        if not q_string:
            # Option A: Skip
            # continue
            # Option B: Fallback to EN (Uncomment to enable)
            q_string = question['question'][0]['string'] + " (Fallback EN)"

        ground_truth_sparql = question['query']['sparql']

        # Parse answers
        expected_result = "No results found."
        answers = question.get('answers', [])

        if answers:
            first_answer = answers[0]
            if 'boolean' in first_answer:
                expected_result = first_answer['boolean']
            elif 'results' in first_answer:
                results = first_answer['results']
                if 'bindings' in results and results['bindings']:
                    expected_result = [
                        binding['result']['value']
                        for binding in results['bindings']
                        if 'result' in binding and 'value' in binding['result']
                    ]

        row = {
            "question": q_string,
            "ground_truth_sparql": ground_truth_sparql,
            "expected_result": expected_result,
        }
        rows.append(row)

    print(f"Loaded {len(rows)} questions for language: {lang}")
    return rows