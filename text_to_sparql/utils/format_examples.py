from typing import List

from qdrant_client.http.models import ScoredPoint


def format_examples(examples: List[ScoredPoint]) -> str:
    template = ''
    for ex in examples:
        template += f"""
        Question:
            {ex.payload['value']}
        Output:
            {ex.payload['answer']}
        """

    return template
