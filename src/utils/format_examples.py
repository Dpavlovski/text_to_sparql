from typing import List

from qdrant_client.http.models import ScoredPoint


def format_examples(examples: List[ScoredPoint]) -> str:
    if not examples:
        return ''
        
    template = ''
    for ex in examples:
        try:
            template += f"""
        Question:
            {ex.payload.get('value', 'No value available')}
        Output:
            {ex.payload.get('answer', 'No answer available')}
        """
        except (AttributeError, KeyError, TypeError) as e:
            # Skip examples with missing or invalid payload
            continue

    return template
