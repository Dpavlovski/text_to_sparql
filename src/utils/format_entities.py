from typing import List

from qdrant_client.grpc import ScoredPoint


def format_entities(entities: List[ScoredPoint]) -> str:
    output = ""
    for entity in entities:
        output += f"""
        Entity ID: {entity.payload.get('id', 'N/A')}
        Label: {entity.payload.get('value', 'N/A')}
        """
    return output
