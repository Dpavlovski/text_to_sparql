def format_results(entity_results, relation_results):
    entity_descriptions = "\n".join(
        [f"Entity: {e['label']} (ID: {e['id']}) - {e.get('description', 'No description available')}"
         for e in entity_results]
    )
    relations_descriptions = "\n".join(
        [f"Relation: {e['label']} (ID: {e['id']}) - {e.get('description', 'No description available')}"
         for e in relation_results]
    )
    return entity_descriptions, relations_descriptions
