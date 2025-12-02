import asyncio

from src.wikidata.api import execute_sparql_query

SCHEMA_QUERY_TEMPLATE = """
SELECT ?value ?valueLabel WHERE {{
  wd:{entity_id} wdt:P31|wdt:P279 ?value.
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 10
"""


async def get_entity_schema(entity_id: str) -> str:
    """
    Queries Wikidata for the most important "is a" relationships (instance of/subclass of)
    and formats them as a clean, LLM-friendly text block.
    """
    query = SCHEMA_QUERY_TEMPLATE.format(entity_id=entity_id)

    try:
        results = await execute_sparql_query(query)
    except Exception as e:
        return f"# Error fetching schema for {entity_id}: {e}"

    if not results:
        return f"# No 'is a' relationships found for {entity_id}."

    is_a_relationships = set()
    for res in results:
        # The result from the corrected api.py will look like:
        # {'valueLabel': {'type': 'literal', 'xml:lang': 'en', 'value': 'taxon'}}
        qid = res.get("value", {}).get("value").removeprefix("http://www.wikidata.org/entity/")
        label = res.get("valueLabel", {}).get("value")
        if qid or label:
            is_a_relationships.add(qid + "-" + label)

    if not is_a_relationships:
        return f"# No labels found for the relationships of {entity_id}."

    # Build the final, clean string for the LLM
    context_str = f"# Context for entity {entity_id}:\n"
    context_str += f"- Is a type of: {', '.join(sorted(list(is_a_relationships)))}"

    return context_str

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    async def run_tests():
        # Test 1: "animal" (wd:Q729)
        animal_id = "Q729"
        print(f"--- Schema for animal ({animal_id}) ---")
        context = await get_entity_schema(animal_id)
        print(context)
        print("\n" + "=" * 80 + "\n")

        # Test 2: Douglas Adams (wd:Q42)
        douglas_adams_id = "Q42"
        print(f"--- Schema for Douglas Adams ({douglas_adams_id}) ---")
        context = await get_entity_schema(douglas_adams_id)
        print(context)
        print("\n" + "=" * 80 + "\n")

        # Test 3: military operation (wd:Q645883)
        mil_op_id = "Q645883"
        print(f"--- Schema for military operation ({mil_op_id}) ---")
        context = await get_entity_schema(mil_op_id)
        print(context)


    asyncio.run(run_tests())
