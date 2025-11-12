import asyncio
import textwrap

from src.wikidata.api import execute_sparql_query

SCHEMA_QUERY_TEMPLATE = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT
  ?entityLabel
  ?directClass ?directClassLabel
  ?superClass ?superClassLabel
  ?relevantProperty ?relevantPropertyLabel
  ?allowedClassLabel
  ?propertyDatatypeLabel
WHERE {{
  BIND(wd:{entity_id} AS ?entity)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?entity rdfs:label ?entityLabel . }}

  OPTIONAL {{
    ?entity wdt:P31 ?directClass .
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?directClass rdfs:label ?directClassLabel . }}

    OPTIONAL {{
      ?directClass wdt:P279* ?superClass .
      FILTER(?directClass != ?superClass)
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?superClass rdfs:label ?superClassLabel . }}
    }}

    OPTIONAL {{
      {{ ?directClass p:P2302 ?statement . }}
      UNION
      {{ ?superClass p:P2302 ?statement . }}
      ?statement ps:P2302 ?relevantProperty .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?relevantProperty rdfs:label ?relevantPropertyLabel . }}

      OPTIONAL {{ ?statement pq:P2308 ?allowedClass . SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?allowedClass rdfs:label ?allowedClassLabel . }} }}
    }}
  }}

  OPTIONAL {{
    ?entity ?p_direct ?someValue .
    ?p_prop wikibase:directClaim ?p_direct ;
            wikibase:propertyType ?propertyDatatype .
    BIND(?p_direct AS ?relevantProperty)
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en".
                              ?relevantProperty rdfs:label ?relevantPropertyLabel .
                              ?propertyDatatype rdfs:label ?propertyDatatypeLabel . }}
  }}
}}
LIMIT 5
"""


async def get_entity_schema(entity_id: str) -> str:
    """
    Queries Wikidata to get the "schema" for a given entity and formats it
    as a text context for an LLM.
    """
    query = SCHEMA_QUERY_TEMPLATE.format(entity_id=entity_id)
    print(query)

    try:
        results = await execute_sparql_query(query)
    except Exception as e:
        return f"An error occurred during schema fetch: {e}"

    if not results:
        return f"No schema information found for entity {entity_id}."

    # --- Formatting Logic ---
    # We now get a flat list of results, so we need to aggregate them.
    entity_label = results[0].get("entityLabel", {}).get("value", entity_id)

    classes = set()
    properties = {}  # Use a dict to avoid duplicate properties

    for res in results:
        # Add class label to a set
        if class_label := res.get("classLabel", {}).get("value"):
            classes.add(class_label)

        # Aggregate property information
        if prop_uri := res.get("property", {}).get("value"):
            if prop_uri not in properties:
                properties[prop_uri] = {
                    "label": res.get("propertyLabel", {}).get("value", ""),
                    "description": res.get("propertyDescription", {}).get("value", ""),
                    "datatype": res.get("propertyDatatypeLabel", {}).get("value", "")
                }

    # --- Build the final LLM-friendly string ---
    context_str = f"Context for entity: '{entity_label}' ({entity_id})\n"
    context_str += "=" * (len(context_str) - 1) + "\n"

    # 1. Format Classes
    if classes:
        context_str += f"This entity is an instance or subclass of: {', '.join(sorted(list(classes)))}\n\n"

    # 2. Format Properties
    if properties:
        context_str += "It has the following properties:\n"
        # Sort properties by label for consistent output
        for prop_uri in sorted(properties, key=lambda p: properties[p]['label']):
            prop_data = properties[prop_uri]
            prop_id = prop_uri.split('/')[-1]  # Get P-number
            # Example line: "- occupation (P106): person's profession or role (type: Item)"
            line = f"- {prop_data['label']} ({prop_id}): {prop_data['description']} (type: {prop_data['datatype']})\n"
            context_str += line

    if not classes and not properties:
        context_str += "This entity has no structured class or property information available.\n"

    return textwrap.dedent(context_str).strip()

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Douglas Adams (a person)
    douglas_adams_id = "Q42"
    print(f"--- Schema for Douglas Adams ({douglas_adams_id}) ---")
    context = asyncio.run(get_entity_schema(douglas_adams_id))
    print(context)
    print("\n" + "=" * 80 + "\n")

    # Example 2: Doctor Who (a creative work)
    doctor_who_id = "Q34316"
    print(f"--- Schema for Doctor Who ({doctor_who_id}) ---")
    context = asyncio.run(get_entity_schema(doctor_who_id))
    print(context)
    print("\n" + "=" * 80 + "\n")

    # Example 3: Earth (a planet)
    earth_id = "Q2"
    print(f"--- Schema for Earth ({earth_id}) ---")
    context = asyncio.run(get_entity_schema(earth_id))
    print(context)
