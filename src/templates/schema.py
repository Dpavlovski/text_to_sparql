import textwrap
from collections import defaultdict

from src.wikidata.api import execute_sparql_query


def get_entity_schema(entity_id: str) -> str:
    """
    Queries Wikidata to get the "schema" for a given entity and formats it
    as a text context for an LLM.

    The schema includes the entity's direct classes, superclasses, and relevant
    properties with their expected data types and constraints.

    Args:
        entity_id: The Wikidata QID of the entity (e.g., "Q42").

    Returns:
        A formatted string describing the entity's schema, or an error message.
    """
    # SPARQL query to fetch the schema context for a given entity
    query = f"""
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

    try:
        # The key change is here: process the full JSON response.
        results = execute_sparql_query(query)
    except Exception as e:
        # Catch a more generic exception if the helper function has its own errors.
        return f"An error occurred: {e}"

    if not results:
        return f"No information found for entity {entity_id}."

        # --- Formatting Logic ---
        # Correctly extract the entity label from the first result row.
    entity_label = results[0]

    schema = defaultdict(lambda: {
        'superclasses': set(),
        'properties': defaultdict(lambda: {'types': set(), 'constraints': set()})
    })

    # Process the list of result dictionaries. This loop should now work correctly.
    for res in results:
        class_label = res[1]
        if not class_label:
            continue

        prop_label = res[2]
        if not prop_label:
            continue

        superclass_label = res[3]
        if superclass_label:
            schema[class_label]['superclasses'].add(superclass_label)

        prop_type = res[4]
        if prop_type:
            schema[class_label]['properties'][prop_label]['types'].add(prop_type)

        prop_constraint = res[5]
        if prop_constraint:
            schema[class_label]['properties'][prop_label]['constraints'].add(prop_constraint)

    # --- Build the final LLM-friendly string ---
    context_str = f"Context for entity: {entity_label} ({entity_id})\n"
    context_str += "=" * (len(context_str) - 1) + "\n\n"

    if not schema:
        context_str += "This entity has no structured class or property information available.\n"
        return context_str

    for class_name, data in sorted(schema.items()):
        context_str += f"The entity is an instance of '{class_name}'.\n"

        if data['superclasses']:
            sorted_superclasses = sorted(list(data['superclasses']))
            context_str += f"  - It is a subclass of: {', '.join(sorted_superclasses)}\n"

        if data['properties']:
            context_str += "  - Relevant properties and their expected value types:\n"
            for prop in sorted(data['properties'].keys()):
                prop_details = data['properties'][prop]
                types = sorted(list(prop_details['types']))
                constraints = sorted(list(prop_details['constraints']))

                details_parts = []
                # Give precedence to the more specific 'constraint' info if it exists
                if constraints:
                    details_parts.append(f"should be an instance of '{', '.join(constraints)}'")
                elif types:
                    details_parts.append(f"is a '{', '.join(types)}' type")

                details_str = "; ".join(details_parts) if details_parts else "type not specified"
                context_str += f"    - {prop}: ({details_str})\n"
        context_str += "\n"

    return textwrap.dedent(context_str).strip()


# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Douglas Adams (a person)
    douglas_adams_id = "Q42"
    print(f"--- Schema for Douglas Adams ({douglas_adams_id}) ---")
    context = get_entity_schema(douglas_adams_id)
    print(context)
    print("\n" + "=" * 80 + "\n")

    # Example 2: Doctor Who (a creative work)
    doctor_who_id = "Q34316"
    print(f"--- Schema for Doctor Who ({doctor_who_id}) ---")
    context = get_entity_schema(doctor_who_id)
    print(context)
    print("\n" + "=" * 80 + "\n")

    # Example 3: Earth (a planet)
    earth_id = "Q2"
    print(f"--- Schema for Earth ({earth_id}) ---")
    context = get_entity_schema(earth_id)
    print(context)
