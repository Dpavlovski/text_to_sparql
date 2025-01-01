import json

from text_to_sparql.llm.chat import chat_with_ollama


def extract_and_refine_wikidata_keywords(question):
    combined_prompt = f"""
        You are an AI specializing in extracting and refining entities and relations from questions to align with Wikidata concepts. 
        Your task is to:

        1. Extract entities and relations from the question.
        2. Refine these entities and relations to their most likely Wikidata-compatible names.

        **Instructions:**
        - Entities: Specific people, places, items, concepts, or other relevant subjects in the question.
        - Relations: Actions, associations, or connections described in the question.
        - Refine all extracted entities and relations to concise keywords matching Wikidata's naming conventions, maintaining their original meanings.

        **Response Format (strict JSON format):**
        {{
            "entities": ["refined_entity1", "refined_entity2", ...],
            "relations": ["refined_relation1", "refined_relation2", ...]
        }}

        **Examples:**

        1. **Input Question**: "Who is the president of the United States?"
           **Output**:
           {{
               "entities": ["President of the United States"],
               "relations": ["is"]
           }}

        2. **Input Question**: "Where was Edison born?"
           **Output**:
           {{
               "entities": ["Thomas Edison"],
               "relations": ["born"]
           }}

        3. **Input Question**: "What are the capitals of European countries?"
           **Output**:
           {{
               "entities": ["capital", "European countries"],
               "relations": ["are"]
           }}

        Now, analyze and refine the following question to return the entities and relations in the required format:

        Question: {question}
    """
    response = chat_with_ollama(combined_prompt,
                                system_message="You are a Wikidata entity and relation extraction assistant.")
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("Error decoding the response. Ensure the response is in valid JSON format.")
        return {"entities": [], "relations": []}
