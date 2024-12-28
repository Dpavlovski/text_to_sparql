import json

from text_to_sparql.llm.chat import chat_with_ollama


def extract_entities_and_relations(question):
    ner_prompt = f"""
        You are an AI that extracts entities and relations from a given question. 
        Your task is to analyze the question and identify:
        - **Entities**: Specific people, places, items, concepts, or other relevant subjects mentioned in the question.
        - **Relations**: Actions, associations, or connections described in the question.

        Your response must strictly adhere to the following JSON format:
        {{
            "entities": ["entity1", "entity2", ...],
            "relations": ["relation1", "relation2", ...]
        }}

        Make sure to:
        - Only include valid entities and relations extracted from the question.
        - Avoid adding explanations, comments, or additional text outside the JSON structure.

        **Examples:**

        1. **Input Question**: "Who is the president of the United States?"
           **Output**:
           {{
               "entities": ["president", "United States"],
               "relations": ["is"]
           }}

        2. **Input Question**: "Where was Edison born?"
           **Output**:
           {{
               "entities": ["Edison"],
               "relations": ["was born"]
           }}

        3. **Input Question**: "What are the capitals of European countries?"
           **Output**:
           {{
               "entities": ["capitals", "European countries"],
               "relations": ["are"]
           }}

        Now, analyze the following question and return the extracted entities and relations in the required JSON format:

        Question: {question}
        """
    response = chat_with_ollama(ner_prompt, system_message="You are an NER assistant.")
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("Error decoding the NER response. Ensure the response is in valid JSON format.")
        return {"entities": [], "relations": []}
