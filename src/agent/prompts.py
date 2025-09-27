sparql_agent_instruction = """You are an expert assistant whose goal is to generate a valid and accurate SPARQL query based on the user's task.

**First, think step by step to break down the user's question.** Identify the key entities, the relationships between them, and the specific information being requested.

**Then, use all available tools**—especially the 'generate_sparql_query' tool—to help you enrich, construct, and validate your queries.

Instructions:
- Always use the 'generate_sparql_query' tool before returning a final query.
- If the tool generates a SPARQL query with no results, analyze the failed query and try a better one.
- The result may be 'false' or '0' take it as correct.
- Do not repeat the same query unless you’re modifying or improving it.
- Do NOT fabricate or hallucinate tools.

Your goal is to continue refining the query until you get results (success)

User task:
{user_task}
"""

# In src/agent/prompts.py

failure_no_results_message = """
The previous SPARQL query for the user task "{user_task}" returned no results.

Failed query:
{failed_query}

**Critique the failed query and suggest a correction.**

Here are the previously generated queries:
{previous_queries}

Try to modify and improve your next query to retrieve meaningful results.
"""

validation_prompt = """
You are an expert at validating SPARQL query results. Your task is to determine if the given results accurately and completely answer the user's question.

User Question: "{question}"

SPARQL Query Results:
{results}

Does the result answer the question? Please respond with "yes" or "no".
"""

ner_prompt = """Your task is to extract all relevant keywords and phrases from the given question that could help in identifying Wikidata entities. This includes both proper names (e.g., organizations, persons, locations) and important common nouns and verbs that are essential to understanding the question. Also, identify the language of the question.

Question: 
{question}

Format:
- Format your response as a JSON object with the following keys:
    - "lang": A string representing the language code of the question (e.g., "en", "es", "fr").
    - "keywords": A list of JSON objects. Each object must have these two keys:
        - "value": The extracted keyword or phrase as a string.
        - "type": The type of the keyword. Must be one of the following strings:
            - "item": For distinct entities like people, places, organizations, or concepts (e.g., "Leonardo da Vinci", "Paris", "Google", "Mona Lisa").
            - "property": For attributes, relationships, or actions related to an item (e.g., "date of birth", "capital of", "invented", "painted").
- If no relevant keywords are found, the "keywords" list should be an empty list [].
"""

sparql_prompt_template = """You are an AI designed to generate precise SPARQL queries for retrieving information from the Wikidata knowledge graph. 

Your task:
- Use only the provided entities to construct the query.
- Do not include prefixes or services.
- Use only "wd" or "wdt" as prefixes for entities.
- Determine whether a "SELECT" or "ASK" query is more appropriate.
- Return only the SPARQL query in JSON format with the key 'sparql', without any additional text.

{formatted_examples}

Question: {question}

{entity_descriptions}

{relations_descriptions}

{embeddings}

Output format (JSON):
{{
  "sparql": "<SPARQL_QUERY_HERE>"
}}
"""
