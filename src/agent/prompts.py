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

ner_prompt = """Your task is to extract all relevant keywords and phrases from the given question that could help in identifying Wikidata entities and properties. You must also identify the language of the question. The question can be in any language.

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

sparql_prompt_template = """You are a highly specialized AI that converts natural language questions into precise SPARQL queries for Wikidata.

**Your Instructions:**

1.  **Analyze the User's Question**.
2.  **Use Provided Confirmed Entities**: You have been given the exact entities to use for the query. You MUST use these IDs. Do not search for other entities.
3.  **Strict SPARQL Syntax**: Use `wd:` for items and `wdt:` for properties.
4.  **Strict Output Format**: Your final output must be a single JSON object with only the "sparql" key.

{examples}

**User's Question:** {question}

**Confirmed Entities for Query Construction:**
{linked_entities_context}

**Required Output (JSON only):**
{{
  "sparql": "YOUR_SPARQL_QUERY_GOES_HERE"
}}
"""

disambiguation_prompt_template = """You are an expert entity disambiguation AI. Your task is to select the single correct entity for each mention from a list of candidates, based on the user's question.

Analyze the user's question to understand its context. Then, for each mention, review its list of candidates. Choose the candidate whose description best fits the context.

User Question: "{question}"

**Candidates:**
{formatted_candidates}

Your response MUST be a JSON object that maps each mention to its single, correct entity ID.
Example response: {{"Paris": "wd:Q90", "France": "wd:Q142"}}
"""
