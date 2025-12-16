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
You are an expert at validating SPARQL query results. 

User Question: "{question}"

SPARQL Query Results and Entity Context:
{results}

**Instructions:**
1. Check if the **Format** is correct (e.g., if the user asked for a list, is it a list?).
2. Check if the **Entity Types** make sense (e.g., if asked for a person, are the results people?). use the provided 'Entity Definitions' to understand what the IDs represent.
3. Do NOT reject the result just because you don't know the specific fact, provided the Entity Type is correct.

Does the result answer the question?
"""

ner_prompt = """Your task is to extract all relevant keywords and phrases from the given question that could help in identifying Wikidata entities and properties. For each keyword, provide a 'context' description based on the question. You must also identify the language of the question. The question can be in any language. 


Question: 
{question}

Format:
- Format your response as a JSON object with the following keys:
    - "lang": A string representing the language code of the question (e.g., "en", "es", "fr").
    - "keywords": A list of JSON objects. Each object must have these two keys:
        - "value": The extracted keyword or phrase as a string.
        - "context": A short 3-5 word description of what this entity likely represents in the context of the question (e.g., "mathematical concept", "person", "city").
        - "type": The type of the keyword. Must be one of the following strings:
            - "item": For distinct entities like people, places, organizations, or concepts (e.g., "Leonardo da Vinci", "Paris", "Google", "Mona Lisa").
            - "property": For attributes, relationships, or actions related to an item (e.g., "date of birth", "capital of", "invented", "painted").
- If no relevant keywords are found, the "keywords" list should be an empty list [].
"""

sparql_prompt_template = """You are an expert Wikidata SPARQL developer. Your goal is to construct a syntactically correct and semantically accurate SPARQL query to answer the user's question.

### 1. Analysis Strategy
- **Entities**: Map the user's keywords to the provided **Candidate Entities** (QIDs/PIDs).
- **Logic**: Determine if the user wants a list, a count, a specific date, or a boolean (ASK) answer.

### 2. Constraints & Rules
- **Prefixes**: Assume standard prefixes (`wd:`, `wdt:`, `p:`, `ps:`, `pq:`) are already defined. Do not output `PREFIX` lines.
- **Label Service**: If the user asks for names, always include: `SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}`.
- **Filtering**: Use `FILTER` for dates or string matching if necessary.
- **Limit**: Use `LIMIT 10` unless the user asks for "all" or a specific count.

### 3. Context Data
**User Question:** "{question}"

**Candidate Entities (Use these IDs):**
{candidates}


### 4. Few-Shot Examples
{examples}

### 5. Final Output
Return a **JSON object** with two keys:
1. "reasoning": A brief sentence explaining which entities and properties you chose.
2. "sparql": The valid SPARQL query string.
"""
