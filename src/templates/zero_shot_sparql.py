def zero_shot_sparql(question: str) -> str:
    return f"""You are an AI that generates precise SPARQL queries to answer the given question related to the Wikidata knowledge graph.
The generated query will be later executed directly on the Wikidata API by the user, so the query mustn't contain prefixes or services.
You must only return the sparql query and nothing else.


Question:
{question}

Output:
{{
    "sparql": ""  
}}

Return in json format with key 'sparql'.
"""
