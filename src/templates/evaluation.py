def evaluation_template(true_query, generated_query, true_answer, got_answer):
    return f"""Your task is to evaluate the correctness of both the SPARQL query and its resulting output.

1. Compare the following SPARQL queries and determine if the generated query is equivalent to the true query, even if the structure or syntax differs slightly.
- True Query: 
{true_query}
- Generated Query:
{generated_query}

2. Evaluate the results of these queries and determine if the they are correct, considering semantic similarity. 
If the expected result appears among the returned results, consider the answer as correct. If the expected result is true/false and the returned result is something else consider it as incorrect.
- Expected Answer:
{true_answer}
- Gotten Answer:
{got_answer}

Output:
{{
 "Query Equivalent": (true/false)
 "Result Equivalent": (true/false)
 "Reason": 
}}

Return only in JSON format and nothing else.
"""
