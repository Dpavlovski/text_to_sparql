import asyncio
import time

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from pydantic import Field

from src.agent.prompts import validation_prompt
from src.databases.qdrant.search_embeddings import fetch_similar_qa_pairs, get_candidates
from src.llm.llm_provider import llm_provider
from src.tools.ner import extract_entities
from src.tools.schema import get_entity_schema
from src.tools.sparql import get_sparql_query


@tool("generate_sparql_query")
async def generate_sparql(
        question: str = Field(description="The possibly rephrased natural language question"),
        original_question: str = Field(description="The original question before any modification"),
) -> dict:
    """
    Generates a SPARQL query from a natural language question by enriching it with
    named entities and relevant examples.

    Args:
        question (str): The user's natural language question.
        original_question (str): The original, unmodified question for logging.

    Returns:
        A dictionary containing the generated SPARQL query or an error message.
    """
    start_time = time.monotonic()

    # 1. Extract named entities
    ner_response = await extract_entities(question)

    # 2. Find similar entities to augment search
    candidates_map = await get_candidates(ner_response.keywords, ner_response.lang)

    # 3. Disambiguate and get confirmed entities
    # linked_entities = await disambiguate_entities(question, candidates_map)

    schema_tasks = []
    candidate_meta = []  # To keep track of which schema belongs to which candidate

    for keyword, candidate_list in candidates_map.items():
        # Iterate through the list of candidates for each keyword
        # Limit to top 3 to avoid overloading the prompt/API if the list is long
        for candidate in candidate_list[:3]:
            qid = candidate.get('id')
            label = candidate.get('label', 'Unknown')

            if qid:
                schema_tasks.append(get_entity_schema(qid))
                candidate_meta.append((keyword, label, qid))

    # Run all schema queries in parallel
    if schema_tasks:
        schema_results = await asyncio.gather(*schema_tasks)
    else:
        schema_results = []

    # 4. Build the Schema Context String
    schema_context = ""
    for i, (keyword, label, qid) in enumerate(candidate_meta):
        result_text = schema_results[i]
        # Only add to context if we found useful info
        if result_text and not result_text.startswith("# No schema"):
            schema_context += f"### Keyword: '{keyword}' -> Candidate: '{label}' ({qid})\n{result_text}\n\n"

    # 4. Retrieve examples to guide the LLM
    examples = await fetch_similar_qa_pairs(question)

    # 5. Generate the SPARQL query
    response = await get_sparql_query(question, examples, candidates_map, None)

    results_for_log = None
    raw_results = response.get('results')

    if raw_results is not None:
        if isinstance(raw_results, bool):
            results_for_log = str(raw_results)
        elif isinstance(raw_results, list):
            extracted_uris = []
            for result_row in raw_results:
                for key, value_dict in result_row.items():
                    if isinstance(value_dict, dict):
                        val = value_dict.get('value')
                        if val:
                            extracted_uris.append(val)
                            break
            results_for_log = "\n".join(extracted_uris)

    # Now use 'results_for_log' in your log_data dictionary
    execution_time = time.monotonic() - start_time

    log_data = {
        "original_question": original_question,
        "rephrased_question": question,
        "ner": str(ner_response),
        "candidates": str(candidates_map),
        "schema_context": str(schema_context),
        "examples": str(examples),
        "generated_query": str(response.get("sparql")),
        "result": results_for_log,
        "time": f"{execution_time:.2f}"
    }

    response['log_data'] = log_data

    return response


@tool("validate_results")
async def validate_results(question: str, results: str) -> bool:
    """
    Validates if the SPARQL query results answer the user's question.
    """
    prompt = PromptTemplate.from_template(validation_prompt).format(
        question=question,
        results=results
    )
    llm = llm_provider.get_model(model_identifier="kwaipilot/kat-coder-pro:free").with_structured_output(
        bool
    )
    response = await llm.ainvoke(prompt)
    return "yes" in response.content.lower()
