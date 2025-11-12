import time

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from pydantic import Field

from src.agent.prompts import validation_prompt
from src.databases.qdrant.search_embeddings import fetch_similar_qa_pairs, get_candidates
from src.llm.generic_llm import generic_llm
from src.tools.ner import extract_entities
from src.tools.sparql import get_sparql_query, disambiguate_entities


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
    linked_entities = await disambiguate_entities(question, candidates_map)

    # 4. Retrieve examples to guide the LLM
    examples = await fetch_similar_qa_pairs(question)

    # 5. Generate the SPARQL query
    response = await get_sparql_query(question, examples, linked_entities)

    # 6. Log the process for analysis
    results = None
    if response.get('results') is not None:
        if isinstance(response.get('results'), bool):
            results = str(response.get('results'))

        else:
            results = "\n".join(str(r.get("value", "")) for r in response.get('results', []))

    execution_time = time.monotonic() - start_time

    log_data = {
        "original_question": original_question,
        "rephrased_question": question,
        "ner": str(ner_response),
        "candidates": str(linked_entities),
        "examples": str(examples),
        "generated_query": str(response.get("sparql")),
        "result": results,
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
    llm = generic_llm()
    response = await llm.ainvoke(prompt)
    return "yes" in response.content.lower()
