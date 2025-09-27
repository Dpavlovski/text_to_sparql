import time

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from pydantic import Field

from src.agent.prompts import validation_prompt
from src.databases.qdrant.search_embeddings import fetch_similar_entities, search_embeddings
from src.llm.generic_llm import generic_llm
from src.tools.ner import extract_entities
from src.tools.sparql import get_sparql_query
from src.utils.format_examples import format_examples


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
    sim_entities = await fetch_similar_entities(ner_response.keywords, ner_response.lang)

    # 3. Retrieve examples to guide the LLM
    examples = await search_embeddings(question, collection_name="lcquad2_0")
    formatted_examples = format_examples(examples)

    # 4. Generate the SPARQL query
    response = await get_sparql_query(question, formatted_examples, sim_entities)

    # 5. Log the process for analysis
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
        "ner_response": str(ner_response),
        "similar_entities": str(sim_entities),
        "examples": str(examples),
        "sparql_query": str(response.get("sparql")),
        "results": results,
        "time_of_execution": f"{execution_time:.2f}"
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
