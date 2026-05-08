from typing import Any, Dict

from loguru import logger
from pydantic import BaseModel, Field
from rdflib.plugins.sparql import prepareQuery

from src.agent.prompts import sparql_prompt_template, zero_shot_sparql_prompt_template
from src.config.settings import settings
from src.llm.llm_provider import LLMProvider
from src.wikidata.api import execute_sparql_query
from src.wikidata.prefixes import ensure_prefixes


class SparqlGenerationResponse(BaseModel):
    reasoning: str = Field(description="Brief explanation of the logic and entities chosen.")
    sparql: str = Field(description="The valid SPARQL query without markdown formatting.")


def validate_sparql(query: str):
    """Validates syntax using rdflib."""
    try:
        prepareQuery(query)
        return True, "Query is valid."
    except Exception as e:
        return False, f"Syntax Error: {e}"


async def get_sparql_query(
        question: str,
        examples: str,
        candidates: str,
) -> Dict[str, Any]:
    # --- DYNAMIC PROMPT SELECTION ---
    # If candidates and examples are empty, we are in Zero-Shot mode!
    if not candidates and not examples:
        logger.debug("Using ZERO-SHOT prompt template.")
        sparql_prompt = zero_shot_sparql_prompt_template.format(
            question=question
        )
    else:
        logger.debug("Using RAG prompt template.")
        sparql_prompt = sparql_prompt_template.format(
            examples=examples,
            question=question,
            candidates=candidates,
        )

    llm = LLMProvider.get_model(settings.default_llm_model)
    structured_llm = llm.with_structured_output(SparqlGenerationResponse, method="json_mode", include_raw=True)

    logger.debug(f"Generating SPARQL query for question: '{question}'")

    try:
        response = await structured_llm.ainvoke(sparql_prompt)
        generation: SparqlGenerationResponse = response['parsed']

        if not generation:
            raise ValueError("LLM returned an empty parsed response.")

    except Exception as e:
        logger.error(f"LLM SPARQL Generation Error: {e}")
        return {"sparql": "", "results": None, "reasoning": "", "is_valid": False, "error": str(e)}

    # Inject prefixes before sending to execution
    final_query = ensure_prefixes(generation.sparql)

    # Validation
    is_valid, validation_msg = validate_sparql(final_query)
    results = None
    error_msg = None

    if is_valid:
        logger.debug("SPARQL syntax is valid. Executing query...")
        try:
            results = await execute_sparql_query(final_query)
        except Exception as e:
            logger.error(f"Execution failed for query: {final_query}\nError: {e}")
            error_msg = str(e)
    else:
        logger.warning(f"Generated SPARQL syntax is INVALID: {validation_msg}")
        error_msg = validation_msg

    return {
        "sparql": final_query,  # Return the query WITH prefixes
        "results": results,
        "reasoning": generation.reasoning,
        "is_valid": is_valid,
        "error": error_msg
    }
