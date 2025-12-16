from typing import Any, Dict

from pydantic import BaseModel, Field
from rdflib.plugins.sparql import prepareQuery

from src.agent.prompts import sparql_prompt_template
from src.llm.llm_provider import llm_provider
from src.wikidata.api import execute_sparql_query
from src.wikidata.prefixes import ensure_prefixes


# 2. Define the output structure
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
    sparql_prompt = sparql_prompt_template.format(
        examples=examples,
        question=question,
        candidates=candidates,
    )

    llm = llm_provider.get_model("gpt-4.1-mini")
    structured_llm = llm.with_structured_output(SparqlGenerationResponse)

    try:
        generation: SparqlGenerationResponse = await structured_llm.ainvoke(sparql_prompt)
    except Exception as e:
        return {"sparql": "", "results": [], "error": f"LLM Generation Error: {e}"}

        # Inject prefixes before sending to execution
    final_query = ensure_prefixes(generation.sparql)

    # Validation
    is_valid, validation_msg = validate_sparql(final_query)
    results = None

    if is_valid:
        try:
            results = await execute_sparql_query(final_query)
        except Exception as e:
            return {"sparql": generation.sparql, "results": None, "error": str(e)}

    return {
        "sparql": generation.sparql,
        "results": results,
        "reasoning": generation.reasoning,
        "is_valid": is_valid
    }
