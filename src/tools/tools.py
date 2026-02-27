import time
from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from pydantic import Field, BaseModel

from src.agent.prompts import validation_prompt
from src.llm.llm_provider import llm_provider
from src.tools.ner import Keyword
from src.tools.sparql import get_sparql_query


@tool("generate_sparql_query")
async def generate_sparql(
        question: str = Field(description="The possibly rephrased natural language question"),
        original_question: str = Field(description="The original question", default=""),
        candidates: str = Field(description="Pre-retrieved candidates", default=""),
        examples: str = Field(description="Pre-retrieved few-shot examples", default=""),
        ner_keywords: List[Keyword] = Field(description="Pre-retrieved NER keywords", default_factory=list),
) -> dict:
    """
    Generates and executes a SPARQL query using the provided context.
    """
    start_time = time.monotonic()

    response = await get_sparql_query(
        question=question,
        examples=examples,
        candidates=candidates,
    )

    results_for_log = None
    raw_results = response.get('results')

    if raw_results is not None:

        # ASK questions
        if isinstance(raw_results, bool):
            results_for_log = raw_results

        # SELECT questions with results
        elif isinstance(raw_results, list) and raw_results:
            extracted_values = []

            for row in raw_results:
                if not isinstance(row, dict):
                    continue

                for var, val_dict in row.items():
                    if isinstance(val_dict, dict):
                        val = val_dict.get("value")
                        if val:
                            extracted_values.append(val)

            # remove duplicates, preserve order
            seen = set()
            extracted_values = [
                x for x in extracted_values
                if not (x in seen or seen.add(x))
            ]

            # 🔹 FINAL TOOL FORMATTING
            results_for_log = " ".join(str(v) for v in extracted_values)

        # SELECT questions with NO results
        elif isinstance(raw_results, list) and not raw_results:
            results_for_log = None

        else:
            results_for_log = None

    execution_time = time.monotonic() - start_time

    ner_log_str = ""
    if ner_keywords:
        items = []
        for k in ner_keywords:
            if isinstance(k, dict):
                items.append(f"{k.get('value')} {k.get('context')} ({k.get('type')})")
            elif hasattr(k, 'value') and hasattr(k, 'type'):
                items.append(f"{k.value} {k.context} ({k.type})")
        ner_log_str = ", ".join(items)

    log_data = {
        "original_question": original_question,
        "rephrased_question": question,
        "ner": ner_log_str,
        "candidates": str(candidates),
        "examples": str(examples),
        "generated_query": str(response.get("sparql")),
        "result": results_for_log,
        "time": f"{execution_time:.2f}"
    }

    response['log_data'] = log_data

    return response


class ValidationResult(BaseModel):
    is_valid: bool = Field(
        description="Set to True if the SPARQL results correctly answer the user's question, otherwise False.")


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
        ValidationResult
    )
    response = await llm.ainvoke(prompt)
    return response.is_valid
