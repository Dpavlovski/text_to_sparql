from typing import List, Literal

from pydantic import BaseModel, Field

from src.agent.prompts import ner_prompt
from src.llm.generic_llm import generic_llm


class Keyword(BaseModel):
    value: str = Field(description="The extracted keyword or phrase.")
    type: Literal["item", "property"] = Field(description="The type of the keyword.")


class NERResponse(BaseModel):
    keywords: List[Keyword] = Field(description="A list of extracted keywords and phrases.")
    lang: str = Field(description="The language code of the question (e.g., 'en', 'es').")


async def extract_entities(question: str) -> NERResponse:
    formatted_prompt = ner_prompt.format(question=question)
    structured_llm = generic_llm().with_structured_output(NERResponse)
    response = await structured_llm.ainvoke(formatted_prompt)
    return response
