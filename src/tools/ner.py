from typing import List, Literal

from pydantic import BaseModel, Field

from src.agent.prompts import ner_prompt
from src.llm.llm_provider import llm_provider


class Keyword(BaseModel):
    value: str = Field(description="The extracted keyword or phrase.")
    type: Literal["item", "property"] = Field(description="The type of the keyword.")
    context: str = Field(
        description="A short 3-5 word description of what this entity likely represents in the context of the question (e.g., 'mathematical concept', 'person', 'city').")


class NERResponse(BaseModel):
    keywords: List[Keyword] = Field(description="A list of extracted keywords.")
    lang: str = Field(description="The language code.")


async def get_ner_result(question: str) -> NERResponse:
    formatted_prompt = ner_prompt.format(question=question)
    structured_llm = llm_provider.get_model("kwaipilot/kat-coder-pro:free").with_structured_output(NERResponse)
    return await structured_llm.ainvoke(formatted_prompt)
