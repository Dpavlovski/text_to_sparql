from typing import Dict, List

from pydantic import BaseModel, Field

from src.llm.chat_with_ollama import chat_with_ollama

ner_prompt = """Your task is to extract all relevant keywords and phrases from the given question that could help in identifying Wikidata entities. This includes both proper names (e.g., organizations, persons, locations) and important common nouns and verbs that are essential to understanding the question. Also, identify the language of the question.

Question: 
{question}

Format:
- Format your response as a JSON object with the following keys:
    - "lang": A string representing the language code of the question (e.g., "en", "es", "fr").
    - "keywords": A list of JSON objects. Each object must have these two keys:
        - "value": The extracted keyword or phrase as a string.
        - "type": The type of the keyword. Must be one of the following strings:
            - "item": For distinct entities like people, places, organizations, or concepts (e.g., "Leonardo da Vinci", "Paris", "Google", "Mona Lisa").
            - "property": For attributes, relationships, or actions related to an item (e.g., "date of birth", "capital of", "invented", "painted").
- If no relevant keywords are found, the "keywords" list should be an empty list [].
"""


class NERResponse(BaseModel):
    keywords: List[Dict[str, str]] = Field(description="Extracted keywords and phrases from the given question.")
    lang: str = Field(description="The language code of the question.")


def extract_entities(question: str) -> NERResponse:
    formatted_prompt = ner_prompt.format(question=question)
    structured_llm = chat_with_ollama().with_structured_output(NERResponse)
    response: NERResponse = structured_llm.invoke(formatted_prompt)
    return NERResponse(keywords=response.keywords, lang=response.lang)
