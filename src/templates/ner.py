from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from src.llm.generic_chat import generic_chat


def ner_template(question: str) -> str:
    return f"""Your task is to extract all relevant keywords and phrases from the given question that could help in identifying Wikidata labels. This includes both proper names (e.g., organizations, persons, locations) and important common nouns and verbs that are essential to understanding the question. Also, identify the language of the question.

Question: {question}

Output Format:
Return the extracted information in **valid JSON** format with the following structure:
{{
    "labels": [
        <label1>,
        <label2>,
        ...
    ],
    "lang": "<language_code>"
}}
"""


def extract_entities(question: str) -> dict:
    base_prompt = ner_template(question)

    labels_field = ResponseSchema(
        name="labels",
        description="A list of all extracted entities, keywords and key phrases."
    )

    lang_field = ResponseSchema(
        name="lang",
        description="The language code of the question, e.g. 'en', 'es', 'fr', etc."
    )

    output_parser = StructuredOutputParser.from_response_schemas([labels_field, lang_field])

    format_instructions = output_parser.get_format_instructions()
    full_prompt = "\n\n".join([base_prompt, format_instructions])

    raw_output = generic_chat(message=full_prompt)

    parsed = output_parser.parse(raw_output)

    return parsed
