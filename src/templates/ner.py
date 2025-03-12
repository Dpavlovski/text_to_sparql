def ner_template(question):
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
