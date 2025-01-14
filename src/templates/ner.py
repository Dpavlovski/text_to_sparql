def ner_template(question):
    return f"""Given the user question below your job is to search wikidata_api and return all relevant items and properties which are connected to the context.
Question: 
{question}
            
Important: The search must be relevant to the question. Make sure the items and properties exists in Wikidata. Dont extract anything outside the question.

Return the extracted information in valid JSON format. Do not include any other text or comments.
Output:
{{  
    "nodes": [
       {{
         "wikidata_id": "",
         "wikidata_label": "",
         "wikidata_type": ""  // Use "item" or "property"
       }}
    ]
}}
"""
