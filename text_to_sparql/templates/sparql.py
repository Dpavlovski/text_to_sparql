from text_to_sparql.utils.format_examples import format_examples


def sparql_template(question, examples, entity_descriptions, relations_descriptions):
    return f"""
        You are an AI that generates precise SPARQL queries to answer the given question. 
        Your task is to carefully select the most relevant entities and relations from the provided options in order to answer the question. Use only the provided entities and relations.
        You must only return the sparql query and nothing else.
        
        Examples:
        {format_examples(examples)}
        
        
        Question: {question}

        Entities:
        {entity_descriptions}

        Relations:
        {relations_descriptions}

        Generate the SPARQL query:
    """
