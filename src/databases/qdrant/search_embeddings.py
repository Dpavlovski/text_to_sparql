from src.databases.qdrant.qdrant import QdrantDatabase


def extract_search_objects(question: str, collection_name: str):
    similar_questions = []
    database = QdrantDatabase()
    similar_questions.extend(database.search_embeddings_str(query=question, score_threshold=0.2, top_k=5,
                                                            collection_name=collection_name))
    return similar_questions
