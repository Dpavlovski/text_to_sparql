import os
import uuid
from typing import List, Dict, Any, Optional, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.conversions import common_types as types
from qdrant_client.http.models import Record

load_dotenv()


class SearchOutput(BaseModel):
    score: float
    value_type: str


T = TypeVar('T')


class QdrantDatabase:
    client: AsyncQdrantClient

    def __init__(self):
        load_dotenv()
        self.client = AsyncQdrantClient(url=os.getenv("QDRANT_HOST"), port=os.getenv("QDRANT_PORT", None))

    async def collection_exists(self, collection_name: str) -> bool:
        return await self.client.collection_exists(collection_name)

    async def create_collection(
            self,
            collection_name: str,
            vector_size: int = 384,
            distance: models.Distance = models.Distance.COSINE
    ):
        try:
            if not await self.collection_exists(collection_name):
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=distance
                    ),
                )
            else:
                collection_info = await self.client.get_collection(collection_name)
                if (collection_info.config.params.vectors.size != vector_size or
                        collection_info.config.params.vectors.distance != distance):
                    raise ValueError(
                        f"Existing collection {collection_name} has incompatible configuration. "
                        f"Expected size={vector_size}, distance={distance}, "
                        f"found size={collection_info.config.params.vectors.size}, "
                        f"distance={collection_info.config.params.vectors.distance}"
                    )
        except Exception as e:
            if "already exists" in str(e):
                return
            raise RuntimeError(f"Failed to create/verify collection {collection_name}: {str(e)}")

    async def embedd_and_upsert_record(
            self,
            value: str,
            collection_name: str,
            unique_id: str = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:

        metadata = {} if metadata is None else metadata
        metadata["value"] = value

        from src.llm.embed_labels import embed_value

        vector = embed_value(value)
        record_id = str(uuid.uuid4()) if unique_id is None else unique_id

        await self.upsert_record(record_id, collection_name, metadata, vector)

    async def delete_all_collections(self):
        collections = await self.client.get_collections()
        for collection in collections.collections:
            await self.client.delete_collection(collection_name=collection.name)

    async def delete_collection(self, collection_name: str):
        await self.client.delete_collection(collection_name=collection_name)

    async def retrieve_point(
            self,
            collection_name: str,
            point_id: str
    ) -> types.Record:
        points = await self.client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_vectors=True,
        )
        return points[0]

    async def search_embeddings(
            self,
            query_vector: List[float],
            collection_name: str,
            score_threshold: float,
            top_k: int,
            filter: Optional[Dict[str, Any]] = None
    ) -> List[types.ScoredPoint]:
        field_condition = QdrantDatabase._generate_filter(filter=filter)

        return await self.client.search(
            query_vector=query_vector,
            score_threshold=score_threshold,
            collection_name=collection_name,
            limit=top_k,
            query_filter=field_condition
        )

    async def search_embeddings_str(
            self,
            query: str,
            collection_name: str,
            score_threshold: float,
            top_k: int,
            filter: Optional[Dict[str, Any]] = None
    ) -> List[types.ScoredPoint]:

        if collection_name == "lcquad2_0":
            from src.llm.embed_examples import embed_examples
            query_vector = embed_examples(query)
        else:
            from src.llm.embed_labels import embed_value
            query_vector = embed_value(query)

        return await self.search_embeddings(collection_name=collection_name, score_threshold=score_threshold,
                                            top_k=top_k,
                                            query_vector=query_vector, filter=filter)

    async def get_all_points(
            self,
            collection_name: str,
            with_vectors: bool = False,
            filter: Optional[Dict[str, Any]] = None
    ) -> List[types.Record]:
        field_condition = QdrantDatabase._generate_filter(filter=filter)
        offset = None
        records = []
        while True:
            response, next_page_offset = await self.client.scroll(
                collection_name=collection_name,
                scroll_filter=field_condition,
                limit=50,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors
            )
            records.extend(response)
            offset = next_page_offset
            if offset is None:
                break
        return records

    async def upsert_record(
            self,
            unique_id: str,
            collection_name: str,
            payload: Dict[str, Any],
            vector: List[float]
    ) -> Record:
        if not await self.collection_exists(collection_name):
            await self.create_collection(collection_name)

        try:
            await self.client.upsert(
                collection_name=collection_name,
                points=[models.PointStruct(
                    id=unique_id,
                    payload=payload,
                    vector=vector,
                )],
                wait=True
            )
            return await self.retrieve_point(collection_name, unique_id)
        except Exception as e:
            if "Not found: Collection" in str(e):
                await self.create_collection(collection_name)
                return await self.upsert_record(unique_id, collection_name, payload, vector)
            raise RuntimeError(f"Failed to upsert record: {str(e)}")

    async def delete_points(
            self,
            collection_name: str,
            filter: Optional[Dict[str, Any]] = None
    ):
        field_condition = QdrantDatabase._generate_filter(filter=filter)
        await self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=field_condition
            ),
        )

    async def update_point(
            self,
            collection_name: str,
            id: str,
            update: Dict[str, Any]
    ):
        await self.client.set_payload(
            collection_name=collection_name,
            wait=True,
            payload=update,
            points=[id]
        )

    @staticmethod
    def _generate_filter(filter: Optional[Dict[str, Any]] = None):
        field_condition = None
        if filter:
            field_condition = models.Filter(must=[
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
                for key, value in filter.items()])

        return field_condition


qdrant_db = QdrantDatabase()
