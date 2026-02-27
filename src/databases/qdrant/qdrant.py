import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.conversions import common_types as types
from qdrant_client.http.models import QueryResponse, Record, ScoredPoint

load_dotenv()


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

    async def search_embeddings(self,
                                vector: List[float],
                                collection_name: str,
                                score_threshold: float,
                                top_k: int,
                                filter: Optional[Dict[str, Any]] = None) -> List[ScoredPoint]:

        field_condition = QdrantDatabase._generate_filter(filter=filter)
        query_response = await self.client.query_points(
            query=vector,
            score_threshold=score_threshold,
            collection_name=collection_name,
            limit=top_k,
            query_filter=field_condition
        )
        return query_response.points

    async def search_embeddings_batch(
            self,
            vectors: List[Any],
            collection_name: str,
            score_threshold: float,
            top_k: int,
            filter: Optional[Dict[str, Any]] = None
    ) -> List[QueryResponse]:
        field_condition = QdrantDatabase._generate_filter(filter=filter)

        search_requests = []
        for vector in vectors:
            search_requests.append(
                models.QueryRequest(
                    query=vector,
                    limit=top_k,
                    filter=field_condition,
                    score_threshold=score_threshold,
                    with_payload=True,
                )
            )

        return await self.client.query_batch_points(
            collection_name=collection_name,
            requests=search_requests
        )

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
