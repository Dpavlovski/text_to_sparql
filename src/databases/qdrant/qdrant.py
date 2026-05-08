import uuid
from typing import List, Dict, Any, Optional, Union

from loguru import logger
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.conversions import common_types as types

from src.utils.map_candidates import QueryResponse, ScoredPoint


class QdrantDatabase:
    """
    A professional, asynchronous service class for interacting with Qdrant.
    Designed to be instantiated once at application startup and injected into dependencies.
    """

    def __init__(self, host: str, port: int):
        self.client = AsyncQdrantClient(url=host, port=port)
        logger.info(f"Initialized Qdrant client at {host}:{port}")

    async def close(self):
        """Closes the underlying HTTP connection to Qdrant."""
        await self.client.close()
        logger.info("Closed Qdrant client connection.")

    @staticmethod
    def _ensure_valid_id(point_id: Union[str, int]) -> Union[str, int]:
        """
        Qdrant strictly requires IDs to be integers or valid UUIDs.
        If a normal string is passed (e.g., "Q42"), we deterministically
        convert it into a UUID so Qdrant accepts it without error.
        """
        if isinstance(point_id, int):
            return point_id

        try:
            # Check if it's already a valid UUID
            uuid.UUID(str(point_id))
            return point_id
        except ValueError:
            # It's an arbitrary string. Hash it into a deterministic UUID.
            # Using uuid5 means "Q42" will ALWAYS result in the exact same UUID.
            generated_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, str(point_id)))
            return generated_uuid

    async def collection_exists(self, collection_name: str) -> bool:
        """Checks if a collection exists."""
        return await self.client.collection_exists(collection_name)

    async def create_collection(
            self,
            collection_name: str,
            vector_size: int = 384,
            distance: models.Distance = models.Distance.COSINE
    ):
        """Creates a collection safely, verifying configuration if it already exists."""
        try:
            if not await self.collection_exists(collection_name):
                logger.info(f"Creating new collection: '{collection_name}' (size: {vector_size})")
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=distance
                    ),
                )
            else:
                collection_info = await self.client.get_collection(collection_name)
                # Verify existing collection has matching config
                if (collection_info.config.params.vectors.size != vector_size or
                        collection_info.config.params.vectors.distance != distance):
                    raise ValueError(
                        f"Existing collection '{collection_name}' has incompatible configuration. "
                        f"Expected size={vector_size}, found size={collection_info.config.params.vectors.size}."
                    )
                logger.debug(f"Collection '{collection_name}' already exists and config is valid.")
        except Exception as e:
            if "already exists" in str(e):
                return
            logger.exception(f"Failed to create/verify collection '{collection_name}'")
            raise RuntimeError(f"Failed to create/verify collection '{collection_name}': {str(e)}")

    async def retrieve_point(
            self,
            collection_name: str,
            point_id: Union[str, int]
    ) -> types.Record:
        """Retrieves a single point by its ID (converting to UUID if necessary)."""
        valid_id = self._ensure_valid_id(point_id)

        points = await self.client.retrieve(
            collection_name=collection_name,
            ids=[valid_id],
            with_vectors=True,
        )
        if not points:
            raise ValueError(f"Point with ID '{point_id}' (UUID: {valid_id}) not found.")
        return points[0]

    async def search_embeddings(self,
                                vector: List[float],
                                collection_name: str,
                                score_threshold: float,
                                top_k: int,
                                filter: Optional[Dict[str, Any]] = None) -> List[ScoredPoint]:
        try:

            field_condition = QdrantDatabase._generate_filter(filter=filter)
            query_response = await self.client.query_points(
                query=vector,
                score_threshold=score_threshold,
                collection_name=collection_name,
                limit=top_k,
                query_filter=field_condition
            )
            return query_response.points
        except Exception as e:
            logger.exception(f"Single search failed in collection '{collection_name}'")
            raise

    async def search_embeddings_batch(
            self,
            vectors: List[Any],
            collection_name: str,
            score_threshold: float,
            top_k: int,
            filter: Optional[Dict[str, Any]] = None
    ) -> List[QueryResponse]:
        try:
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
        except Exception as e:
            logger.exception(f"Batch search failed in collection '{collection_name}'")
            raise

    async def upsert_record(
            self,
            unique_id: Union[str, int],
            collection_name: str,
            payload: Dict[str, Any],
            vector: List[float]
    ) -> models.Record:
        """
        Upserts a single record into Qdrant. Creates the collection if it doesn't exist.
        """
        valid_id = self._ensure_valid_id(unique_id)

        if not await self.collection_exists(collection_name):
            await self.create_collection(collection_name)

        try:
            await self.client.upsert(
                collection_name=collection_name,
                points=[models.PointStruct(
                    id=valid_id,
                    payload=payload,
                    vector=vector,
                )],
                wait=True  # Wait to ensure it's queryable immediately
            )
            logger.debug(f"Upserted record '{unique_id}' into '{collection_name}'")
            return await self.retrieve_point(collection_name, valid_id)

        except Exception as e:
            # Auto-recover if the collection was dropped mid-execution
            if "Not found: Collection" in str(e):
                logger.warning(f"Collection '{collection_name}' disappeared. Recreating...")
                await self.create_collection(collection_name)
                return await self.upsert_record(unique_id, collection_name, payload, vector)

            logger.exception(f"Failed to upsert record '{unique_id}'")
            raise RuntimeError(f"Failed to upsert record: {str(e)}")

    @staticmethod
    def _generate_filter(filter: Optional[Dict[str, Any]] = None) -> Optional[models.Filter]:
        """
        Converts a simple Python dictionary into Qdrant's Filter object format.
        Example: {"lang": "en"} -> models.Filter(...)
        """
        if not filter:
            return None

        return models.Filter(must=[
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value),
            )
            for key, value in filter.items()
        ])
