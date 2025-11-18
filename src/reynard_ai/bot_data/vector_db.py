import numpy
import hashlib

from enum import Enum
from typing import Any
from dataclasses import dataclass
from ..ai_apis.providers import ProviderData
from ..ai_apis.client import EmbeddingsClient
from pymilvus import MilvusClient, AsyncMilvusClient, DataType

class VectorDatabaseConnection:
    def __init__(self, _async_client: AsyncMilvusClient, _sync_client: MilvusClient, vectorizer: EmbeddingsClient):
        self._sync_client = _sync_client
        self._async_client = _async_client
        self.vectorizer = vectorizer
        
    @dataclass
    class DBEntry:
        id: numpy.int64
        metadata: dict
        text: str

    @dataclass
    class Hit:
        id: int
        distance: float
        entity: dict[str, Any]

    class Indexes(Enum):
        KNOWLEDGE = "knowledge"
        MEMORIES = "memories"

    async def index(self, index: Indexes, data: DBEntry | list[DBEntry]):
        if isinstance(data, list):
            texts = [entry.text for entry in data]
            vectors = await self.vectorizer.vectorize(texts)
            to_index = [
                {
                    "id": entry.id, 
                    "metadata": entry.metadata, 
                    "vector": vectors[i], 
                    "text": entry.text
                }
                for i, entry in enumerate(data)
            ]
            await self._async_client.insert(index.value, to_index)
        else:
            to_index = {
                "id": data.id, 
                "metadata": data.metadata, 
                "vector": await self.vectorizer.vectorize(data.text), 
                "text": data.text
            }
            await self._async_client.insert(index.value, to_index)

    async def search(self, index: Indexes, text: str, limit=5) -> list[list[dict]]:
        return await self._async_client.search(
            collection_name=index.value,
            output_fields=["id", "metadata", "text"],
            data=[await self.vectorizer.vectorize(text)],
            limit=limit
        )

class VectorDatabase:
    @dataclass
    class Entry:
        data: str
        metadata: dict[str, Any]
        entry_id: int | None = None

        def __post_init__(self):
            if self.entry_id is None:
                combined = self.data + str(self.metadata)
                self.entry_id = int(hashlib.sha256(combined.encode()).hexdigest(), 16) & 0x7FFFFFFF
        
    def __init__(self, vectorizer: EmbeddingsClient, path: str):
        self.vectorizer = vectorizer
        self.async_client = AsyncMilvusClient(path)
        self.sync_client = MilvusClient(path)

    async def connect(self) -> VectorDatabaseConnection:
        async def make_schema(name: str):
            schema = self.sync_client.create_schema(
                auto_id=False,
                description="Brain schema",
            )
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=3072)
            schema.add_field("metadata", DataType.JSON)
            schema.add_field("text", DataType.VARCHAR, max_length=8192)
            return schema

        async def create_collection_index(name: str):
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                metric_type="COSINE",
                index_type="IVF_FLAT",
                index_name="vector_index"
            )
            await self.async_client.create_index(
                collection_name=name,
                index_params=index_params
            )

        if not self.sync_client.has_collection("knowledge"):
            knowledge_schema = await make_schema("knowledge")
            await self.async_client.create_collection(collection_name="knowledge", schema=knowledge_schema)
            await create_collection_index("knowledge")
        if not self.sync_client.has_collection("memories"):
            memories_schema = await make_schema("memories")
            await self.async_client.create_collection(collection_name="memories", schema=memories_schema)
            await create_collection_index("memories")
        
        return VectorDatabaseConnection(self.async_client, self.sync_client, self.vectorizer)