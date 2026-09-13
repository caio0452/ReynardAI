import os
import numpy
import hashlib

from enum import Enum
from typing import Any
from dataclasses import dataclass
from ..ai_apis.providers import ProviderData
from ..ai_apis.client import EmbeddingsClient
from pymilvus import AsyncMilvusClient, DataType

class VectorDatabaseConnection:
    def __init__(self, client: AsyncMilvusClient, vectorizer: EmbeddingsClient):
        self._async_client = client
        self.client = client
        self.vectorizer = vectorizer

    async def close(self):
        await self._async_client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
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
        parent_dir = os.path.dirname(os.path.abspath(path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self.async_client = AsyncMilvusClient(path)

    async def close(self):
        await self.async_client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect(self) -> VectorDatabaseConnection:
        def make_schema():
            schema = AsyncMilvusClient.create_schema(
                auto_id=False,
                description="Brain schema",
            )
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.vectorizer.embedding_dim)
            schema.add_field("metadata", DataType.JSON)
            schema.add_field("text", DataType.VARCHAR, max_length=8192)
            return schema

        def create_collection_index_params():
            index_params = AsyncMilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                metric_type="COSINE",
                index_type="IVF_FLAT",
                index_name="vector_index"
            )
            return index_params

        for collection_name in ("knowledge", "memories"):
            if not await self.async_client.has_collection(collection_name):
                schema = make_schema()
                index_params = create_collection_index_params()
                await self.async_client.create_collection(
                    collection_name=collection_name,
                    schema=schema,
                    index_params=index_params,
                )
            else:
                await self.async_client.load_collection(collection_name)
        
        return VectorDatabaseConnection(self.async_client, self.vectorizer)

