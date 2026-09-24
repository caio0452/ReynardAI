import re
import os
import json
import glob
import numpy
import asyncio
import hashlib
import logging

from typing import Literal, Dict, Any

from pydantic import BaseModel, Field
from ..ai_apis.client import EmbeddingsClient
from ..chat_base.message_snapshot import MessageSnapshot
from ..bot_data.vector_db import MAX_VARCHAR_BYTES, VectorDatabase, VectorDatabaseConnection

class RetrievalConfig(BaseModel):
    max_chunks: int = 5
    max_characters: int = -1

class ChunkingConfig(BaseModel):
    strategy: Literal['fixed_size', 'separator'] = 'fixed_size'
    chunk_size: int = 2000
    overlap: int = 400
    separator: str = "\n\n"

class KnowledgeStrategyConfig(BaseModel):
    default: ChunkingConfig = Field(default_factory=ChunkingConfig)
    files: Dict[str, ChunkingConfig] = Field(default_factory=dict)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    enable_mrl: bool = False
    mrl_dim: int | None = None

class LongTermMemoryIndex:
    def __init__(self, _db_conn: VectorDatabaseConnection): 
        self._db_conn = _db_conn

    async def close(self):
        await self._db_conn.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @staticmethod
    async def from_vectorizer(vectorizer: EmbeddingsClient, memory_settings: Any | None = None) -> "LongTermMemoryIndex":
        if memory_settings and getattr(memory_settings, "enable_mrl", False) and getattr(memory_settings, "mrl_dim", None) is not None:
            vectorizer.mrl_dim = memory_settings.mrl_dim
            vectorizer.embedding_dim = memory_settings.mrl_dim
        memories_db_path = os.path.join(os.getcwd(), 'brain_content', 'memories', 'memories.db')
        vector_db: VectorDatabase = VectorDatabase(vectorizer, memories_db_path)
        db_conn = await vector_db.connect()
        return LongTermMemoryIndex(db_conn)

    async def memorize(self, message: MessageSnapshot):
        await self._db_conn.index(
            VectorDatabaseConnection.Indexes.MEMORIES,
            VectorDatabaseConnection.DBEntry(
                numpy.int64(message.message_id),
                 {"type": "memory"},
                message.text, 
            )
        )

    async def mass_memorize(self, messages: list[MessageSnapshot]):
        entries = []
        for message in messages:
            entries.append(VectorDatabaseConnection.DBEntry(
                numpy.int64(message.message_id),
                 {"type": "memory"},
                message.text, 
            ))
        await self._db_conn.index(
            VectorDatabaseConnection.Indexes.MEMORIES,
            entries
        )

    async def get_closest_messages(self, query: str, *, n=5) -> list[VectorDatabaseConnection.Hit]:
        hits_for_query_list = await self._db_conn.search(
            VectorDatabaseConnection.Indexes.MEMORIES, query, n)
        hits_for_query = hits_for_query_list[0]

        ret: list[VectorDatabaseConnection.Hit] = []
        for hit in hits_for_query:
            ret.append(VectorDatabaseConnection.Hit(
                id=hit["id"],
                distance=hit["distance"],
                entity=hit["entity"]
            ))
        return ret

class KnowledgeIndex:
    def __init__(self, _db_conn: VectorDatabaseConnection, config: KnowledgeStrategyConfig): 
        self._db_conn = _db_conn
        self.config = config

    async def close(self):
        await self._db_conn.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @staticmethod
    async def from_vectorizer(vectorizer: EmbeddingsClient, strategy_config: KnowledgeStrategyConfig | None = None) -> "KnowledgeIndex":
        base_path = os.path.join(os.getcwd(), 'brain_content', 'knowledge')
        knowledge_db_path = os.path.join(base_path, 'knowledge.db')
        config_path = os.path.join(base_path, 'chunking_strategy.json')
        
        if strategy_config is None:
            strategy_config = KnowledgeStrategyConfig()
            if os.path.exists(config_path):
                try:
                    def _read_config():
                        with open(config_path, 'r') as f:
                            return json.load(f)
                    json_data = await asyncio.to_thread(_read_config)
                    strategy_config = KnowledgeStrategyConfig(**json_data)
                    logging.info(f"Loaded knowledge strategy from {config_path}")
                except Exception as e:
                    logging.error(f"Failed to load chunking_strategy.json: {e}. Using defaults.")

        if strategy_config.enable_mrl and strategy_config.mrl_dim is not None:
            vectorizer.mrl_dim = strategy_config.mrl_dim
            vectorizer.embedding_dim = strategy_config.mrl_dim

        vector_db: VectorDatabase = VectorDatabase(vectorizer, knowledge_db_path)
        db_conn = await vector_db.connect()
        return KnowledgeIndex(db_conn, strategy_config)
    
    @staticmethod
    def chunk_text(text: str, config: ChunkingConfig) -> list[str]:
        chunks = []

        if config.strategy == 'separator':
            raw_chunks = text.split(config.separator)
            chunks = [c.strip() for c in raw_chunks if c.strip()]
            return chunks
        elif config.strategy == 'fixed_size':
            chunk_size = config.chunk_size
            overlap = config.overlap
            start = 0
            
            while start < len(text):
                end = min(start + chunk_size, len(text))
                if start != 0:
                    start -= overlap
                if end < len(text):
                    # Don't cut off words
                    boundary_end = re.search(r'\b', text[end:])
                    if boundary_end is not None:
                        end = boundary_end.start() + end
                chunks.append(text[start:end])
                start += chunk_size
            return chunks
        else:
            raise RuntimeError(f"Unknown chunking strategy: {config.strategy}")

    @staticmethod
    def split_for_storage(text: str, max_bytes: int = MAX_VARCHAR_BYTES) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_bytes = 0

        for character in text:
            character_bytes = len(character.encode("utf-8"))
            if current and current_bytes + character_bytes > max_bytes:
                chunks.append("".join(current))
                current = []
                current_bytes = 0
            current.append(character)
            current_bytes += character_bytes

        if current:
            chunks.append("".join(current))
        return chunks

    async def chunk_and_index(self, text: str, config: ChunkingConfig, *, metadata={"type": "knowledge"}) -> int:
        chunks = KnowledgeIndex.chunk_text(text, config)
        chunks = [
            stored_chunk
            for chunk in chunks
            for stored_chunk in KnowledgeIndex.split_for_storage(chunk)
        ]
        if not chunks: 
            return 0
            
        entries = []
        for chunk in chunks:
            if not chunk.strip():
                continue

            hash_obj = hashlib.sha256(chunk.encode('utf-8'))
            hash_int = numpy.int64(int.from_bytes(hash_obj.digest()[:8], byteorder='big', signed=True))
            entries.append(
                VectorDatabaseConnection.DBEntry(
                    hash_int,
                    metadata,
                    chunk,
                )
            )
        
        INDEX_BATCH_SIZE = 16
        total_indexed = 0
        for i in range(0, len(entries), INDEX_BATCH_SIZE):
            batch = entries[i : i + INDEX_BATCH_SIZE]
            try:
                await self._db_conn.index(
                    VectorDatabaseConnection.Indexes.KNOWLEDGE,
                    batch
                )
                total_indexed += len(batch)
            except Exception as e:
                logging.error(f"Failed to index batch starting at {i}: {e}")
                raise
                
        return total_indexed

    async def index_files(self, files: list[str], max_concurrent_tasks: int = 4):
        if max_concurrent_tasks < 1:
            raise ValueError("max_concurrent_tasks must be at least 1")

        strategy_config = self.config
        txt_files = [file for file in files if file.endswith('.txt')]
        non_txt_files = [file for file in files if not file.endswith('.txt') and not file.endswith('.json')]
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        for file in non_txt_files:
            logging.info(f"Error: {file} is not a .txt file. All knowledge must be in text files. Skipping.")

        async def process_file(file_path):
            async with semaphore:
                file_name = os.path.basename(file_path)
                file_config = strategy_config.files.get(file_name, strategy_config.default)
                def _read_file():
                    with open(file_path, 'r') as file:
                        return file.read()
                text = await asyncio.to_thread(_read_file)
                logging.info(f"Chunking file '{file_name}' using strategy: {file_config.strategy}")
                n_chunks = await self.chunk_and_index(text, file_config)
                return n_chunks

        tasks = [process_file(file) for file in txt_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_chunks = 0
        failed_files = 0
        for file_path, result in zip(txt_files, results):
            if isinstance(result, Exception):
                failed_files += 1
                logging.error(f"Error indexing {file_path}", exc_info=result)
            elif isinstance(result, int):
                total_chunks += result
                logging.info(f"Indexed {file_path}: {result} chunks")

        logging.info(f"Total chunks indexed: {total_chunks}")
        if failed_files:
            logging.error(f"Failed to index {failed_files} of {len(txt_files)} knowledge files")
    
    async def index_from_folder(self, path, max_concurrent_tasks=4):
        if not os.path.exists(path):
            logging.info(f"The knowledge folder, located in '{path}' does not exist. Skipping knowledge indexing.")
            return
        
        folder_files = glob.glob(f"{path}/*")
        if len(folder_files) == 0:
            logging.info(f"No files in knowledge folder '{path}', skipping indexing")
            return
        
        await self.index_files(folder_files, max_concurrent_tasks=max_concurrent_tasks)

    async def retrieve(self, related_text: str):
        max_chunks = self.config.retrieval.max_chunks
        max_chars = self.config.retrieval.max_characters

        search_results = await self._db_conn.search(
            VectorDatabaseConnection.Indexes.KNOWLEDGE, 
            related_text, 
            max_chunks
        )
        
        hits = search_results[0]
        character_limit_exists = max_chars > 0
        if not character_limit_exists:
            return [hits]
        
        filtered_hits = []
        current_chars = 0    
        for hit in hits:
            chunk_text = hit.get("entity", "")
            chunk_len = len(chunk_text)
            if current_chars + chunk_len > max_chars:
                break
            filtered_hits.append(hit)
            current_chars += chunk_len
        return [filtered_hits]
