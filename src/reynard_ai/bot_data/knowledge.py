import re
import os
import glob
import numpy
import asyncio
import hashlib
import logging

from ..ai_apis.client import EmbeddingsClient
from ..chat.message_snapshot import MessageSnapshot
from ..bot_data.vector_db import VectorDatabase, VectorDatabaseConnection

class LongTermMemoryIndex:
    def __init__(self, _db_conn: VectorDatabaseConnection): 
        self._db_conn = _db_conn

    @staticmethod
    async def from_vectorizer(vectorizer: EmbeddingsClient) -> "LongTermMemoryIndex":
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
    def __init__(self, _db_conn: VectorDatabaseConnection): 
        self._db_conn = _db_conn

    @staticmethod
    async def from_vectorizer(vectorizer: EmbeddingsClient) -> "KnowledgeIndex":
        knowledge_db_path = os.path.join(os.getcwd(), 'brain_content', 'knowledge', 'knowledge.db')
        vector_db: VectorDatabase = VectorDatabase(vectorizer, knowledge_db_path)
        db_conn = await vector_db.connect()
        return KnowledgeIndex(db_conn)
    
    @staticmethod
    def chunk_text(text, chunk_size=2000, overlap=400):
        chunks = []
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

    async def chunk_and_index(self, text: str, *, metadata={"type": "knowledge"}) -> int:
        chunks = KnowledgeIndex.chunk_text(text)
        if not chunks: 
            return 0
            
        entries = []
        for chunk in chunks:
            hash_obj = hashlib.sha256(text.encode('utf-8'))
            hash_int = numpy.int64(int.from_bytes(hash_obj.digest()[:8], byteorder='big', signed=True))
            entries.append(
                VectorDatabaseConnection.DBEntry(
                    hash_int,
                    metadata,
                    chunk,
                )
            )
        
        await self._db_conn.index(
            VectorDatabaseConnection.Indexes.KNOWLEDGE,
            entries
        )
        return len(entries)

    async def index_from_folder(self, path, max_concurrent_tasks=8): 
        if not os.path.exists(path):
            logging.info(f"The knowledge folder, located in '{path}' does not exist. Skipping knowledge indexing.")
            return

        all_files = glob.glob(f"{path}/*")
        txt_files = [file for file in all_files if file.endswith('.txt')]
        non_txt_files = [file for file in all_files if not file.endswith('.txt')]

        for file in non_txt_files:
            logging.info(f"Error: {file} is not a .txt file. All knowledge must be in text files. Skipping.")

        if not txt_files:
            logging.info(f"No files in knowledge folder: '{path}', nothing to index'")
            return

        async def process_file(file_path):
            with open(file_path, 'r') as file:
                text = file.read()
                n_chunks = await self.chunk_and_index(text)
                return n_chunks

        tasks = [process_file(file) for file in txt_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_chunks = 0
        for file_path, result in zip(txt_files, results):
            if isinstance(result, Exception):
                logging.info(f"Error indexing {file_path}: {result}")
            elif isinstance(result, int):
                total_chunks += result
                logging.info(f"Indexed {file_path}: {result} chunks")

        logging.info(f"Total chunks indexed: {total_chunks}")

    def retrieve(self, related_text: str, n=5):
        return self._db_conn.search(
            VectorDatabaseConnection.Indexes.KNOWLEDGE, 
            related_text, 
            n
        )