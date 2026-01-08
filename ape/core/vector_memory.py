from __future__ import annotations
import asyncio
import json
import os
from typing import List, Dict, Any
from loguru import logger
import numpy as np
import faiss

from ape.settings import settings
from ape.utils import get_ollama_model_info
import ollama

class VectorMemory:
    """Manages long-term vector memory using FAISS and Ollama embeddings."""

    def __init__(self):
        self.ollama_client = ollama.Client(host=str(settings.OLLAMA_BASE_URL))
        self.index_path = os.path.join(settings.VECTOR_DB_PATH, "faiss.index")
        self.metadata_path = os.path.join(settings.VECTOR_DB_PATH, "metadata.json")
        self.index: faiss.Index | None = None
        self.metadata: Dict[int, Dict[str, Any]] = {}

        self.embedding_dimension = settings.EMBEDDING_SIZE or 384
        if settings.EMBEDDING_SIZE:
            logger.info(f"Using configured embedding dimension: {self.embedding_dimension}")

    async def _init_db(self):
        """Initializes the FAISS index and metadata from files."""
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)

        # ------------------------------------------------------------------
        # If explicit size wasn't set, try to detect it from the model.
        # This MUST be done here because get_ollama_model_info is async.
        # ------------------------------------------------------------------
        if not settings.EMBEDDING_SIZE:
            try:
                model_info = await get_ollama_model_info(settings.EMBEDDING_MODEL)
                detected = model_info.get('embedding_length')
                if detected:
                    self.embedding_dimension = detected
                    logger.info(f"Detected embedding dimension for {settings.EMBEDDING_MODEL}: {self.embedding_dimension}")
                else:
                    logger.warning(f"Could not determine embedding dimension from model info. Using default: {self.embedding_dimension}")
            except Exception as e:
                logger.warning(f"Failed to get embedding model info: {e}. Falling back to default dimension {self.embedding_dimension}.")
        if os.path.exists(self.index_path):
            logger.info(f"Loading FAISS index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.metadata = {int(k): v for k, v in json.load(f).items()}
        else:
            logger.info("Creating new FAISS index.")
            self.index = faiss.IndexFlatL2(self.embedding_dimension)

    def _save_storage(self):
        """Saves the FAISS index and metadata to files."""
        if self.index:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f)
            logger.info("Saved FAISS index and metadata.")

    async def _embed_and_store(self, text: str, metadata: dict | None = None):
        """The actual workhorse, designed to be run in the background."""
        try:
            embedding = self.ollama_client.embeddings(
                model=settings.EMBEDDING_MODEL,
                prompt=text
            )['embedding']

            if self.index is not None:
                vector = np.array([embedding], dtype=np.float32)
                new_id = self.index.ntotal
                self.index.add(vector)
                self.metadata[new_id] = {"text": text, "metadata": metadata}
                self._save_storage()
                logger.info(f"Successfully embedded and stored text: {text[:50]}...")
        except Exception as e:
            logger.error(f"Background embedding task failed: {e}")

    def add(self, text: str, metadata: dict | None = None):
        """
        Schedules the generation of an embedding and storage in the vector DB
        as a background task.
        """
        logger.debug(f"Scheduling embedding for text: {text[:50]}...")
        asyncio.create_task(self._embed_and_store(text, metadata))

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Performs a semantic search against the vector DB."""
        if self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = self.ollama_client.embeddings(
            model=settings.EMBEDDING_MODEL,
            prompt=query
        )['embedding']
        
        query_vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i in self.metadata:
                meta = self.metadata[i]
                results.append({
                    "id": i,
                    "text": meta["text"],
                    "metadata": meta["metadata"],
                    "distance": float(dist)
                })
        return results

# Singleton instance
_vector_memory_instance = None

async def get_vector_memory() -> VectorMemory:
    global _vector_memory_instance
    if _vector_memory_instance is None:
        _vector_memory_instance = VectorMemory()
        await _vector_memory_instance._init_db()
    return _vector_memory_instance