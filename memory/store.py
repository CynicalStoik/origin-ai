from __future__ import annotations

import os
import config
import chromadb
from typing import Any
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

_client: chromadb.ClientAPI | None = None
_embed_fn: SentenceTransformerEmbeddingFunction | None = None

COLLECTION_NAMES = ("episodic", "semantic", "procedural", "perspective")


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    return _client


def _get_embed_fn() -> SentenceTransformerEmbeddingFunction:
    global _embed_fn
    if _embed_fn is None:
        print("[memory] Loading embedding model …")
        _embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL,
        )
    return _embed_fn


def get_collection(name: str) -> chromadb.Collection:
    assert name in COLLECTION_NAMES, f"Unknown collection: {name}"
    return _get_client().get_or_create_collection(
        name=name,
        embedding_function=_get_embed_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def add(
    collection_name: str,
    doc_id: str,
    document: str,
    metadata: dict[str, Any] | None = None,
):
    col = get_collection(collection_name)
    col.upsert(ids=[doc_id], documents=[document], metadatas=[metadata or {}])


def query(
    collection_name: str,
    query_text: str,
    n_results: int = 5,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return list of dicts with keys: id, document, metadata, distance."""
    col = get_collection(collection_name)
    if col.count() == 0:
        return []
    kwargs: dict[str, Any] = {
        "query_texts": [query_text],
        "n_results": min(n_results, col.count()),
    }
    if where:
        kwargs["where"] = where
    try:
        results = col.query(**kwargs)
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for i in range(len(results["ids"][0])):
        out.append(
            {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": (
                    results["distances"][0][i] if results["distances"] else None
                ),
            }
        )
    return out


def update_metadata(collection_name: str, doc_id: str, metadata: dict[str, Any]):
    col = get_collection(collection_name)
    col.update(ids=[doc_id], metadatas=[metadata])


def delete(collection_name: str, doc_id: str):
    col = get_collection(collection_name)
    col.delete(ids=[doc_id])
