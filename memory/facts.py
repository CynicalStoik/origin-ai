import chromadb
import uuid
from datetime import datetime
from config import CHROMA_PATH, FACTS_RETRIEVE_K


class FactMemory:
    def __init__(self, embed_fn):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name="facts",
            metadata={"hnsw:space": "cosine"}
        )
        self.embed = embed_fn

    def add(self, content: str, tags: list, fact_type: str = "semantic", time_span: str = "ongoing"):
        existing = self._find_similar(content, threshold=0.92)
        if existing:
            self._update(existing["id"], content, tags, fact_type, time_span)
            return

        fact_id = str(uuid.uuid4())
        self.collection.add(
            documents=[content],
            embeddings=[self.embed(content)],
            metadatas=[{
                "tags": ",".join(tags),
                "type": fact_type,
                "time_span": time_span,
                "timestamp": str(datetime.now()),
                "boost": 0.0
            }],
            ids=[fact_id]
        )

    def _find_similar(self, content: str, threshold: float = 0.92) -> dict | None:
        count = self.collection.count()
        if count == 0:
            return None

        results = self.collection.query(
            query_embeddings=[self.embed(content)],
            n_results=1,
            include=["documents", "metadatas", "distances"]
        )

        if not results["ids"][0]:
            return None

        dist = results["distances"][0][0]
        similarity = 1 - dist  # cosine: dist = 1 - cosine_sim

        if similarity >= threshold:
            return {
                "id": results["ids"][0][0],
                "text": results["documents"][0][0],
                "metadata": results["metadatas"][0][0]
            }
        return None

    def _update(self, fact_id: str, new_content: str, tags: list, fact_type: str, time_span: str):
        existing = self.collection.get(ids=[fact_id], include=["metadatas"])
        meta = existing["metadatas"][0] if existing["metadatas"] else {}
        meta.update({
            "tags": ",".join(tags),
            "type": fact_type,
            "time_span": time_span,
            "timestamp": str(datetime.now())
        })
        self.collection.update(
            ids=[fact_id],
            documents=[new_content],
            embeddings=[self.embed(new_content)],
            metadatas=[meta]
        )

    def retrieve_best(self, query: str, k: int = FACTS_RETRIEVE_K) -> dict | None:
        count = self.collection.count()
        if count == 0:
            return None

        k = min(k, count)

        results = self.collection.query(
            query_embeddings=[self.embed(query)],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        candidates = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            similarity = 1 - dist  # cosine: dist = 1 - cosine_sim
            boost = float(meta.get("boost", 0.0))
            final_score = similarity + boost

            candidates.append({
                "text": doc,
                "metadata": meta,
                "similarity": round(similarity, 4),
                "boost": round(boost, 4),
                "final_score": round(final_score, 4)
            })

        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return candidates[0] if candidates else None

    def boost_memory(self, query: str, boost_amount: float = 0.1, k: int = 5):
        count = self.collection.count()
        if count == 0:
            return

        k = min(k, count)

        results = self.collection.query(
            query_embeddings=[self.embed(query)],
            n_results=k,
            include=["metadatas"]
        )

        for meta, id_ in zip(results["metadatas"][0], results["ids"][0]):
            current_boost = float(meta.get("boost", 0.0))
            meta["boost"] = round(current_boost + boost_amount, 4)
            self.collection.update(ids=[id_], metadatas=[meta])

    def count(self) -> int:
        return self.collection.count()