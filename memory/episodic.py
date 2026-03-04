import chromadb
from datetime import datetime
from config import CHROMA_PATH, EPISODIC_RETRIEVE_K


class EpisodicMemory:
    def __init__(self, embed_fn):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name="episodes",
            metadata={"hnsw:space": "cosine"}
        )
        self.embed = embed_fn

    def add(self, text: str, tags: list, episode_id: str):
        existing = self.collection.get(ids=[episode_id])
        if existing["ids"]:
            return
        self.collection.add(
            documents=[text],
            embeddings=[self.embed(text)],
            metadatas=[{
                "tags": ",".join(tags),
                "timestamp": str(datetime.now()),
                "boost": 0.0
            }],
            ids=[episode_id]
        )

    def retrieve_best(self, query: str, k: int = EPISODIC_RETRIEVE_K) -> dict | None:
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
            include=["metadatas", "documents"]
        )

        for meta, id_ in zip(results["metadatas"][0], results["ids"][0]):
            current_boost = float(meta.get("boost", 0.0))
            meta["boost"] = round(current_boost + boost_amount, 4)
            self.collection.update(ids=[id_], metadatas=[meta])
    def find_conflicting(self, new_content: str, threshold_low: float = 0.75, threshold_high: float = 0.92) -> dict | None:
        """
        Find an episode that is semantically related but potentially contradictory.
        """
        count = self.collection.count()
        if count == 0:
            return None

        results = self.collection.query(
            query_embeddings=[self.embed(new_content)],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            similarity = 1 - dist
            if threshold_low <= similarity < threshold_high:
                return {
                    "text": doc,
                    "metadata": meta,
                    "similarity": round(similarity, 4)
                }
        return None
    def count(self) -> int:
        return self.collection.count()