from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]

        self.bm25 = BM25Okapi([t.lower().split() for t in self.texts])

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.embedder.encode(self.texts, normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int, bm25_top_n: int) -> List[Dict]:
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_idx = np.argsort(bm25_scores)[-bm25_top_n:]

        query_embedding = self.embedder.encode(query, normalize_embeddings=True)

        scored = []
        for idx in bm25_top_idx:
            sim = np.dot(query_embedding, self.embeddings[idx])
            scored.append((sim, idx))

        scored.sort(reverse=True)
        top_indices = [idx for _, idx in scored[:top_k]]

        return [self.chunks[i] for i in top_indices]
