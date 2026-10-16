"""
utils/vector_store.py
======================
RAG Components: Vector Database + Similarity Search + Retrieval
----------------------------------------------------------------
Responsibilities:
  1. Maintain an in-memory FAISS index of document chunk embeddings.
  2. Add new chunks (and their embeddings) to the index.
  3. Accept a query embedding and return the Top-K most similar chunks.

FAISS (Facebook AI Similarity Search) is a highly efficient library for
nearest-neighbour search over dense vectors.  We use IndexFlatIP
(inner-product / cosine similarity on L2-normalised vectors), which gives
exact nearest-neighbour results without approximation — perfect for the
document sizes a student will use.

The VectorStore object is designed to be stored in Streamlit session_state
so it persists across user interactions without re-processing.
"""

from __future__ import annotations

from typing import List, Tuple

import faiss
import numpy as np

from utils.document_processor import DocumentChunk


# ---------------------------------------------------------------------------
# VectorStore  (RAG Components: Vector Database + Similarity Search)
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Wraps a FAISS flat inner-product index with a parallel list of
    DocumentChunks for easy metadata retrieval after a search.
    """

    def __init__(self, embedding_dim: int):
        """
        Parameters
        ----------
        embedding_dim : dimensionality of the vectors to store (e.g. 384).
        """
        self._dim = embedding_dim
        # IndexFlatIP = exact inner-product search.
        # Because embeddings are L2-normalised, inner product == cosine similarity.
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(embedding_dim)
        self._chunks: List[DocumentChunk] = []   # parallel list to the FAISS index

    # ------------------------------------------------------------------
    # Ingestion  (RAG Component: Vector Database — storing embeddings)
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add document chunks and their pre-computed embeddings to the store.

        Parameters
        ----------
        chunks     : list of DocumentChunk objects (metadata).
        embeddings : float32 array of shape (len(chunks), embedding_dim).
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings."
            )

        # FAISS requires contiguous float32 arrays
        vecs = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(vecs)
        self._chunks.extend(chunks)

    # ------------------------------------------------------------------
    # Similarity Search  (RAG Component: Similarity Search)
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        RAG Component: Similarity Search
        ----------------------------------
        Find the `top_k` chunks most similar to `query_embedding`.

        Parameters
        ----------
        query_embedding : float32 array of shape (1, embedding_dim).
        top_k           : number of results to return.

        Returns
        -------
        List of (DocumentChunk, score) tuples, sorted by descending score.
        Score is cosine similarity ∈ [-1, 1]; higher = more relevant.
        """
        if self._index.ntotal == 0:
            return []

        # Cap top_k to the number of stored vectors
        k = min(top_k, self._index.ntotal)

        query_vec = np.ascontiguousarray(query_embedding, dtype=np.float32)
        scores, indices = self._index.search(query_vec, k)

        results: List[Tuple[DocumentChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS uses -1 for empty slots
                continue
            results.append((self._chunks[idx], float(score)))

        # Already sorted descending by FAISS, but be explicit
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Retrieval  (RAG Component: Retrieval)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        RAG Component: Retrieval
        -------------------------
        Public retrieval entry-point used by the RAG pipeline.

        Wraps similarity_search and applies an optional score threshold to
        discard irrelevant results (scores below `score_threshold`).

        Parameters
        ----------
        query_embedding  : float32 array of shape (1, embedding_dim).
        top_k            : max number of chunks to return.
        score_threshold  : minimum cosine similarity to keep a chunk.

        Returns
        -------
        Filtered list of (DocumentChunk, score) tuples.
        """
        results = self.similarity_search(query_embedding, top_k=top_k)
        if score_threshold > 0.0:
            results = [(chunk, score) for chunk, score in results if score >= score_threshold]
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset the index (e.g. when the user uploads new documents)."""
        self._index.reset()
        self._chunks.clear()

    @property
    def total_chunks(self) -> int:
        """Number of chunks currently indexed."""
        return self._index.ntotal

    @property
    def is_empty(self) -> bool:
        return self._index.ntotal == 0
