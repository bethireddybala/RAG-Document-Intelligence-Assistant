"""
utils/embeddings.py
====================
RAG Component: Embedding Generation
-------------------------------------
Responsibilities:
  1. Convert document text chunks into dense vector embeddings.
  2. Convert a user query string into a vector embedding for similarity search.

We use the sentence-transformers library with the lightweight
'all-MiniLM-L6-v2' model (≈22 MB), which runs entirely on CPU and
requires no API key.  The EmbeddingModel class is intentionally thin so
students can swap in any other model (OpenAI, Cohere, Google, etc.) by
changing only this file.
"""

from __future__ import annotations

import io
import sys
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.document_processor import DocumentChunk


# ---------------------------------------------------------------------------
# Constants — change these to swap the embedding model
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384   # dimensionality of all-MiniLM-L6-v2 output


# ---------------------------------------------------------------------------
# EmbeddingModel  (RAG Component: Embeddings)
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """
    Thin wrapper around a SentenceTransformer model.

    Why a wrapper?
    - Makes unit testing easier (mock this class).
    - Makes model swapping trivial (override _model or subclass).
    - Keeps the rest of the codebase independent of sentence-transformers.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Load the embedding model.

        The first call downloads the model weights from HuggingFace (~22 MB).
        Subsequent calls use the cached version.
        """
        self._model_name = model_name
        # On Windows inside Streamlit, sys.stderr is a non-TTY pipe that
        # causes tqdm (used internally by transformers) to raise
        # OSError: [Errno 22] Invalid argument when it tries to flush.
        # Redirect stderr to a safe buffer for the duration of the load.
        _real_stderr = sys.stderr
        try:
            sys.stderr = io.StringIO()
            self._model = SentenceTransformer(model_name)
        finally:
            sys.stderr = _real_stderr

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings.

        Returns
        -------
        np.ndarray of shape (n_texts, embedding_dim), dtype float32.
        """
        if not texts:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2-normalise → cosine = dot product
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns
        -------
        np.ndarray of shape (1, embedding_dim), dtype float32.
        """
        return self.embed_texts([query])

    def embed_chunks(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """
        Convenience method: embed a list of DocumentChunk objects.

        Returns
        -------
        np.ndarray of shape (n_chunks, embedding_dim).
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self._model.get_sentence_embedding_dimension()
