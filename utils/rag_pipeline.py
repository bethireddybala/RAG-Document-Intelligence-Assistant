"""
utils/rag_pipeline.py
======================
RAG Components: Prompt Augmentation + LLM Generation + Grounded Response + Source Citation
-------------------------------------------------------------------------------------------
This module ties together every RAG component into a single end-to-end pipeline:

  User Question
      ↓
  Question Embedding         (EmbeddingModel.embed_query)
      ↓
  Vector Similarity Search   (VectorStore.retrieve)
      ↓
  Relevant Document Chunks   (top-k results)
      ↓
  Augmented Prompt           (_build_prompt)
      ↓
  Gemini LLM                 (google-generativeai)
      ↓
  Grounded Answer + Sources  (RAGResponse)

Security note:
  The Gemini API key is read exclusively from environment variables or
  Streamlit secrets — it is NEVER hard-coded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import google.generativeai as genai

from utils.document_processor import DocumentChunk
from utils.embeddings import EmbeddingModel
from utils.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fallback answer when retrieval finds nothing relevant
NO_CONTEXT_ANSWER = (
    "I could not find sufficient information in the uploaded documents "
    "to answer this question."
)

# Gemini model to use — can be overridden via the constructor
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"


# ---------------------------------------------------------------------------
# Response data model  (RAG Component: Source Citation)
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    """
    Structured output of the RAG pipeline.

    Attributes
    ----------
    answer          : generated answer string
    sources         : list of (DocumentChunk, similarity_score) used as context
    augmented_prompt: the full prompt sent to the LLM (for transparency)
    error           : non-empty string if the pipeline encountered an error
    """
    answer: str
    sources: List[Tuple[DocumentChunk, float]] = field(default_factory=list)
    augmented_prompt: str = ""
    error: str = ""

    @property
    def has_error(self) -> bool:
        return bool(self.error)


# ---------------------------------------------------------------------------
# Prompt builder  (RAG Component: Prompt Augmentation)
# ---------------------------------------------------------------------------

def _build_prompt(question: str, context_chunks: List[Tuple[DocumentChunk, float]]) -> str:
    """
    RAG Component: Prompt Augmentation
    ------------------------------------
    Construct the augmented prompt:
      - System instructions (answer only from context, cite sources)
      - Retrieved context blocks (numbered)
      - The user question

    The numbered context blocks map directly to the source citations
    shown in the UI, so users can verify every claim.
    """
    context_parts = []
    for i, (chunk, score) in enumerate(context_chunks, start=1):
        context_parts.append(
            f"[Source {i}] File: {chunk.source} | Page: {chunk.page_number} "
            f"| Chunk: {chunk.chunk_id} | Relevance: {score:.2f}\n"
            f"{chunk.text}"
        )

    context_text = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a helpful document assistant that answers questions strictly based on the provided document context.

IMPORTANT INSTRUCTIONS:
1. Answer ONLY using the information found in the context below.
2. If the context does not contain enough information to answer the question, respond exactly with:
   "{NO_CONTEXT_ANSWER}"
3. Be concise and accurate.
4. When referencing information, mention which source it came from (e.g., "According to Source 1...").
5. Do not invent, extrapolate, or assume information not present in the context.

DOCUMENT CONTEXT:
{context_text}

USER QUESTION:
{question}

ANSWER:"""

    return prompt


# ---------------------------------------------------------------------------
# RAGPipeline  (orchestrator)
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG pipeline orchestrator.

    Usage
    -----
    pipeline = RAGPipeline(api_key="...", embedding_model=model, vector_store=store)
    response = pipeline.query("What is the main topic of the document?", top_k=5)
    print(response.answer)
    """

    def __init__(
        self,
        api_key: str,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        gemini_model: str = DEFAULT_GEMINI_MODEL,
    ):
        """
        Parameters
        ----------
        api_key         : Gemini API key (read from env/secrets — not hard-coded).
        embedding_model : pre-initialised EmbeddingModel instance.
        vector_store    : pre-populated VectorStore instance.
        gemini_model    : Gemini model identifier string.
        """
        if not api_key:
            raise ValueError(
                "Gemini API key is missing. "
                "Set GEMINI_API_KEY in your environment or Streamlit secrets."
            )

        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._gemini_model_name = gemini_model

        # Configure the Gemini client  (RAG Component: LLM Generation)
        genai.configure(api_key=api_key)
        self._llm = genai.GenerativeModel(gemini_model)

    # ------------------------------------------------------------------
    # Public query method
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 5) -> RAGResponse:
        """
        Run the full RAG pipeline for a given question.

        Parameters
        ----------
        question : the user's natural-language question.
        top_k    : number of document chunks to retrieve.

        Returns
        -------
        RAGResponse with answer, sources, and the augmented prompt.
        """
        if self._vector_store.is_empty:
            return RAGResponse(
                answer=NO_CONTEXT_ANSWER,
                error="Vector store is empty. Please upload and process documents first.",
            )

        # ---- Step 1: Question Embedding ----------------------------------
        try:
            query_embedding = self._embedding_model.embed_query(question)
        except Exception as exc:
            return RAGResponse(
                answer="",
                error=f"Embedding error: {exc}",
            )

        # ---- Step 2: Retrieval (Similarity Search) -----------------------
        try:
            retrieved: List[Tuple[DocumentChunk, float]] = self._vector_store.retrieve(
                query_embedding=query_embedding,
                top_k=top_k,
            )
        except Exception as exc:
            return RAGResponse(
                answer="",
                error=f"Retrieval error: {exc}",
            )

        if not retrieved:
            return RAGResponse(
                answer=NO_CONTEXT_ANSWER,
                sources=[],
            )

        # ---- Step 3: Prompt Augmentation ---------------------------------
        augmented_prompt = _build_prompt(question, retrieved)

        # ---- Step 4: LLM Generation  (RAG Component: LLM Generation) ----
        try:
            llm_response = self._llm.generate_content(augmented_prompt)
            answer = llm_response.text.strip()
        except Exception as exc:
            return RAGResponse(
                answer="",
                sources=retrieved,
                augmented_prompt=augmented_prompt,
                error=f"Gemini API error: {exc}",
            )

        # ---- Step 5: Grounded Response  (RAG Component: Grounded Response)
        # If the model echoes back the no-context sentinel, keep it clean.
        if not answer:
            answer = NO_CONTEXT_ANSWER

        return RAGResponse(
            answer=answer,
            sources=retrieved,            # Source Citation — passed to UI
            augmented_prompt=augmented_prompt,
        )


# ---------------------------------------------------------------------------
# Convenience: load API key from environment or Streamlit secrets
# ---------------------------------------------------------------------------

def load_api_key() -> Optional[str]:
    """
    Attempt to load the Gemini API key from:
      1. Environment variable  GEMINI_API_KEY
      2. Streamlit secrets      st.secrets["GEMINI_API_KEY"]

    Returns None if neither source has the key.
    """
    # Try environment variable first
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    # Try Streamlit secrets (only available when running inside Streamlit)
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY", "").strip()
        if key:
            return key
    except Exception:
        pass

    return None
