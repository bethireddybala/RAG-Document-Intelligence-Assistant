"""
utils/document_processor.py
============================
RAG Component: Document Ingestion + Text Chunking
--------------------------------------------------
Responsibilities:
  1. Accept uploaded file objects from Streamlit.
  2. Extract raw text from PDF and TXT files.
  3. Clean and normalise the extracted text.
  4. Split the text into overlapping chunks suitable for embedding.

Each chunk is returned as a dict so downstream components are decoupled
from the extraction logic.
"""

import io
import re
from dataclasses import dataclass, field
from typing import List

import PyPDF2


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """A single piece of a document, ready to be embedded."""
    chunk_id: int                   # sequential index within the document
    source: str                     # original filename
    page_number: int                # 1-based page (PDF) or 0 if not applicable
    text: str                       # cleaned chunk text
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_text_from_pdf(file_bytes: bytes) -> List[dict]:
    """Return a list of {page_number, text} dicts, one per PDF page."""
    pages = []
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page_num, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        pages.append({"page_number": page_num, "text": raw})
    return pages


def _extract_text_from_txt(file_bytes: bytes) -> List[dict]:
    """Treat the whole TXT file as a single 'page'."""
    text = file_bytes.decode("utf-8", errors="replace")
    return [{"page_number": 1, "text": text}]


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Remove excessive whitespace and control characters."""
    # Replace form-feeds, carriage returns with spaces
    text = re.sub(r"[\r\f\v]", " ", text)
    # Collapse multiple blank lines into two newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Text chunking  (RAG Component: Text Chunking)
# ---------------------------------------------------------------------------

def _split_into_chunks(
    pages: List[dict],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> List[dict]:
    """
    Split page text into overlapping character-level chunks.

    Overlap ensures that context spanning chunk boundaries is not lost
    during retrieval — a core requirement of good RAG chunking strategy.

    Returns a list of {page_number, text, chunk_index} dicts.
    """
    chunks = []
    global_chunk_index = 0

    for page in pages:
        text = page["text"]
        page_num = page["page_number"]

        if not text.strip():
            continue

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "page_number": page_num,
                    "text": chunk_text,
                    "chunk_index": global_chunk_index,
                })
                global_chunk_index += 1

            # Move forward by (chunk_size - overlap) to create overlapping windows
            start += chunk_size - chunk_overlap

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_document(
    file_bytes: bytes,
    filename: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> List[DocumentChunk]:
    """
    RAG Component: Document Ingestion
    -----------------------------------
    Full pipeline: extract → clean → chunk.

    Parameters
    ----------
    file_bytes   : raw bytes of the uploaded file
    filename     : original filename (used for source attribution)
    chunk_size   : max characters per chunk
    chunk_overlap: characters shared between consecutive chunks

    Returns
    -------
    List[DocumentChunk] ready to be passed to the embedding module.
    """
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        pages = _extract_text_from_pdf(file_bytes)
    elif ext == "txt":
        pages = _extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Please upload PDF or TXT.")

    # Clean each page
    for p in pages:
        p["text"] = _clean_text(p["text"])

    raw_chunks = _split_into_chunks(pages, chunk_size, chunk_overlap)

    document_chunks = [
        DocumentChunk(
            chunk_id=c["chunk_index"],
            source=filename,
            page_number=c["page_number"],
            text=c["text"],
            metadata={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        )
        for c in raw_chunks
    ]

    return document_chunks
