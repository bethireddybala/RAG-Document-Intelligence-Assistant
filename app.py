"""
app.py
=======
RAG Document Intelligence Assistant
-------------------------------------
Streamlit front-end that ties together all RAG components:

  Sidebar  : upload, API key input, Top-K setting, clear/reset
  Main page: title, RAG explanation, document status, Q&A, pipeline viz,
             conversation history, source citations

Run with:
    streamlit run app.py
"""

import os
import time

import streamlit as st

from utils.document_processor import process_document, DocumentChunk
from utils.embeddings import EmbeddingModel
from utils.vector_store import VectorStore
from utils.rag_pipeline import RAGPipeline, load_api_key


# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Document Intelligence Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_session_state():
    """Initialise all session-state keys on first run."""
    defaults = {
        "chat_history": [],          # list of {"role": ..., "content": ..., "sources": ...}
        "processed_docs": [],        # list of {filename, num_chunks, status}
        "embedding_model": None,     # EmbeddingModel singleton
        "vector_store": None,        # VectorStore singleton
        "rag_pipeline": None,        # RAGPipeline singleton (keyed by api_key)
        "rag_pipeline_key": "",      # api_key used to build the cached pipeline
        "api_key": "",               # Gemini API key (from input or env)
        "top_k": 5,                  # retrieval count
        "processing_done": False,    # whether documents have been indexed
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedding_model() -> EmbeddingModel:
    """
    Load the embedding model once and cache it across reruns.
    @st.cache_resource ensures the ~22 MB model is downloaded only once.
    """
    return EmbeddingModel()


def _reset_vector_store(embedding_dim: int) -> VectorStore:
    """Create (or recreate) a fresh VectorStore."""
    return VectorStore(embedding_dim=embedding_dim)


def _process_and_index_files(uploaded_files, top_k: int):
    """
    Process each uploaded file, embed chunks, and add them to the vector store.
    Updates st.session_state directly.
    """
    model: EmbeddingModel = get_embedding_model()
    vs = _reset_vector_store(model.dimension)

    processed_docs = []
    all_ok = True

    progress_bar = st.progress(0, text="Processing documents…")

    for i, uf in enumerate(uploaded_files):
        filename = uf.name
        progress_bar.progress((i) / len(uploaded_files), text=f"Processing {filename}…")

        try:
            file_bytes = uf.read()

            # ----- RAG Component: Document Ingestion -----
            chunks = process_document(file_bytes, filename)

            if not chunks:
                processed_docs.append({
                    "filename": filename,
                    "num_chunks": 0,
                    "status": "⚠️ Empty document — no text extracted",
                })
                continue

            # ----- RAG Component: Embeddings -----
            embeddings = model.embed_chunks(chunks)

            # ----- RAG Component: Vector Database -----
            vs.add_chunks(chunks, embeddings)

            processed_docs.append({
                "filename": filename,
                "num_chunks": len(chunks),
                "status": "✅ Indexed",
            })

        except ValueError as exc:
            processed_docs.append({
                "filename": filename,
                "num_chunks": 0,
                "status": f"❌ {exc}",
            })
            all_ok = False

        except Exception as exc:
            processed_docs.append({
                "filename": filename,
                "num_chunks": 0,
                "status": f"❌ Processing error: {exc}",
            })
            all_ok = False

    progress_bar.progress(1.0, text="Done!")
    time.sleep(0.4)
    progress_bar.empty()

    st.session_state["embedding_model"] = model
    st.session_state["vector_store"] = vs
    st.session_state["processed_docs"] = processed_docs
    st.session_state["processing_done"] = vs.total_chunks > 0
    st.session_state["top_k"] = top_k
    # Invalidate cached pipeline so it's rebuilt with the new vector store
    st.session_state["rag_pipeline"] = None

    return all_ok


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔍 RAG Assistant")
    st.caption("Document Intelligence Powered by Gemini")
    st.divider()

    # ---- API Key ----
    st.subheader("🔑 Gemini API Key")
    env_key = load_api_key()
    if env_key:
        st.success("API key loaded from environment / secrets.", icon="✅")
        st.session_state["api_key"] = env_key
    else:
        user_key = st.text_input(
            "Enter your Gemini API key",
            type="password",
            placeholder="AIza…",
            help="Get a free key at https://aistudio.google.com/app/apikey",
        )
        if user_key:
            st.session_state["api_key"] = user_key

    st.caption(
        "Tip: set `GEMINI_API_KEY` in your environment or `.streamlit/secrets.toml` "
        "to avoid entering it each time."
    )
    st.divider()

    # ---- Document Upload ----
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="You can upload multiple files. They are processed in-memory.",
    )

    # ---- Top-K setting ----
    top_k = st.slider(
        "Top-K retrieval chunks",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of document chunks retrieved per question. Higher = more context but slower.",
    )

    # ---- Process button ----
    process_btn = st.button(
        "⚙️ Process Documents",
        disabled=not uploaded_files,
        use_container_width=True,
        type="primary",
    )

    if process_btn and uploaded_files:
        with st.spinner("Indexing documents…"):
            ok = _process_and_index_files(uploaded_files, top_k)
        if ok:
            st.success(f"✅ {len(uploaded_files)} document(s) indexed successfully!")
        else:
            st.warning("Some documents could not be processed. See status below.")

    # ---- Clear/Reset ----
    st.divider()
    if st.button("🗑️ Clear All & Reset", use_container_width=True):
        for key in ["chat_history", "processed_docs", "vector_store", "processing_done"]:
            st.session_state[key] = [] if key != "processing_done" else False
        st.session_state["vector_store"] = None
        st.rerun()

    # ---- About ----
    st.divider()
    st.subheader("ℹ️ About")
    st.markdown(
        """
        This application demonstrates **Retrieval Augmented Generation (RAG)**:

        1. Upload your documents
        2. Ask questions about their content
        3. The app retrieves relevant passages and uses **Gemini** to generate a grounded answer

        Built with **Streamlit**, **FAISS**, **Sentence-Transformers**, and **Google Gemini**.
        """
    )


# ---------------------------------------------------------------------------
# Main page — title & RAG explanation
# ---------------------------------------------------------------------------

st.title("🔍 RAG Document Intelligence Assistant")
st.markdown(
    """
    > **What is RAG?**  
    > *Retrieval Augmented Generation* combines a **retrieval system** (finding relevant passages
    > from your documents) with a **large language model** (generating a natural-language answer).
    > The LLM is *grounded* in your documents, so it cannot hallucinate facts that aren't there.
    """
)
st.divider()


# ---------------------------------------------------------------------------
# RAG Pipeline Visualisation
# ---------------------------------------------------------------------------

with st.expander("📊 RAG Pipeline Visualisation", expanded=False):
    st.markdown(
        """
        ```
        ┌─────────────────────────────────────────────────────────────┐
        │                    RAG PIPELINE                             │
        ├─────────────────────────────────────────────────────────────┤
        │                                                             │
        │   📄 Uploaded Documents                                     │
        │           │                                                 │
        │           ▼                                                 │
        │   ✂️  Text Extraction & Chunking                            │
        │           │                                                 │
        │           ▼                                                 │
        │   🔢 Embedding Generation  (all-MiniLM-L6-v2)              │
        │           │                                                 │
        │           ▼                                                 │
        │   🗄️  FAISS Vector Store  ◄──── stored in memory           │
        │                                                             │
        │   ────────────────────────────────────────────────         │
        │                                                             │
        │   ❓ User Question                                          │
        │           │                                                 │
        │           ▼                                                 │
        │   🔢 Question Embedding                                     │
        │           │                                                 │
        │           ▼                                                 │
        │   🔍 Cosine Similarity Search  →  Top-K Chunks             │
        │           │                                                 │
        │           ▼                                                 │
        │   📝 Augmented Prompt  (question + context + instructions)  │
        │           │                                                 │
        │           ▼                                                 │
        │   🤖 Gemini LLM                                             │
        │           │                                                 │
        │           ▼                                                 │
        │   ✅ Grounded Answer + 📎 Source Citations                  │
        │                                                             │
        └─────────────────────────────────────────────────────────────┘
        ```
        """
    )


# ---------------------------------------------------------------------------
# Document Status
# ---------------------------------------------------------------------------

st.subheader("📄 Document Status")

if not st.session_state["processed_docs"]:
    st.info("No documents processed yet. Upload files in the sidebar and click **Process Documents**.")
else:
    import pandas as pd
    df = pd.DataFrame(st.session_state["processed_docs"])
    df.columns = ["Filename", "Chunks Indexed", "Status"]
    st.dataframe(df, use_container_width=True, hide_index=True)

    vs: VectorStore = st.session_state.get("vector_store")
    if vs:
        col1, col2 = st.columns(2)
        col1.metric("Total Chunks in Vector Store", vs.total_chunks)
        col2.metric("Top-K Retrieval", st.session_state["top_k"])

st.divider()


# ---------------------------------------------------------------------------
# Conversation History
# ---------------------------------------------------------------------------

st.subheader("💬 Conversation")

if st.session_state["chat_history"]:
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show source citations for assistant messages
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Retrieved Sources (click to inspect)", expanded=False):
                    for j, (chunk, score) in enumerate(msg["sources"], start=1):
                        st.markdown(
                            f"**Source {j}** — `{chunk.source}` | "
                            f"Page {chunk.page_number} | Chunk {chunk.chunk_id} | "
                            f"Similarity: `{score:.3f}`"
                        )
                        st.text_area(
                            label=f"Chunk {j} text",
                            value=chunk.text,
                            height=120,
                            key=f"src_{id(msg)}_{j}",
                            disabled=True,
                        )
else:
    st.caption("No conversation yet. Ask a question below.")


# ---------------------------------------------------------------------------
# Question Input
# ---------------------------------------------------------------------------

st.subheader("❓ Ask a Question")

col_q, col_btn = st.columns([5, 1])
with col_q:
    user_question = st.text_input(
        "Your question",
        placeholder="e.g. What is the main conclusion of the document?",
        label_visibility="collapsed",
    )
with col_btn:
    ask_btn = st.button("Ask ➤", type="primary", use_container_width=True)

col_clear, _ = st.columns([1, 4])
with col_clear:
    if st.button("🗑️ Clear Chat"):
        st.session_state["chat_history"] = []
        st.rerun()


# ---------------------------------------------------------------------------
# RAG Query execution
# ---------------------------------------------------------------------------

if ask_btn and user_question.strip():

    # -- Guard: documents must be processed --
    if not st.session_state["processing_done"]:
        st.warning("⚠️ Please upload and process documents before asking questions.")
        st.stop()

    # -- Guard: API key must be present --
    api_key = st.session_state.get("api_key", "").strip()
    if not api_key:
        st.error(
            "❌ Gemini API key is missing. "
            "Enter it in the sidebar or set the `GEMINI_API_KEY` environment variable."
        )
        st.stop()

    # Append user message to history
    st.session_state["chat_history"].append({
        "role": "user",
        "content": user_question,
        "sources": [],
    })

    with st.spinner("🔍 Searching documents and generating answer…"):
        try:
            # Reuse the cached pipeline; only rebuild if the api_key changed
            # or the vector store was re-indexed (processing_done flip).
            cached = st.session_state.get("rag_pipeline")
            if cached is None or st.session_state.get("rag_pipeline_key") != api_key:
                cached = RAGPipeline(
                    api_key=api_key,
                    embedding_model=st.session_state["embedding_model"],
                    vector_store=st.session_state["vector_store"],
                )
                st.session_state["rag_pipeline"] = cached
                st.session_state["rag_pipeline_key"] = api_key
            pipeline = cached
            response = pipeline.query(
                question=user_question,
                top_k=st.session_state["top_k"],
            )

            if response.has_error:
                answer_text = f"⚠️ **Error:** {response.error}"
            else:
                answer_text = response.answer

        except Exception as exc:
            answer_text = f"⚠️ **Unexpected error:** {exc}"
            response = None

    # Append assistant message
    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": answer_text,
        "sources": response.sources if response else [],
    })

    st.rerun()

elif ask_btn and not user_question.strip():
    st.warning("Please enter a question before clicking Ask.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "RAG Document Intelligence Assistant · "
    "Powered by [Google Gemini](https://aistudio.google.com/) · "
    "Built with [Streamlit](https://streamlit.io/)"
)
