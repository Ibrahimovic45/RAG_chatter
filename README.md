# LLM-RAG (RAG_chatter)

This repository is an adaptation/demonstration of Retrieval-Augmented Generation (RAG) implemented as a Streamlit chatbot that answers questions over custom document collections stored as FAISS vector stores.

This README replaces the original high-level description and matches the actual code and layout in this repository.

## What this repo contains

- A Streamlit chat UI: `rag_chatbot.py` (the main entrypoint)
- Backend logic for document loading, embeddings, FAISS vector store handling, and retrieval+LLM chains: `pages/backend/rag_functions.py`
- A Streamlit page for document uploads: `pages/document_uploading.py`
- Several prebuilt FAISS vector-store directories in `vector store/` (e.g. `Harry_Potter_1`, `Williams`, etc.)
- A simple devcontainer configuration that auto-starts Streamlit for development: `.devcontainer/devcontainer.json`
- Dockerfile(s) and app config (present but may need adjustment for your environment)
- `requirements.txt` listing Python dependencies used by the app

## Key implementation details (things the README previously didn't match)

- The app uses Streamlit as the UI and LangChain + FAISS for retrieval. The LLM integration in this repo uses the `langchain_groq`/`ChatGroq` wrapper and expects a Groq API token in Streamlit secrets.
- Document/vector store handling:
  - The repo includes a `vector store/` directory with several saved FAISS stores. The app also downloads files from a Google Cloud Storage bucket named `rag_chat_bucket_2` (hard-coded in `pages/backend/rag_functions.py`) into a local `data/` folder.
  - The downloader expects Google Cloud credentials available to the runtime (see environment setup below).
- The Streamlit UI expects the user to provide a username (used as `session_id`) and select a vector store name. The app prepares a history-aware retriever and a retrieval chain before answering queries.

## Prerequisites / secrets

- Python 3.11 (devcontainer uses mcr.microsoft.com/devcontainers/python:1-3.11-bullseye)
- Google Cloud credentials (service account JSON) with read access to the GCS bucket `rag_chat_bucket_2` if you intend to download remote data. Set environment variable:

  - GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

- Groq API token: the code uses `st.secrets["token"]` to set `GROQ_API_KEY` inside `pages/backend/rag_functions.py` (function Llm()). Provide this token in Streamlit secrets (e.g., create a `.streamlit/secrets.toml` containing a `token` entry) or adapt the code to read from another env var.

- Optional: If you do not use the GCS downloader and only rely on the prebuilt vector stores in `vector store/`, you do not strictly need GCS credentials.

## Install & run (shortest path)

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Notes on requirements: `requirements.txt` lists both CPU and GPU versions of FAISS and some heavy models. If you are on a CPU-only environment, prefer a CPU-only FAISS package (for example `faiss-cpu`) and remove or avoid GPU-only entries.

3. Provide required secrets:

- Set GOOGLE_APPLICATION_CREDENTIALS if using GCS downloader:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

- Provide Groq API token to Streamlit secrets (create `.streamlit/secrets.toml`):

```toml
# .streamlit/secrets.toml
token = "your-groq-api-token"
```

4. Run the Streamlit app:

```bash
streamlit run rag_chatbot.py
```

Or open the repo in Codespaces / devcontainer: the devcontainer config (`.devcontainer/devcontainer.json`) contains a `postAttachCommand` that runs Streamlit and forwards port 8501.

## Where files and vector stores live

- The app writes or expects local data under `data/` (created at runtime). Prebuilt FAISS stores are in `vector store/<collection_name>/`.
- `pages/backend/rag_functions.py` hard-codes the GCS bucket name: `rag_chat_bucket_2` and lists blobs to let the user choose collections to download.
- The LLM model loading uses `ChatGroq` and a history-aware retriever created via LangChain helpers. See `pages/backend/rag_functions.py` for the implementation details.

## Common issues and notes

- Make sure the Groq token is available in Streamlit secrets before the app starts; otherwise the LLM initialization will fail.
- The downloader writes into `data/` using blob names; if multiple collections share names there may be collisions. The code attempts to put vector-store folders under `data/<name>/` when saving FAISS stores.
- The repository mixes Jupyter notebooks (majority language detected) with the Streamlit app code — notebooks may contain experiments and examples but the runnable app is `rag_chatbot.py`.
- If you'd like a minimal CPU-only requirements file or a simplified Dockerfile, I can prepare a trimmed `requirements_cpu.txt` and a Dockerfile adjusted for CPU-only usage.

---

If you want, I can commit this updated README to the repository now and/or produce a trimmed requirements file and an updated Dockerfile to make running on a small VM easier.
