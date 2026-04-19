# rag-pdf-qa

Ask questions about PDFs in natural language. Answers come with inline citations pointing back to the source pages.

## Stack

- **LangChain** — RAG chain + document splitting
- **ChromaDB** — local vector store (persisted to disk)
- **SentenceTransformers** — free local embeddings (swap for OpenAI in `.env`)
- **Claude / GPT-4** — generation
- **Streamlit** — UI

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your API key
streamlit run ui/streamlit_app.py

or 

uv run streamlit run ui/streamlit_app.py --server.fileWatcherType=none
```

## Architecture

```
PDF → pdfplumber → RecursiveCharacterTextSplitter → ChromaDB
                                                         ↑
query → embed → MMR search ──────────────────────────────┘
                    ↓
              top-k chunks → prompt → LLM → answer [1][2]
```

MMR retrieval (`search_type="mmr"`) over plain similarity search to avoid getting 4 nearly-identical chunks when the same sentence appears across multiple pages.

Files are deduplicated by MD5 hash on upload — re-uploading the same PDF is a no-op.

## Config

| Variable | Default | |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `EMBEDDING_PROVIDER` | `local` | `local` or `openai` |
| `CHUNK_SIZE` | `800` | characters |
| `CHUNK_OVERLAP` | `150` | characters |
| `RETRIEVAL_K` | `4` | chunks per query |

## Tests

```bash
pytest tests/ -v
```
