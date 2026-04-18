# Enterprise NLP Financial RAG System

An **AI-powered Financial Research Assistant** built using **Retrieval-Augmented Generation (RAG)** that enables semantic question answering and executive summarization over **18,755 earnings call transcripts** from publicly listed companies.

The system retrieves relevant transcript chunks using **FAISS vector search** and generates grounded, verifiable answers using **LLaMA 3.2** running locally via Ollama — with built-in faithfulness evaluation and relevance scoring.

> Built as part of an academic exploration of Enterprise NLP Systems and RAG architectures.

---

## Problem Statement

Financial analysts need to extract insights from thousands of pages of earnings call transcripts quickly and accurately.

- **Keyword search** fails to capture semantic meaning
- **Plain LLMs** hallucinate financial figures not in their training data
- **Manual reading** takes hours per transcript

This system solves all three problems — semantic retrieval finds the right context, and RAG grounds the LLM answer in real transcript data, making every answer verifiable with source citations.

---

## Key Features

- **Semantic Question Answering** — ask any financial question, get answers grounded in real transcripts
- **Executive Summarization** — generate structured summaries by company, year, and quarter
- **FAISS IVFFlat Index** — retrieves relevant chunks in ~0.1 seconds across 1.2M vectors
- **Query Rewriting** — expands user queries into richer financial language before search
- **Faithfulness Evaluation** — scores how grounded each answer is in the retrieved context
- **Relevance Scoring** — shows cosine similarity score for each source chunk
- **Latency Breakdown** — displays retrieval vs generation time separately
- **Boilerplate Filtering** — removes legal disclaimers and operator instructions from results
- **Streamlit UI** — clean web interface with collapsible source cards

---

## System Architecture

```
User Query
    ↓
Query Rewriter (LLaMA 3.2 via Ollama)
    ↓
Embedding Model (all-MiniLM-L6-v2 → 384-dim vector)
    ↓
FAISS IVFFlat Search (4,376 clusters, nprobe=10)
    ↓
Filtering (remove boilerplate, Q&A sections, duplicates)
    ↓
Prompt Builder (chunks + query → structured prompt)
    ↓
LLM Generation (LLaMA 3.2 via Ollama)
    ↓
Faithfulness Evaluation + Relevance Scoring
    ↓
Streamlit UI (answer + sources + metrics)
```

---

## Dataset

- **Source:** [Motley Fool Earnings Call Transcripts](https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts) (Kaggle)
- **Size:** 18,755 transcripts
- **Columns:** date, exchange, quarter, ticker, transcript
- **Processed into:** 1,196,893 chunks (900 characters, 150 character overlap)

> Large data files (FAISS index, embeddings, chunks) are not included in this repository due to GitHub file size limits. Run the pipeline scripts to regenerate them.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| FAISS (IVFFlat) | High-performance vector similarity search |
| Sentence Transformers (all-MiniLM-L6-v2) | 384-dimensional semantic embeddings |
| LLaMA 3.2 via Ollama | Local LLM for answer generation and query rewriting |
| Streamlit | Interactive web UI |
| Pandas / NumPy | Data processing and vector operations |
| functools.lru_cache | Query embedding caching |

---

## Project Structure

```
enterprise-nlp-financial-rag/
│
├── data/
│   ├── raw/                        # Raw transcript data (not in repo)
│   └── processed/                  # Chunks, embeddings, FAISS index (not in repo)
│
├── src/
│   ├── embeddings/
│   │   └── generate_embeddings.py  # Generate embeddings + build IVF index
│   │
│   ├── preprocessing/
│   │   └── load_data.py            # Clean, section-split, chunk transcripts
│   │
│   └── rag/
│       ├── retrieve_chunks.py      # FAISS search + filtering + relevance scores
│       ├── query_rewriter.py       # Expand queries into richer financial language
│       ├── prompt_builder.py       # Build structured prompt from chunks + query
│       ├── generate_answer.py      # LLM generation via Ollama
│       ├── rag_pipeline.py         # Full pipeline orchestration with latency tracking
│       ├── summarizer.py           # Company-level summarization mode
│       └── evaluator.py            # Faithfulness scoring module
│
├── app.py                          # Streamlit web application
├── build_ivf_index.py              # One-time script to build IVF index from embeddings
├── requirements.txt
└── README.md
```

---

## Pipeline Workflow

### 1. Data Preprocessing
Transcripts are cleaned, split into **prepared remarks** and **Q&A sections**, and chunked into 900-character pieces with 150-character overlap.
```bash
python src/preprocessing/load_data.py
```

### 2. Embedding Generation
Each chunk is converted into a 384-dimensional normalized vector using `all-MiniLM-L6-v2`.
```bash
python src/embeddings/generate_embeddings.py
```

### 3. IVF Index Construction
Build a FAISS IVFFlat index with 4,376 clusters from the generated embeddings. Only needs to be run once.
```bash
python build_ivf_index.py
```

### 4. Launch the App
```bash
ollama serve          # Start Ollama in one terminal
streamlit run app.py  # Launch the UI in another terminal
```

---

## Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed locally
- LLaMA 3.2 model pulled: `ollama pull llama3.2`

### Setup

```bash
# Clone the repository
git clone https://github.com/Vaishali-1234/enterprise-nlp-financial-rag.git
cd enterprise-nlp-financial-rag

# Install dependencies
pip install -r requirements.txt
```

---

## Key Technical Decisions

**Why RAG over Fine-tuning?**
Earnings data changes every quarter. Fine-tuning would require expensive GPU retraining each quarter. RAG lets us update the index in minutes and makes answers verifiable with source citations.

**Why FAISS IVFFlat over Flat index?**
Flat index searches all 1.2M vectors per query. IVFFlat clusters vectors into 4,376 groups and searches only the nearest 10 clusters — reducing search from 1.2M comparisons to ~2,700, achieving 0.1s retrieval.

**Why Ollama over OpenAI API?**
Financial data is sensitive — Ollama runs everything locally with no data leaving the machine. Also eliminates per-query API costs.

**Why normalize embeddings?**
Normalization makes vector length exactly 1 so only direction (meaning) matters. It also makes inner product equal to cosine similarity — giving us relevance scores for free from FAISS.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Faithfulness Score | Fraction of key answer claims verified in retrieved context (0-100%) |
| Relevance Score | Cosine similarity between query and each source chunk (0-1) |
| Latency Breakdown | Retrieval time (~0.1s) vs generation time (~2min) measured separately |

---

## Limitations

- Response time ~2 minutes due to CPU-based LLM (no GPU)
- No conversation memory — each query is independent
- Faithfulness scorer uses keyword matching, not semantic NLI
- Runs locally only — not deployed

---

## Future Improvements

- Cross-encoder re-ranking for improved retrieval accuracy
- Hybrid search combining FAISS semantic search with BM25 keyword search
- GPU deployment to reduce generation time from 2 minutes to under 10 seconds
- Conversation memory for follow-up questions
- Cloud deployment on HuggingFace Spaces or Streamlit Cloud

---

## Contributors

**Vaishali V** — [@Vaishali-1234](https://github.com/Vaishali-1234)
Computer Science Engineering, Lovely Professional University

**Shivali V** — [@Shivali-10](https://github.com/Shivali-10)
Computer Science Engineering, Lovely Professional University
