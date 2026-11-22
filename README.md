<div align="center">

# 📚 GenAI-Powered Enterprise Knowledge Assistant (RAG)

**An intelligent document question-answering system built with Retrieval-Augmented Generation**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.4-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-8E75B2?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Tech Stack](#-tech-stack) · [Learning Resources](#-learning-resources)

</div>

---

## 🎯 Overview

This project implements a **production-style Retrieval-Augmented Generation (RAG)** pipeline that enables users to upload PDF documents and ask natural-language questions against them. The system retrieves the most relevant passages using vector similarity search and generates grounded, citation-backed answers using Google's Gemini 2.5 Flash model.

> Built as a hands-on exploration of **Generative AI**, **RAG architectures**, and **LLM-powered knowledge retrieval** — core competencies in the modern AI/ML landscape.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **Multi-PDF Ingestion** | Upload and process multiple PDF documents simultaneously |
| 🔍 **Semantic Search** | Vector similarity search using Hugging Face sentence embeddings |
| 🤖 **Context-Grounded Answers** | LLM responses strictly grounded in retrieved document context |
| 📎 **Source Citations** | Every answer includes source filename and page number references |
| 🔎 **Chunk Inspection** | Expandable UI to inspect retrieved chunks with Chroma distance scores |
| 🛡️ **Prompt Safety** | System prompt prevents hallucination and prompt-injection attacks |
| ⚡ **Cached Embeddings** | Embedding model loaded once and cached across Streamlit reruns |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE (Streamlit)                    │
│                                                                      │
│   ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐    │
│   │  PDF Upload   │    │  Question    │    │  Answer + Sources  │    │
│   │  & Process    │    │  Input       │    │  + Chunk Inspector │    │
│   └──────┬───────┘    └──────┬───────┘    └────────▲───────────┘    │
└──────────┼───────────────────┼─────────────────────┼────────────────┘
           │                   │                     │
           ▼                   ▼                     │
┌──────────────────┐  ┌───────────────────┐  ┌──────┴───────────────┐
│  DOCUMENT LOADER │  │  RETRIEVAL ENGINE │  │  GENERATION ENGINE   │
│  ────────────────│  │  ─────────────────│  │  ────────────────────│
│  PyPDFLoader     │  │  Chroma Vector DB │  │  Gemini 2.5 Flash   │
│  ────────────────│  │  Similarity Search│  │  Context-Grounded   │
│  Recursive Text  │  │  Top-4 Retrieval  │  │  Prompt Template    │
│  Splitter        │  │                   │  │  Citation Generation│
│  (1000 / 150)    │  │                   │  │                     │
└────────┬─────────┘  └───────▲───────────┘  └─────────────────────┘
         │                    │
         ▼                    │
┌──────────────────────────────┐
│     EMBEDDING LAYER          │
│  ────────────────────────────│
│  all-MiniLM-L6-v2 (384-dim) │
│  HuggingFace Sentence        │
│  Transformers                │
└──────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- A [Google AI Studio API key](https://aistudio.google.com/apikey) (free tier available)

### 1. Clone the Repository

```bash
git clone https://github.com/Pranay890/GenAI-Powered-Enterprise-Knowledge-Assistant-RAG.git
cd GenAI-Powered-Enterprise-Knowledge-Assistant-RAG
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and add your Gemini API key:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your default browser.

---

## 🔬 How It Works

The RAG pipeline follows a structured five-stage workflow:

### Stage 1 — Document Ingestion
PDF files are uploaded through the Streamlit interface and parsed page-by-page using LangChain's `PyPDFLoader`. Each page retains its source filename and page number as metadata.

### Stage 2 — Chunking
Documents are split into overlapping chunks using `RecursiveCharacterTextSplitter`:
- **Chunk size:** 1,000 characters
- **Chunk overlap:** 150 characters
- Overlap ensures that context spanning chunk boundaries is preserved.

### Stage 3 — Embedding & Indexing
Each chunk is embedded into a 384-dimensional vector using the `all-MiniLM-L6-v2` sentence transformer model, then indexed into an in-memory **Chroma** vector store.

### Stage 4 — Retrieval
When a user submits a query, it is embedded using the same model and compared against the indexed vectors. The **top 4 most similar chunks** are retrieved using cosine similarity.

### Stage 5 — Generation
The retrieved chunks, along with the user's question, are passed to **Gemini 2.5 Flash** via a carefully engineered prompt template that:
- Restricts answers to the provided context only
- Prevents hallucination and fabrication
- Requires source citations (filename + page)
- Treats retrieved text as evidence, not as executable instructions

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Framework** | [LangChain](https://langchain.com) | Orchestration of the RAG pipeline |
| **LLM** | [Gemini 2.5 Flash](https://ai.google.dev) | Answer generation with context grounding |
| **Embeddings** | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 384-dim sentence embeddings via Hugging Face |
| **Vector Store** | [Chroma](https://www.trychroma.com) | In-memory vector similarity search |
| **Document Loader** | [PyPDF](https://github.com/py-pdf/pypdf) | PDF text extraction |
| **Text Splitter** | LangChain `RecursiveCharacterTextSplitter` | Chunk documents with overlap |
| **Frontend** | [Streamlit](https://streamlit.io) | Interactive web interface |
| **Environment** | [python-dotenv](https://github.com/theskumar/python-dotenv) | Secure API key management |

---

## 📁 Project Structure

```
genai-enterprise-rag-assistant/
│
├── app.py                  # Main application — RAG pipeline + Streamlit UI
├── requirements.txt        # Python dependencies (pinned versions)
├── .env.example            # Template for environment variables
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT License
└── README.md               # Project documentation (this file)
```

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|---|---|
| **In-memory Chroma** | Keeps the demo self-contained with zero infrastructure; swap to persistent Chroma or Pinecone for production |
| **all-MiniLM-L6-v2** | Lightweight (80 MB), fast inference, strong performance on semantic similarity benchmarks |
| **1,000-char chunks / 150 overlap** | Balances context richness with retrieval precision; overlap prevents information loss at boundaries |
| **Top-4 retrieval** | Provides sufficient context without overwhelming the LLM's context window |
| **Gemini 2.5 Flash** | Low-latency, cost-effective, high-quality generation; free tier available for experimentation |
| **System prompt guardrails** | Prevents the model from hallucinating, following injected instructions, or inventing policies |

---

## 📖 Learning Resources

This project demonstrates core concepts across **Generative AI**, **RAG**, and **LLM application development**. Here are curated resources to deepen your understanding:

### RAG (Retrieval-Augmented Generation)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/) — Official end-to-end RAG walkthrough
- [RAG From Scratch (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/) — Free course series on RAG fundamentals
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Original RAG paper by Lewis et al.

### LangChain
- [LangChain Documentation](https://python.langchain.com/docs/introduction/) — Comprehensive framework docs
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/concepts/lcel/) — Composable chain syntax used in this project

### Vector Databases & Embeddings
- [Chroma Documentation](https://docs.trychroma.com/) — Vector store used in this project
- [Sentence Transformers](https://www.sbert.net/) — Framework behind the embedding model
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — Embedding model benchmarks

### Generative AI & LLMs
- [Google AI for Developers](https://ai.google.dev/) — Gemini API documentation and quickstarts
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — The foundational Transformer paper
- [Prompt Engineering Guide](https://www.promptingguide.ai/) — Techniques for effective LLM prompting

---

## 🔮 Future Enhancements

- [ ] Persistent vector store with Chroma's disk-backed mode
- [ ] Multi-modal RAG (images, tables from PDFs)
- [ ] Conversational memory with LangChain `ConversationBufferMemory`
- [ ] Deployment on Streamlit Community Cloud or Google Cloud Run
- [ ] Evaluation framework with RAGAS metrics (faithfulness, relevancy)
- [ ] Support for additional document formats (DOCX, TXT, HTML)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by [Pranay Patil](https://github.com/Pranay890)**

*If you found this project helpful, consider giving it a ⭐!*

</div>

