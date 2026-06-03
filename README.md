# ⚡ RAGCHAT AI

> **Upload. Ask. Understand.**

An AI-powered document chatbot built with Retrieval Augmented Generation (RAG). Upload any PDF and get instant, context-grounded answers via semantic search and LLM generation.

🔗 **Live Demo:** https://ragchataiv1.streamlit.app

---

## 🚀 Features

- 📄 Upload any PDF document
- 🔷 Semantic chunking with HuggingFace embeddings
- 🔍 FAISS vector similarity search
- ⚡ Gemini 2.5 Flash LLM for answer generation
- 📊 Real-time metrics — pages, chunks, words, dimensions
- 🎨 Custom dark-theme UI built with Streamlit

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| LLM | Google Gemini 2.5 Flash |
| PDF Parsing | PyPDF2 |
| Framework | LangChain |

## ⚙️ Run Locally

```bash
git clone https://github.com/your-username/RAG_AI
cd RAG_AI
pip install -r requirements.txt
```

Add your API key to `.env`:
```
GOOGLE_API_KEY=your-key-here
```

```bash
streamlit run app.py
```

## 📌 RAG Pipeline

```
PDF Upload → Text Extraction → Chunking → Embedding → FAISS Index → Semantic Search → Gemini LLM → Answer
```
