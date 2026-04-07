# 📄 Local PDF LLM Wrapper (RAG)

Local Retrieval-Augmented Generation (RAG) system for querying PDF documents using **LangChain**, **FAISS**, **HuggingFace embeddings**, and **Ollama (LLaMA3)**.

The system allows you to ask questions about a local PDF and get accurate answers grounded in the document, with source page tracking.

---

## 🚀 Features

- Semantic search over PDF using FAISS
- Local LLM via Ollama (no API required)
- Context-aware answers (RAG pipeline)
- Source page tracking in answers
- Conversational memory (multi-turn chat)
- Automatic reindexing when PDF changes
- Configurable via CLI and environment variables
- Chat logging with timestamps and latency
- Adjustable retrieval (`top-k`)

---

## 🏗 Architecture

```
PDF → Text Splitting → Embeddings → FAISS Vector DB
↓
User Query → Retriever → Context
↓
LLM (Ollama) → Answer + Sources
```

---

## 📦 Tech Stack

- **Python**
- **LangChain**
- **FAISS**
- **HuggingFace Embeddings**
- **Ollama (LLaMA3)**
- **PyPDF**

---

## ⚙️ Installation

### 1. Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/local-pdf-llm-wrapper.git
cd local-pdf-llm-wrapper
```

---

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Install and run Ollama

Download: https://ollama.com

```bash
ollama run llama3
```

---

## 📄 Usage

### 1. Put your PDF

```
docs/doc.pdf
```

### 2. Run the app

```bash
python main.py
```

---

## 💬 Example

```
> User: What is the main topic of the document?
AI Answer: ...
[Pages: 2, 5] | [1.42s]
```

---

## 🛠 CLI Options

```bash
python main.py \
  --pdf docs/doc.pdf \
  --model llama3 \
  --top-k 4 \
  --chunk-size 1000 \
  --chunk-overlap 150 \
  --stream
```

---

## ⚙️ Environment Variables

You can configure via `.env`:

```
PDF_PATH=docs/doc.pdf
LLM_MODEL=llama3
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=4
```

---

## 📁 Project Structure

```
local-pdf-llm-wrapper/
│
├── main.py
├── requirements.txt
├── README.md
│
├── docs/              # PDF files
├── logs/              # chat logs
├── vectorstore/       # FAISS index
```

---

## 🧠 How it works

1. PDF is loaded and split into chunks  
2. Each chunk is converted into embeddings  
3. FAISS stores vectors locally  
4. User query → embedding → similarity search  
5. Top-k chunks passed to LLM  
6. LLM generates answer grounded in context  

---

## 📊 Future Improvements

- [ ] Real streaming from LLM (not simulated)
- [ ] Web UI (Streamlit / FastAPI)
- [ ] Multi-PDF support
- [ ] Hybrid search (BM25 + embeddings)
- [ ] Evaluation metrics (RAG quality)
- [ ] Chunk caching optimization

---

## 🧑‍💻 Author

Mikhail B  
AI / ML / GenAI Engineer  

---

## 📜 License

MIT
