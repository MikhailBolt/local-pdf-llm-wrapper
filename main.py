import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# =========================
# Default settings
# =========================
DEFAULT_DB_FAISS_PATH = "vectorstore/db_faiss"
DEFAULT_PDF_PATH = "docs/doc.pdf"
DEFAULT_LOG_DIR = "logs"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama3"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
DEFAULT_HISTORY_LIMIT = 10


# =========================
# Helpers
# =========================
def ensure_directories(log_dir: str, pdf_path: str, db_path: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local PDF LLM Wrapper with FAISS + Ollama"
    )

    parser.add_argument("--pdf", default=os.getenv("PDF_PATH", DEFAULT_PDF_PATH),
                        help="Path to PDF file")
    parser.add_argument("--db", default=os.getenv("DB_FAISS_PATH", DEFAULT_DB_FAISS_PATH),
                        help="Path to FAISS vector database folder")
    parser.add_argument("--log-dir", default=os.getenv("LOG_DIR", DEFAULT_LOG_DIR),
                        help="Directory for logs")
    parser.add_argument("--embedding-model",
                        default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
                        help="HuggingFace embedding model name")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
                        help="Ollama model name")
    parser.add_argument("--chunk-size", type=int,
                        default=int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
                        help="Chunk size for text splitting")
    parser.add_argument("--chunk-overlap", type=int,
                        default=int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)),
                        help="Chunk overlap for text splitting")
    parser.add_argument("--top-k", type=int,
                        default=int(os.getenv("TOP_K", DEFAULT_TOP_K)),
                        help="Number of retrieved chunks")
    parser.add_argument("--history-limit", type=int,
                        default=int(os.getenv("HISTORY_LIMIT", DEFAULT_HISTORY_LIMIT)),
                        help="Number of last messages to keep in memory")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild of vector index")
    parser.add_argument("--stream", action="store_true",
                        help="Print answer with lightweight fake streaming")
    parser.add_argument("--stream-delay", type=float, default=0.0,
                        help="Delay between characters in stream mode, e.g. 0.005")

    return parser.parse_args()


def get_log_file_path(log_dir: str) -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    return Path(log_dir) / f"chat_{date_str}.txt"


def log_chat(log_dir: str, query: str, answer: str, pages: List[int], latency: float) -> None:
    log_path = get_log_file_path(log_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}]\n")
        f.write(f"User: {query}\n")
        f.write(f"AI: {answer}\n")
        f.write(f"Pages: {pages if pages else 'N/A'}\n")
        f.write(f"Latency: {latency:.2f}s\n")
        f.write("-" * 60 + "\n")


def validate_pdf(pdf_path: str) -> None:
    if not os.path.exists(pdf_path):
        print(f"(X) Error: PDF file not found at '{pdf_path}'.")
        print("Put your PDF there or pass a custom path with --pdf")
        sys.exit(1)

    if not pdf_path.lower().endswith(".pdf"):
        print(f"(X) Error: File '{pdf_path}' is not a PDF.")
        sys.exit(1)


def is_index_stale(pdf_path: str, db_path: str) -> bool:
    """
    Index is stale if:
    - vector db folder does not exist
    - FAISS files do not exist
    - PDF is newer than saved index files
    """
    db_dir = Path(db_path)
    index_file = db_dir / "index.faiss"
    meta_file = db_dir / "index.pkl"

    if not db_dir.exists():
        return True
    if not index_file.exists() or not meta_file.exists():
        return True

    pdf_mtime = Path(pdf_path).stat().st_mtime
    index_mtime = min(index_file.stat().st_mtime, meta_file.stat().st_mtime)

    return pdf_mtime > index_mtime


def build_vectorstore(
    pdf_path: str,
    db_path: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> FAISS:
    print("--- Processing PDF and building vector DB... ---")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        raise ValueError("PDF was loaded, but no text could be extracted.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = splitter.split_documents(documents)

    if not splits:
        raise ValueError("No text chunks were created from the PDF.")

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(db_path)

    return vectorstore


def load_or_create_vectorstore(args: argparse.Namespace) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)

    should_rebuild = args.rebuild or is_index_stale(args.pdf, args.db)

    if should_rebuild:
        if args.rebuild:
            print("(!) Force rebuild requested.")
        else:
            print("(!) PDF changed or index missing. Rebuilding vector DB...")
        return build_vectorstore(
            pdf_path=args.pdf,
            db_path=args.db,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

    print("--- Loading existing vector DB... ---")
    return FAISS.load_local(
        args.db,
        embeddings,
        allow_dangerous_deserialization=True
    )


def create_rag_chain(vectorstore: FAISS, model_name: str, top_k: int):
    llm = OllamaLLM(model=model_name)

    system_prompt = (
        "You are a professional AI assistant for answering questions about a local PDF document.\n"
        "Use ONLY the provided context to answer the user's question.\n"
        "If the answer is not present in the context, say clearly that you could not find it in the document.\n"
        "Do not invent facts. Be concise, accurate, and helpful.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    return rag_chain


def extract_source_pages(response: dict) -> List[int]:
    pages = []
    for doc in response.get("context", []):
        page = doc.metadata.get("page")
        if isinstance(page, int):
            pages.append(page + 1)
    return sorted(set(pages))


def print_answer(text: str, stream: bool = False, delay: float = 0.0) -> None:
    print("AI Answer: ", end="", flush=True)

    if not stream:
        print(text)
        return

    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        if delay > 0:
            time.sleep(delay)
    print()


def print_startup_info(args: argparse.Namespace) -> None:
    print("\n[SUCCESS] AI Ready!")
    print(f"PDF: {args.pdf}")
    print(f"Model: {args.model}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Vector DB: {args.db}")
    print(f"Top-K: {args.top_k}")
    print("Type 'exit', 'quit', or 'выход' to quit.\n")


def main() -> None:
    args = parse_args()

    ensure_directories(args.log_dir, args.pdf, args.db)
    validate_pdf(args.pdf)

    try:
        vectorstore = load_or_create_vectorstore(args)
    except Exception as e:
        print(f"(X) Failed to prepare vector DB: {e}")
        sys.exit(1)

    try:
        rag_chain = create_rag_chain(
            vectorstore=vectorstore,
            model_name=args.model,
            top_k=args.top_k
        )
    except Exception as e:
        print(f"(X) Failed to initialize LLM/RAG chain: {e}")
        sys.exit(1)

    chat_history = []
    print_startup_info(args)

    while True:
        try:
            query = input("> User: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not query:
            print("Please enter a question.")
            continue

        if query.lower() in ["exit", "quit", "выход"]:
            print("Goodbye!")
            break

        start_time = time.time()

        try:
            response = rag_chain.invoke({
                "input": query,
                "chat_history": chat_history
            })

            answer_text = response.get("answer", "").strip()
            if not answer_text:
                answer_text = "I could not generate an answer."

            pages = extract_source_pages(response)

            print_answer(
                answer_text,
                stream=args.stream,
                delay=args.stream_delay
            )

            elapsed = time.time() - start_time
            print(f"[Pages: {', '.join(map(str, pages)) if pages else 'N/A'}] | [{elapsed:.2f}s]")

            chat_history.extend([
                HumanMessage(content=query),
                AIMessage(content=answer_text)
            ])

            # Keep only last N messages
            if args.history_limit > 0:
                chat_history = chat_history[-args.history_limit:]

            log_chat(args.log_dir, query, answer_text, pages, elapsed)

        except Exception as e:
            print(f"(X) Error while processing question: {e}")


if __name__ == "__main__":
    main()