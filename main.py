import os
import sys
import json
import time
import hashlib
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()


DEFAULT_DB_FAISS_PATH = "vectorstore/db_faiss"
DEFAULT_DOCS_PATH = "docs"
DEFAULT_LOG_DIR = "logs"
DEFAULT_SESSION_FILE = "logs/session_history.json"
DEFAULT_MANIFEST_FILE = "vectorstore/index_manifest.json"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama3"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
DEFAULT_HISTORY_LIMIT = 10
DEFAULT_RETRIEVAL_TYPE = "similarity"  # similarity | mmr


def ensure_directories(log_dir: str, docs_path: str, db_path: str, manifest_file: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    docs = Path(docs_path)
    if docs.suffix.lower() == ".pdf":
        docs.parent.mkdir(parents=True, exist_ok=True)
    else:
        docs.mkdir(parents=True, exist_ok=True)

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_file).parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local PDF LLM Wrapper with FAISS + Ollama (single PDF or multi-PDF folder)"
    )

    parser.add_argument(
        "--docs-path",
        default=os.getenv("DOCS_PATH", DEFAULT_DOCS_PATH),
        help="Path to PDF file or folder with PDF files"
    )
    parser.add_argument(
        "--db",
        default=os.getenv("DB_FAISS_PATH", DEFAULT_DB_FAISS_PATH),
        help="Path to FAISS vector database folder"
    )
    parser.add_argument(
        "--log-dir",
        default=os.getenv("LOG_DIR", DEFAULT_LOG_DIR),
        help="Directory for logs"
    )
    parser.add_argument(
        "--session-file",
        default=os.getenv("SESSION_FILE", DEFAULT_SESSION_FILE),
        help="Path to session history JSON file"
    )
    parser.add_argument(
        "--manifest-file",
        default=os.getenv("MANIFEST_FILE", DEFAULT_MANIFEST_FILE),
        help="Path to index manifest JSON file"
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="HuggingFace embedding model name"
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
        help="Ollama model name"
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL),
        help="Ollama base URL, e.g. http://localhost:11434"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
        help="Chunk size for text splitting"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)),
        help="Chunk overlap for text splitting"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("TOP_K", DEFAULT_TOP_K)),
        help="Number of retrieved chunks"
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=int(os.getenv("HISTORY_LIMIT", DEFAULT_HISTORY_LIMIT)),
        help="Number of last messages to keep in memory"
    )
    parser.add_argument(
        "--retrieval-type",
        choices=["similarity", "mmr"],
        default=os.getenv("RETRIEVAL_TYPE", DEFAULT_RETRIEVAL_TYPE),
        help="Retriever strategy"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of vector index"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Print answer with lightweight fake streaming"
    )
    parser.add_argument(
        "--stream-delay",
        type=float,
        default=0.0,
        help="Delay between characters in stream mode, e.g. 0.005"
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Show detailed source chunks metadata"
    )

    return parser.parse_args()


def check_ollama_server(ollama_url: str, model_name: str) -> None:
    base = ollama_url.rstrip("/")
    tags_url = f"{base}/api/tags"

    try:
        with urllib.request.urlopen(tags_url, timeout=3) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as e:
        print("(X) Ollama server is not reachable.")
        print(f"    URL: {base}")
        print(f"    Error: {e}")
        print("    Start Ollama and try again, e.g.:")
        print(f"      ollama run {model_name}")
        sys.exit(1)

    try:
        data = json.loads(raw)
    except Exception:
        # If Ollama is reachable but response isn't JSON (very unlikely), don't block startup.
        return

    models = data.get("models", [])
    available_names = {m.get("name") for m in models if isinstance(m, dict)}
    if available_names and model_name not in available_names:
        print(f"(!) Ollama is running, but model '{model_name}' is not listed in /api/tags.")
        print("    You may need to pull it first, e.g.:")
        print(f"      ollama run {model_name}")


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


def save_session_history(session_file: str, chat_history: List[Any]) -> None:
    session_path = Path(session_file)
    session_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_history = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            serializable_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            serializable_history.append({"role": "assistant", "content": msg.content})

    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(serializable_history, f, ensure_ascii=False, indent=2)


def load_session_history(session_file: str) -> List[Any]:
    session_path = Path(session_file)
    if not session_path.exists():
        return []

    try:
        with open(session_path, "r", encoding="utf-8") as f:
            raw_history = json.load(f)

        chat_history = []
        for item in raw_history:
            if item.get("role") == "user":
                chat_history.append(HumanMessage(content=item.get("content", "")))
            elif item.get("role") == "assistant":
                chat_history.append(AIMessage(content=item.get("content", "")))
        return chat_history
    except Exception:
        return []


def collect_pdf_files(docs_path: str) -> List[Path]:
    path = Path(docs_path)

    if not path.exists():
        print(f"(X) Error: '{docs_path}' does not exist.")
        sys.exit(1)

    if path.is_file():
        if path.suffix.lower() != ".pdf":
            print(f"(X) Error: File '{docs_path}' is not a PDF.")
            sys.exit(1)
        return [path]

    pdf_files = sorted([p for p in path.glob("*.pdf") if p.is_file()])

    if not pdf_files:
        print(f"(X) Error: No PDF files found in '{docs_path}'.")
        print("Put one or more PDF files there and restart.")
        sys.exit(1)

    return pdf_files


def file_fingerprint(file_path: Path) -> str:
    stat = file_path.stat()
    raw = f"{file_path.resolve()}|{stat.st_mtime}|{stat.st_size}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_manifest_data(pdf_files: List[Path]) -> Dict[str, Any]:
    return {
        "files": [
            {
                "name": pdf.name,
                "path": str(pdf.resolve()),
                "fingerprint": file_fingerprint(pdf),
            }
            for pdf in pdf_files
        ]
    }


def load_manifest(manifest_file: str) -> Dict[str, Any]:
    path = Path(manifest_file)
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_manifest(manifest_file: str, data: Dict[str, Any]) -> None:
    path = Path(manifest_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_index_stale(pdf_files: List[Path], db_path: str, manifest_file: str) -> bool:
    db_dir = Path(db_path)
    index_file = db_dir / "index.faiss"
    meta_file = db_dir / "index.pkl"

    if not db_dir.exists():
        return True
    if not index_file.exists() or not meta_file.exists():
        return True

    current_manifest = build_manifest_data(pdf_files)
    saved_manifest = load_manifest(manifest_file)

    return current_manifest != saved_manifest


def build_vectorstore(
    pdf_files: List[Path],
    db_path: str,
    manifest_file: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[FAISS, List[str]]:
    print("--- Processing PDF files and building vector DB... ---")

    all_documents = []
    loaded_names = []

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        if not documents:
            print(f"(!) Skipping '{pdf_path.name}': no text extracted.")
            continue

        loaded_names.append(pdf_path.name)

        for doc in documents:
            doc.metadata["document_name"] = pdf_path.name
            doc.metadata["document_path"] = str(pdf_path.resolve())

        all_documents.extend(documents)

    if not all_documents:
        raise ValueError("No text could be extracted from the provided PDF files.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = splitter.split_documents(all_documents)

    if not splits:
        raise ValueError("No text chunks were created from the PDF files.")

    for idx, split in enumerate(splits):
        split.metadata["chunk_id"] = idx

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(db_path)
    save_manifest(manifest_file, build_manifest_data(pdf_files))

    return vectorstore, loaded_names


def load_or_create_vectorstore(args: argparse.Namespace, pdf_files: List[Path]) -> Tuple[FAISS, List[str]]:
    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
    should_rebuild = args.rebuild or is_index_stale(pdf_files, args.db, args.manifest_file)
    loaded_names = [pdf.name for pdf in pdf_files]

    if should_rebuild:
        if args.rebuild:
            print("(!) Force rebuild requested.")
        else:
            print("(!) PDF set changed or index missing. Rebuilding vector DB...")
        return build_vectorstore(
            pdf_files=pdf_files,
            db_path=args.db,
            manifest_file=args.manifest_file,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

    print("--- Loading existing vector DB... ---")
    try:
        vectorstore = FAISS.load_local(
            args.db,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore, loaded_names
    except Exception:
        print("(!) Existing vector DB is corrupted or incompatible. Rebuilding...")
        return build_vectorstore(
            pdf_files=pdf_files,
            db_path=args.db,
            manifest_file=args.manifest_file,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )


def create_rag_chain(
    vectorstore: FAISS,
    model_name: str,
    top_k: int,
    retrieval_type: str,
    ollama_url: str,
):
    try:
        llm = OllamaLLM(model=model_name, base_url=ollama_url)
    except TypeError:
        # Backward/forward compatibility with different langchain-ollama versions.
        llm = OllamaLLM(model=model_name)

    system_prompt = (
        "You are a professional AI assistant for answering questions about local PDF documents.\n"
        "Use ONLY the provided context to answer the user's question.\n"
        "If the answer is not present in the context, say clearly that you could not find it in the documents.\n"
        "Do not invent facts.\n"
        "If possible, keep the answer concise and well-structured.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    if retrieval_type == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": max(top_k * 2, 8)}
        )
    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    return rag_chain


def extract_source_pages(response: Dict[str, Any]) -> List[int]:
    pages = []
    for doc in response.get("context", []):
        page = doc.metadata.get("page")
        if isinstance(page, int):
            pages.append(page + 1)
    return sorted(set(pages))


def extract_source_details(response: Dict[str, Any]) -> List[str]:
    details = []
    for doc in response.get("context", []):
        page = doc.metadata.get("page")
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        source = doc.metadata.get("source", "unknown")
        document_name = doc.metadata.get("document_name", Path(source).name if source != "unknown" else "unknown")
        page_str = str(page + 1) if isinstance(page, int) else "N/A"
        details.append(
            f"document={document_name}, source={source}, page={page_str}, chunk={chunk_id}"
        )
    return details


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


def print_help() -> None:
    print("\nAvailable commands:")
    print("  help     - show this help message")
    print("  history  - show current session history")
    print("  clear    - clear current session history")
    print("  docs     - list loaded PDF documents")
    print("  status   - show current configuration")
    print("  reindex  - rebuild vector index")
    print("  exit     - quit the application\n")


def print_history(chat_history: List[Any]) -> None:
    if not chat_history:
        print("(i) Session history is empty.")
        return

    print("\n--- Session History ---")
    for msg in chat_history:
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        print(f"{role}: {msg.content}")
    print("-----------------------\n")


def print_docs(loaded_documents: List[str]) -> None:
    print("\n--- Loaded Documents ---")
    for idx, name in enumerate(loaded_documents, start=1):
        print(f"{idx}. {name}")
    print("------------------------\n")


def print_status(args: argparse.Namespace, loaded_documents: List[str], history_count: int) -> None:
    print("\n--- Current Status ---")
    print(f"Docs path: {args.docs_path}")
    print(f"Loaded PDFs: {len(loaded_documents)}")
    print(f"Model: {args.model}")
    print(f"Ollama URL: {args.ollama_url}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Vector DB: {args.db}")
    print(f"Manifest file: {args.manifest_file}")
    print(f"Top-K: {args.top_k}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Chunk overlap: {args.chunk_overlap}")
    print(f"Retrieval type: {args.retrieval_type}")
    print(f"History messages: {history_count}")
    print("----------------------\n")


def print_startup_info(args: argparse.Namespace, loaded_history_count: int, loaded_documents: List[str]) -> None:
    print("\n[SUCCESS] AI Ready!")
    print(f"Docs path: {args.docs_path}")
    print(f"Loaded PDFs: {len(loaded_documents)}")
    print(f"Model: {args.model}")
    print(f"Ollama URL: {args.ollama_url}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Vector DB: {args.db}")
    print(f"Top-K: {args.top_k}")
    print(f"Retrieval type: {args.retrieval_type}")
    print(f"Loaded session messages: {loaded_history_count}")
    print("Type 'help' to see available commands.")
    print("Type 'exit', 'quit', or 'выход' to quit.\n")


def rebuild_index(args: argparse.Namespace, pdf_files: List[Path]):
    return build_vectorstore(
        pdf_files=pdf_files,
        db_path=args.db,
        manifest_file=args.manifest_file,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


def main() -> None:
    args = parse_args()

    ensure_directories(args.log_dir, args.docs_path, args.db, args.manifest_file)
    pdf_files = collect_pdf_files(args.docs_path)

    check_ollama_server(args.ollama_url, args.model)

    try:
        vectorstore, loaded_documents = load_or_create_vectorstore(args, pdf_files)
    except Exception as e:
        print(f"(X) Failed to prepare vector DB: {e}")
        sys.exit(1)

    try:
        rag_chain = create_rag_chain(
            vectorstore=vectorstore,
            model_name=args.model,
            top_k=args.top_k,
            retrieval_type=args.retrieval_type,
            ollama_url=args.ollama_url,
        )
    except Exception as e:
        print(f"(X) Failed to initialize LLM/RAG chain: {e}")
        sys.exit(1)

    chat_history = load_session_history(args.session_file)

    if args.history_limit > 0:
        chat_history = chat_history[-args.history_limit:]

    print_startup_info(args, len(chat_history), loaded_documents)

    while True:
        try:
            query = input("> User: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            save_session_history(args.session_file, chat_history)
            break

        if not query:
            print("Please enter a question.")
            continue

        normalized = query.lower()

        if normalized in ["exit", "quit", "выход"]:
            print("Goodbye!")
            save_session_history(args.session_file, chat_history)
            break

        if normalized == "help":
            print_help()
            continue

        if normalized == "clear":
            chat_history = []
            save_session_history(args.session_file, chat_history)
            print("(i) Session history cleared.")
            continue

        if normalized == "history":
            print_history(chat_history)
            continue

        if normalized == "docs":
            print_docs(loaded_documents)
            continue

        if normalized == "status":
            print_status(args, loaded_documents, len(chat_history))
            continue

        if normalized == "reindex":
            try:
                pdf_files = collect_pdf_files(args.docs_path)
                vectorstore, loaded_documents = rebuild_index(args, pdf_files)
                rag_chain = create_rag_chain(
                    vectorstore=vectorstore,
                    model_name=args.model,
                    top_k=args.top_k,
                    retrieval_type=args.retrieval_type,
                    ollama_url=args.ollama_url,
                )
                print("(i) Vector index rebuilt successfully.")
            except Exception as e:
                print(f"(X) Failed to rebuild index: {e}")
            continue

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

            if args.show_sources:
                source_details = extract_source_details(response)
                if source_details:
                    print("--- Sources ---")
                    for item in source_details:
                        print(item)

            chat_history.extend([
                HumanMessage(content=query),
                AIMessage(content=answer_text)
            ])

            if args.history_limit > 0:
                chat_history = chat_history[-args.history_limit:]

            save_session_history(args.session_file, chat_history)
            log_chat(args.log_dir, query, answer_text, pages, elapsed)

        except Exception as e:
            print(f"(X) Error while processing question: {e}")


if __name__ == "__main__":
    main()