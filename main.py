import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Настройки ---
DB_FAISS_PATH = 'vectorstore/db_faiss'
PDF_PATH = "docs/doc.pdf"

def prepare_vector_db():
    # Создаем папки, если их нет
    os.makedirs("docs", exist_ok=True)
    os.makedirs("vectorstore", exist_ok=True)

    if not os.path.exists(PDF_PATH):
        print(f"(X) Error: Put your PDF file at '{PDF_PATH}' and restart.")
        exit()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(DB_FAISS_PATH):
        print("--- Loading Vector DB from disk... ---")
        return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    print("--- Creating new Vector DB (this may take a while)... ---")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(DB_FAISS_PATH)
    return vectorstore

# --- Инициализация ---
vectorstore = prepare_vector_db()
llm = OllamaLLM(model="llama3")

system_prompt = (
    "You are a professional AI Assistant. Use the provided context to answer accurately. "
    "If the answer is not in the context, say you don't know based on the document. "
    "\n\n {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vectorstore.as_retriever(), qa_chain)

# --- Chat Loop ---
print("\n[SUCCESS] System ready! Ask about your PDF (type 'exit' to quit):")

while True:
    query = input("\n> Your Question: ")
    if query.lower() in ['exit', 'quit', 'выход']:
        break
    
    start_time = time.time()
    print("Thinking...")
    
    try:
        response = rag_chain.invoke({"input": query})
        end_time = time.time()
        
        print(f"\nAnswer: {response['answer']}")
        
        # Источники
        sources = sorted(list(set(doc.metadata.get("page", 0) + 1 for doc in response.get("context", []))))
        print(f"Sources: Pages {', '.join(map(str, sources))}")
        print(f"--- (Processing time: {end_time - start_time:.2f}s) ---")
            
    except Exception as e:
        print(f"(X) Error: Check if Ollama is running. Detail: {e}")
