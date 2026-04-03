import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Константы
DB_FAISS_PATH = 'vectorstore/db_faiss'
PDF_PATH = "docs/doc.pdf"

def prepare_vector_db():
    if not os.path.exists(PDF_PATH):
        print(f"Error: {PDF_PATH} not found!")
        exit()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Если база уже создана — просто загружаем её
    if os.path.exists(DB_FAISS_PATH):
        print("--- Loading existing vector database... ---")
        return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Если базы нет — создаем
    print("--- Creating new vector database from PDF... ---")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(DB_FAISS_PATH)
    return vectorstore

# Инициализация
vectorstore = prepare_vector_db()
llm = OllamaLLM(model="llama3")

system_prompt = (
    "You are a professional AI Assistant. Use the context to answer. "
    "If the answer is not in the context, say you don't know based on the document. "
    "\n\n {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)

# --- Chat Loop ---
print("\nReady! Ask me anything about the document (type 'exit' to quit):")
while True:
    query = input("\nYour Question: ")
    if query.lower() in ['exit', 'quit', 'выход']:
        break
    
    print("Thinking...")
    try:
        # Теперь response содержит не только 'answer', но и 'context'
        response = rag_chain.invoke({"input": query})
        
        print(f"\nAnswer: {response['answer']}")
        
        # Вытаскиваем уникальные номера страниц из метаданных
        sources = set()
        for doc in response.get("context", []):
            page_num = doc.metadata.get("page", 0) + 1  # Индекс страниц в PDF начинается с 0
            sources.add(page_num)
        
        if sources:
            sorted_sources = sorted(list(sources))
            print(f"Sources (Pages): {', '.join(map(str, sorted_sources))}")
            
    except Exception as e:
        print(f"Error connecting to Ollama: {e}. Is it running?")
