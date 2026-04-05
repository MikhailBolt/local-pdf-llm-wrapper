import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- Настройки ---
DB_FAISS_PATH = 'vectorstore/db_faiss'
PDF_PATH = "docs/doc.pdf"

def prepare_vector_db():
    os.makedirs("docs", exist_ok=True)
    os.makedirs("vectorstore", exist_ok=True)

    if not os.path.exists(PDF_PATH):
        print(f"(X) Error: Put your PDF file at '{PDF_PATH}' and restart.")
        exit()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(DB_FAISS_PATH):
        print("--- Loading Vector DB... ---")
        return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    print("--- Processing PDF... ---")
    loader = PyPDFLoader(PDF_PATH)
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load())
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(DB_FAISS_PATH)
    return vectorstore

# --- Инициализация ---
vectorstore = prepare_vector_db()
# Добавляем аргумент для стриминга
llm = OllamaLLM(model="llama3")

system_prompt = (
    "You are a professional AI Assistant. Use the provided context to answer. "
    "If unsure, say you don't know based on the document. "
    "\n\n {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vectorstore.as_retriever(), qa_chain)

# --- Chat Loop ---
chat_history = []
print("\n[SUCCESS] AI Ready! Ask anything (type 'exit' to quit):")

while True:
    query = input("\n> User: ")
    if query.lower() in ['exit', 'quit', 'выход']: break
    
    start_time = time.time()
    print("AI is thinking...", end="\r")
    
    try:
        # Получаем ответ
        response = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })
        
        print(f"AI Answer: {response['answer']}")
        
        # Обновляем историю (храним последние 5 сообщений)
        chat_history.extend([
            HumanMessage(content=query),
            AIMessage(content=response['answer'])
        ])
        chat_history = chat_history[-10:] 

        # Вывод источников и времени
        sources = sorted(list(set(doc.metadata.get("page", 0) + 1 for doc in response.get("context", []))))
        print(f"[Pages: {', '.join(map(str, sources))}] | [{time.time() - start_time:.2f}s]")
            
    except Exception as e:
        print(f"(X) Error: {e}")
