from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# 1. Загрузка и нарезка PDF
loader = PyPDFLoader("docs/your_file.pdf") # Положи любой PDF в папку docs
docs = loader.load()

# Режем на куски по 1000 символов с перекрытием, чтобы не терять смысл
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# 2. Создаем "мозг" для поиска (Embeddings) - качается один раз (~400mb)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Создаем векторную базу данных в памяти
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

# 4. Инициализируем модель и цепочку ответов
llm = Ollama(model="llama3")
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# 5. Интерактив
question = "What are the main conclusions of this document?"
print(f"\nQuestion: {question}")
response = qa_chain.invoke(question)
print(f"\nAnswer:\n{response['result']}")
