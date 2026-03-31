from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama

# 1. Загружаем PDF
loader = PyPDFLoader("docs/твой_файл.pdf")
pages = loader.load_and_split()

# 2. Инициализируем модель
llm = Ollama(model="llama3")

# 3. Просим модель проанализировать первую страницу
context = pages[0].page_content
question = "Summarize the main idea of this page in 3 bullet points."

response = llm.invoke(f"Context: {context}\n\nQuestion: {question}")
print(response)
