from langchain_community.llms import Ollama

llm = Ollama(model="llama3")
response = llm.invoke("Explain RAG in 2 sentences for a Senior AI Lead.")
print(response)
