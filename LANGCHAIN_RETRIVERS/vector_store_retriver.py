from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

import os 
load_dotenv()

key=os.getenv("GOOGLE_API_KEY")
model=GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    temperature=0.7,
    google_api_key=key
)

document = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]
vector_store=Chroma.from_documents(
    embedding=model,
    documents=document,
    collection_name="retrival_db"
)

# converts vector store into reriver
retriver=vector_store.as_retriever(search_kwargs={"k" :1})

query="What is Chroma used for?"
result=retriver.invoke(query)

for i,doc in enumerate(result):
    print(f"\n ---result {i+1}---")
    print(doc.page_content)