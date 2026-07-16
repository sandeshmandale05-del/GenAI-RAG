from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
load_dotenv()
import os 

key = os.getenv("GOOGLE_API_KEY")

model=GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=key,
    temperature=0.7
)

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

vector_store=FAISS.from_documents(
    documents=docs,
    embedding=model
)

retriever=vector_store.as_retriever(
    search_type="mmr",   # <-- This enables MMR
    search_kwargs={"k":3 , "lambda_mult":0.5} # k = top results, lambda_mult = relevance-diversity balance
)

query="what is langchain ? "

result =retriever.invoke(query)

for i , doc in enumerate(result):
    print(f"\n----  result {i+1} ---")
    print(docs)