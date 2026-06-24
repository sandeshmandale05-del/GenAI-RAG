from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
import os 
load_dotenv()

key=os.getenv("GOOGLE_API_KEY")
model=GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=key
)

doc1 = Document(
    page_content="virat kohli is one of the succeful and consistant batsman in IPL hitory. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team" : "RCB"}
)

doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    metadata={"team" : "MI"} 
)

doc3 = Document( page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document( page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document( page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Chennai Super Kings"}
)

docs=[doc1 , doc2 , doc3 , doc4 , doc5]

vector_store=Chroma(
    embedding_function=model,
    persist_directory="my_chroma_db",
    collection_name="sample"
)

# vector_store.add_documents(docs)

ids=vector_store.get()["ids"]

# vector_store.delete(ids=ids)
print("Total vectors:", vector_store._collection.count())
print(ids)

print(vector_store.similarity_search(
    query="who among this are a bowler? ",
    k=1
))