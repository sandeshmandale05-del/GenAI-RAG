from langchain_community.retrievers import WikipediaRetriever
from dotenv import load_dotenv
import wikipedia
import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda

load_dotenv()

key=os.getenv("GOOGLE_API_KEY")

model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=key
)


query= "india pakistan war" 


retrivers = WikipediaRetriever(
    top_k_results=3 , 
    lang="en"
)
prompt=PromptTemplate(
    template="Summarize the following text in 5 lines:\n\n{text}",
    input_variables=["text"]
)

format_docs = RunnableLambda(
    lambda docs: {
        "text": "\n\n".join(doc.page_content for doc in docs)
    }
)
# docs=retrivers.invoke(query)
# print(len(docs))
# print(docs)
chain=retrivers |format_docs | prompt | model 
result =chain.invoke(query)
print(result)