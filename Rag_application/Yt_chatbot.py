from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_core.runnables import RunnableParallel , RunnableLambda , RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import os

load_dotenv()

key2=os.getenv("GOOGLE_API_KEY")
key=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Step 1a -Indexing(Document Ingestion)
video_id="YIszsqhLGIs"  #only the id not full URl

try:
    #if you dont care about language , this return the "best" one 
    transcript_list=YouTubeTranscriptApi().fetch(video_id)
    
    
    # flatten it to plain text
    transcript=" ".join(chunk.text for chunk in transcript_list)
    # print(transcript)
    
except TranscriptsDisabled:
    print("No caption available for this video")
 
 
    
#Step 1b - Indexing (TZext splitting)

splitter=RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200
)
chunks=splitter.create_documents([transcript])
print(len(chunks))


#step 1c & 1d - Indexing (Embedding generation and storing in vectorStore)

embedding=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store=Chroma.from_documents(
    chunks,
    embedding
)



#Step 2 - Retrival

retriver=vector_store.as_retriever(search_type="similarity" , search_kwargs={"k":4})

# print(retriver.invoke("why do rich people think diffrent"))

model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=key2
)

prompt=PromptTemplate(
    template="""you are helpfull assistant .
    Answer only from the provided transcript context.
    if the contenxt is insufficient , just say you dont know.
    
    {context}
    Question:{question}
    """,
    
    input_variables=["context" , "question"]
)

question="""what is the topic of "thinking" discussed in this video ? if yes then what was discussed"""
result=retriver_docs=retriver.invoke(question)
# print(result)

context_text = "\n\n".join(doc.page_content for doc in result)
# print(context_text)

final_prompt = prompt.invoke({"context": context_text, "question": question})
# print("this is final prompt    : ---   -",final_prompt)


# Generation

try:
    answer=model.invoke(final_prompt)
    # print("\n this is the ans     :  --------------------------" ,answer.content)
except Exception as e:
    print(e)
    
    
def format_docs(result):
    return " \n\n".join(doc.page_content for doc in result)
    
parallel_chain = RunnableParallel({
    'context': retriver | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parallel_chain.invoke('who is raj shamani')

parser = StrOutputParser()

main_chain = parallel_chain | prompt | model | parser

print(main_chain.invoke('Can you summarize the video'))