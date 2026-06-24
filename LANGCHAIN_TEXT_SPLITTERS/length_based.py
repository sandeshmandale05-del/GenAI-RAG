from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("LANGCHAIN_TEXT_SPLITTERS/dl-curriculum.pdf")
docs=loader.load()

splitter = CharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=0,
    separator=''
)
# print("\n ",docs[3].page_content, "\n")
result=splitter.split_documents(docs)
print(result[3].page_content)
print(len(result))
for i, doc in enumerate(result):
    print(f"Chunk {i}")
    print(doc.page_content)
    print("-"*30)