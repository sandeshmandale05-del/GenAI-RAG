from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',   # this is the path where all/some books are located or the location of a book you want to load . 
    glob='*.pdf',   # "**/*.txt"-> load all .txt files in all subfolders , "*.pdf" ->all .pdf files in a root directory , "data/*.csv"->all .csv files in a data folder ,  "**/*" -> all files (any types, all folders), ** -> recursive search therough any folder. 
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

for document in docs:
    print(document.metadata)