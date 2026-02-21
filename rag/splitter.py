from langchain_text_splitters import RecursiveCharacterTextSplitter



def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)
