from langchain_text_splitters import RecursiveCharacterTextSplitter
 
'''def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )
    return splitter.split_documents(docs)'''
'''def split_docs(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120
    )   '''
    
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " "
        ]
    )

    final_chunks = []
    global_chunk_counter = 0  

    for doc in docs:
        splits = splitter.split_documents([doc])

        for chunk in splits:
            chunk.metadata["chunk_id"] = f"{doc.metadata.get('source','sop')}_{global_chunk_counter}"
            chunk.metadata["page"] = doc.metadata.get("page", "unknown")
            final_chunks.append(chunk)

            global_chunk_counter += 1  

    return final_chunks