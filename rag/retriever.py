def get_retriever(vectorstore):

    return vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={
            "k": 3   # only top 3 chunks
        }
    )