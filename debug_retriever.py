"""
debug_retriever.py
Run this from your project root:
    python debug_retriever.py

It shows exactly what chunks are retrieved for any query,
so you can see WHY the bot gives wrong answers.
"""
import os
from dotenv import load_dotenv
from rag.vectorstore import load_existing_vectorstore
from rag.retriever import get_retriever

load_dotenv()

vectorstore = load_existing_vectorstore()
if vectorstore is None:
    print("❌ No chroma_db found. Run the app first to build the vectorstore.")
    exit()

retriever = get_retriever(vectorstore)

TEST_QUERIES = [
    "what is the dress code",
    "men's dress code formal attire",
    "men dress code monday to friday",
    "and for men",
    "formal attire for women",
    "women dress code acceptable clothing",
    "data encryption",
    "leave policy",
]

print("=" * 70)
print("RETRIEVER DEBUG — showing top chunks for each query")
print("=" * 70)

for query in TEST_QUERIES:
    print(f"\n🔍 Query: '{query}'")
    print("-" * 60)
    docs = retriever.invoke(query)
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        dtype  = doc.metadata.get("type", "text")
        # Show first 300 chars of each chunk
        preview = doc.page_content[:300].replace("\n", " ")
        print(f"  [{i+1}] source={os.path.basename(source)} | type={dtype}")
        print(f"       {preview}...")
        print()

print("=" * 70)
print("If 'men dress code' retrieves harassment/leave chunks → wrong documents")
print("are scoring higher than dress code chunks in your vectorstore.")
print("Fix: delete chroma_db folder and restart the app to rebuild.")
print("=" * 70)