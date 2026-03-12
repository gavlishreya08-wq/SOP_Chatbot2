import streamlit as st
from sop_auto_sync_v2 import SOPAutoSync
import os
import json

st.set_page_config(page_title="Admin Panel", page_icon="⚙️", layout="centered")

st.title("⚙️ Admin Panel - SOP Management")

# -----------------------------
# Authentication
# -----------------------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("## 🔐 Admin Login")
    password = st.text_input("Enter admin password:", type="password")

    if st.button("Login"):
        if password == "admin123":  # ⚠️ CHANGE THIS PASSWORD
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password")

    st.stop()

st.success("✅ Logged in as Admin")

st.markdown("---")

# -----------------------------
# Configuration
# -----------------------------

BASE_URL = "https://upaygoa.com/geltm/helpndoc"
DOWNLOAD_DIR = "./sop_documents"

# -----------------------------
# Show Sync Info
# -----------------------------

if os.path.exists("sync_log.json"):
    with open("sync_log.json", "r") as f:
        log = json.load(f)
        last_sync = log.get("last_sync", "Never")
        doc_count = len(log.get("documents", {}))

    st.info(f"""
    **Last Sync:** {last_sync}  
    **Total Documents:** {doc_count}
    """)
else:
    st.warning("No sync history found. Run sync for the first time.")

st.markdown("---")

# -----------------------------
# Sync Section
# -----------------------------

st.markdown("### 🔄 Sync SOP Documents")
st.markdown("This will:")
st.markdown("- Download new PDFs from the SOP website")
st.markdown("- Update changed PDFs")
st.markdown("- Incrementally update the vector store")

if st.button("🚀 Start Sync", type="primary", use_container_width=True):

    with st.spinner("Syncing... This may take a few minutes..."):

        progress_placeholder = st.empty()

        # Create syncer
        syncer = SOPAutoSync(BASE_URL, DOWNLOAD_DIR)

        # Run sync
        progress_placeholder.info("📥 Downloading PDFs...")
        new_count, updated_count, changed_files = syncer.sync()

        # If there are changes
        if changed_files:

            progress_placeholder.info("🧠 Updating vector store...")

            try:
                from rag.loader import load_pdfs
                from rag.splitter import split_docs
                from rag.vectorstore import load_existing_vectorstore

                # Load only changed PDFs
                docs = load_pdfs(filepaths=changed_files)

                # Split into chunks
                chunks = split_docs(docs)

                # Load existing vector store
                vectorstore = load_existing_vectorstore()

                # Add only new embeddings
                vectorstore.add_documents(chunks)

                progress_placeholder.empty()

                st.success(f"""
                ✅ **Sync Complete!**

                - New documents: {new_count}
                - Updated documents: {updated_count}
                - Vector store: ✅ Incrementally Updated

                The chatbot now has the latest SOPs!
                """)

            except Exception as e:
                progress_placeholder.empty()
                st.error(f"❌ Error updating vector store: {str(e)}")

        else:
            progress_placeholder.empty()
            st.info("✅ No changes detected. All SOPs are up to date!")

st.markdown("---")

# -----------------------------
# Logout
# -----------------------------

if st.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.rerun()

st.markdown("---")

# -----------------------------
# Link to Chatbot
# -----------------------------

st.markdown("### 🤖 Go to Chatbot")
st.markdown("[Open SOP Chatbot →](http://localhost:8501)", unsafe_allow_html=True)