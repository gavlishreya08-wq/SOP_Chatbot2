import streamlit as st
from sop_auto_sync_v2 import SOPAutoSync, rebuild_vectorstore
import os

st.set_page_config(page_title="Admin Panel", page_icon="⚙️", layout="centered")

st.title("⚙️ Admin Panel - SOP Management")

# Simple password protection
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("#  ## 🔐 Admin Login")
    password = st.text_input("Enter admin password:", type="password")
    
    if st.button("Login"):
        # Change this to your actual password
        if password == "admin123":  # ⚠️ CHANGE THIS PASSWORD!
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password")
    
    st.stop()

# Admin is authenticated
st.success("✅ Logged in as Admin")

st.markdown("---")

# Configuration
BASE_URL = "https://upaygoa.com/geltm/helpndoc"
DOWNLOAD_DIR = "./sop_documents"

# Show last sync info
if os.path.exists("sync_log.json"):
    import json
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

# Sync button
st.markdown("### 🔄 Sync SOP Documents")
st.markdown("This will:")
st.markdown("- Download new PDFs from the SOP website")
st.markdown("- Update changed PDFs")
st.markdown("- Rebuild the vector store for the chatbot")

if st.button("🚀 Start Sync", type="primary", use_container_width=True):
    with st.spinner("Syncing... This may take a few minutes..."):
        progress_placeholder = st.empty()
        
        # Create syncer
        syncer = SOPAutoSync(BASE_URL, DOWNLOAD_DIR)
        
        # Run sync
        progress_placeholder.info("📥 Downloading PDFs...")
        new_count, updated_count = syncer.sync()
        
        # Check if we need to rebuild
        if new_count > 0 or updated_count > 0:
            progress_placeholder.info("🔨 Rebuilding vector store...")
            try:
                rebuild_vectorstore()
                progress_placeholder.empty()
                
                st.success(f"""
                ✅ **Sync Complete!**
                
                - New documents: {new_count}
                - Updated documents: {updated_count}
                - Vector store: ✅ Rebuilt
                
                The chatbot now has the latest SOPs!
                """)
            except Exception as e:
                st.error(f"❌ Error rebuilding vector store: {str(e)}")
        else:
            progress_placeholder.empty()
            st.info("✅ No changes detected. All SOPs are up to date!")

st.markdown("---")

# Logout button
if st.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.rerun()

# Link to main chatbot
st.markdown("---")
st.markdown("### 🤖 Go to Chatbot")
st.markdown("[Open SOP Chatbot →](http://localhost:8501)", unsafe_allow_html=True)
