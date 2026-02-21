import streamlit as st
from dotenv import load_dotenv
import os
import re
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rag.vectorstore import create_vectorstore, load_existing_vectorstore
from rag.loader import load_pdfs
from rag.splitter import split_docs
from rag.retriever import get_retriever
from cache.semantic_cache import SemanticCache
from sop_auto_sync_v2 import SOPAutoSync

load_dotenv()

st.set_page_config(page_title="SOP Chatbot", page_icon="🤖", layout="centered")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f172a, #1e293b, #020617); }
.main > div { max-width: 900px; margin: auto; }
h1 { text-align: center; color: #e2e8f0; }

.user-msg {
    background: #2563eb; color: white; padding: 12px 16px;
    border-radius: 12px 12px 2px 12px; margin: 8px 0;
    width: fit-content; margin-left: auto; max-width: 70%;
}

.bot-msg {
    background: #1e293b; color: #e2e8f0; padding: 12px 16px;
    border-radius: 12px 12px 12px 2px; margin: 8px 0;
    width: fit-content; max-width: 70%;
}

.followup-box {
    background: #334155; color: #cbd5e1; padding: 10px 14px;
    border-radius: 8px; margin: 8px 0;
    border-left: 3px solid #3b82f6;
    font-size: 0.95em; max-width: 70%;
}

.typing-indicator {
    background: #1e293b; color: #e2e8f0; padding: 12px 16px;
    border-radius: 12px 12px 12px 2px; margin: 8px 0;
    width: fit-content; max-width: 70%;
}

.typing-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background-color: #e2e8f0;
    margin: 0 2px;
    animation: typing 1.4s infinite;
}

.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing {
    0%, 60%, 100% { opacity: 0.3; }
    30% { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)


# ──  Session State ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_followup" not in st.session_state:
    st.session_state.pending_followup = None

if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False

if "quick_question" not in st.session_state:
    st.session_state.quick_question = None

if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = []

if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

if "show_admin_login" not in st.session_state:
    st.session_state.show_admin_login = False


# ── Helper Functions ──────────────────────────────────────────────
def track_question(q: str):
    """Add question to asked list, keep max 8 unique questions"""
    q = q.strip()
    if q.lower() in {"yes", "y", "no", "n", "ok", "okay", "sure", "nope", "yep", "yeah"}:
        return
    if q in st.session_state.asked_questions:
        st.session_state.asked_questions.remove(q)
    st.session_state.asked_questions.insert(0, q)
    st.session_state.asked_questions = st.session_state.asked_questions[:8]


def show_typing_indicator():
    return """
    <div class='typing-indicator'>
        <span class='typing-dot'></span>
        <span class='typing-dot'></span>
        <span class='typing-dot'></span>
    </div>
    """


def extract_followup_question(answer: str):
    lines = answer.strip().split('\n')
    for line in reversed(lines[-3:]):
        line = line.strip()
        if line and line.endswith('?') and len(line) < 150:
            return line
    return None


def clean_main_answer(answer: str):
    lines = answer.strip().split('\n')
    cleaned_lines = []
    for i, line in enumerate(lines):
        if i > 0 and line.strip().endswith('?') and len(line.strip()) < 150:
            if (i == len(lines) - 1) or (i > 0 and not lines[i-1].strip()):
                break
        cleaned_lines.append(line)
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    main_answer = '\n'.join(cleaned_lines).strip()
    return main_answer if main_answer else answer


def build_chat_text():
    """Build downloadable chat text from messages"""
    if not st.session_state.messages:
        return None
    lines = [
        "=" * 50,
        "    SOP CHATBOT - CONVERSATION HISTORY",
        f"    Downloaded: {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
        "=" * 50, ""
    ]
    for role, msg in st.session_state.messages:
        if role == "user":
            lines.append(f"YOU:  {msg}\n")
        elif role == "bot":
            lines.append(f"BOT:  {msg}\n")
        elif role == "followup":
            lines.append(f"BOT (follow-up):  {msg}\n")
    lines.append("=" * 50)
    return "\n".join(lines)


# ──  setup_system() ─────────────────────────────────────────────
@st.cache_resource
def setup_system():
    vectorstore = load_existing_vectorstore()

    if vectorstore is None:
        docs = load_pdfs()
        chunks = split_docs(docs)
        vectorstore = create_vectorstore(chunks)

    retriever = get_retriever(vectorstore)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_template("""
You are an intelligent internal company SOP assistant.

CRITICAL RULES - Follow exactly:
1. Answer ONLY using the provided SOP context below
2. If the answer is not in the context, say: "I could not find this information in the SOP."
3. Provide COMPLETE and COMPREHENSIVE answers - if a topic has multiple sections, subsections, or steps, cover ALL of them
4. For complex topics with multiple parts, organize your answer clearly with proper structure
5. Do NOT repeat the user's question in your answer
6. Do NOT make up or invent information
7. Do NOT provide lists of multiple questions
8. Do NOT number your questions

ANSWER LENGTH GUIDELINE:
- Simple questions (single concept): 3-5 lines
- Complex questions (multi-section topics): Provide FULL coverage of all sections, subsections, and details
- If the SOP contains multiple related sections, include ALL relevant information
- For procedures with steps, list ALL steps completely
- For topics with requirements, include ALL requirements
- Never truncate or summarize important details - completeness is critical

FOLLOW-UP RULE:
- After your main answer, if there is ONE related topic in the SOP context that would be helpful:
  * Ask ONE single, natural question about it (maximum 12 words)
  * Make it conversational, like you're talking to a colleague
  * Do NOT write "Follow-up:" or "Follow-up question:" or any prefix
  * Do NOT add "Would you like to know?" at the end
  * Just ask the question naturally
- If no relevant follow-up exists, just end your answer naturally

EXAMPLE 1 (Simple):
User: "What access control should be used?"
Your response: "According to the SOP, role-based access control (RBAC) should be implemented to ensure only authorized users can access or modify data, as stated in section 2.1 of the Access Control standards.

What are the different user types for access control?"

EXAMPLE 2 (Complex with multiple sections):
User: "What is the complete data backup procedure?"
Your response: "The SOP outlines a comprehensive data backup procedure with the following components:

**Backup Schedule:**
- Daily incremental backups at 2 AM for all critical systems
- Weekly full backups every Sunday at 1 AM
- Monthly archive backups retained for 12 months

**Backup Locations:**
- Primary: On-site NAS storage with RAID 6 configuration
- Secondary: Off-site cloud storage (AWS S3 with encryption)
- Tertiary: Physical tape backup stored in secure facility

**Verification Process:**
- Automated integrity checks run after each backup
- Monthly restore tests on random sample data
- Quarterly full disaster recovery drills

**Responsibilities:**
- IT Operations Team: Execute daily and weekly backups
- System Administrators: Monitor backup logs and verify completion
- Security Team: Ensure encryption and access controls
- Backup Coordinator: Monthly reporting and audit compliance

**Retention Policy:**
- Daily backups: 30 days
- Weekly backups: 90 days
- Monthly backups: 12 months
- Critical data: 7 years per regulatory requirements

All backup activities must be logged in the Backup Management System as per section 4.3.

Would you like to know about the disaster recovery procedures?"

Context from SOP:
{context}

User question:
{question}

Your answer:
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ──  Instantiate chain and cache ───────────────────────────────
qa_chain = setup_system()
cache = SemanticCache()


# ──  Sidebar UI ────────────────────────────────────────────────
with st.sidebar:

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_followup = None
            st.session_state.waiting_for_response = False
            st.session_state.asked_questions = []
            st.rerun()

    with col_b:
        chat_text = build_chat_text()
        if chat_text:
            st.download_button(
                label="Download Chat",
                data=chat_text,
                file_name=f"sop_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.button("Download Chat", disabled=True, use_container_width=True)

    st.markdown("---")
    st.title("💡 Recently Asked")

    if st.session_state.asked_questions:
        st.markdown("**Click to ask again:**")
        for q in st.session_state.asked_questions:
            if st.button(q, key=f"faq_{q}", use_container_width=True):
                st.session_state.quick_question = q
                st.rerun()
    else:
        st.markdown(
            "<p style='color: #64748b; font-size: 0.9em;'>Questions you ask will appear here for quick re-access.</p>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.subheader("🔒 Admin")

    if not st.session_state.is_admin:

        if st.button("Admin Login"):
            st.session_state.show_admin_login = True

        if st.session_state.show_admin_login:
            admin_password = st.text_input("Enter Admin Password", type="password")

            if admin_password == os.getenv("ADMIN_PASSWORD"):
                st.session_state.is_admin = True
                st.session_state.show_admin_login = False
                st.success("Admin Access Granted")
                st.rerun()

    else:
        st.success("Logged in as Admin")

        if st.button("🔄 Sync SOPs", use_container_width=True):
            with st.spinner("Syncing SOPs and updating system..."):
                syncer = SOPAutoSync(
                    base_url="https://upaygoa.com/geltm/helpndoc",
                    download_dir="./sop_documents"
                )
                new_files, updated_files = syncer.sync()

                if new_files or updated_files:
                    setup_system.clear()
                    qa_chain = setup_system()
                    st.success("SOPs synced and updated successfully!")
                else:
                    st.info("No new or updated SOPs found.")

        if st.button("Logout Admin"):
            st.session_state.is_admin = False
            st.rerun()


# ──  Main Chat UI ──────────────────────────────────────────────
st.title(" Standard Operating Procedures Chatbot 🤖")

for role, msg in st.session_state.messages:
    if role == "user":
        st.markdown(f"<div class='user-msg'>{msg}</div>", unsafe_allow_html=True)
    elif role == "bot":
        st.markdown(f"<div class='bot-msg'>{msg}</div>", unsafe_allow_html=True)
    elif role == "followup":
        st.markdown(f"<div class='followup-box'>💬 <strong>{msg}</strong><br><br><em>Would you like me to explain this? Type 'yes' or 'no'</em></div>", unsafe_allow_html=True)

typing_placeholder = st.empty()

with st.form("chat_form", clear_on_submit=True):
    question = st.text_input("Ask about SOP...", placeholder="Type your question here...")
    send = st.form_submit_button("Send")


# ── 7️⃣ Message Handling ──────────────────────────────────────────
YES_WORDS = {"yes", "y", "ok", "okay", "sure", "continue", "please", "yep", "yeah", "yup"}
NO_WORDS = {"no", "n", "nope", "skip", "not now", "no thanks", "nah"}


# FAQ Handler
if st.session_state.quick_question:
    question = st.session_state.quick_question
    st.session_state.quick_question = None

    track_question(question)
    st.session_state.messages.append(("user", question))
    typing_placeholder.markdown(show_typing_indicator(), unsafe_allow_html=True)

    cached_answer = cache.search(question)
    if cached_answer:
        answer = cached_answer
    else:
        answer = qa_chain.invoke(question)
        cache.add(question, answer)

    typing_placeholder.empty()
    followup = extract_followup_question(answer)
    main_answer = clean_main_answer(answer)
    st.session_state.messages.append(("bot", main_answer))

    if followup:
        st.session_state.messages.append(("followup", followup))
        st.session_state.pending_followup = followup
        st.session_state.waiting_for_response = True
    else:
        st.session_state.pending_followup = None
        st.session_state.waiting_for_response = False

    st.rerun()


# Chat Input Handler
if send and question.strip():
    q_clean = question.lower().strip()

    if st.session_state.waiting_for_response:
        st.session_state.messages.append(("user", question))

        if q_clean in NO_WORDS:
            st.session_state.messages.append(("bot", "No problem! Feel free to ask if you have any other questions about the SOP."))
            st.session_state.pending_followup = None
            st.session_state.waiting_for_response = False
            st.rerun()

        elif q_clean in YES_WORDS:
            if st.session_state.pending_followup:
                real_question = st.session_state.pending_followup
                typing_placeholder.markdown(show_typing_indicator(), unsafe_allow_html=True)

                cached_answer = cache.search(real_question)
                if cached_answer:
                    answer = cached_answer
                else:
                    answer = qa_chain.invoke(real_question)
                    cache.add(real_question, answer)

                typing_placeholder.empty()
                new_followup = extract_followup_question(answer)
                main_answer = clean_main_answer(answer)
                st.session_state.messages.append(("bot", main_answer))

                if new_followup:
                    st.session_state.messages.append(("followup", new_followup))
                    st.session_state.pending_followup = new_followup
                    st.session_state.waiting_for_response = True
                else:
                    st.session_state.pending_followup = None
                    st.session_state.waiting_for_response = False
            else:
                st.session_state.waiting_for_response = False
                st.session_state.pending_followup = None

            st.rerun()

        else:
            st.session_state.waiting_for_response = False
            st.session_state.pending_followup = None
            track_question(question)
            typing_placeholder.markdown(show_typing_indicator(), unsafe_allow_html=True)

            cached_answer = cache.search(question)
            if cached_answer:
                answer = cached_answer
            else:
                answer = qa_chain.invoke(question)
                cache.add(question, answer)

            typing_placeholder.empty()
            followup = extract_followup_question(answer)
            main_answer = clean_main_answer(answer)
            st.session_state.messages.append(("bot", main_answer))

            if followup:
                st.session_state.messages.append(("followup", followup))
                st.session_state.pending_followup = followup
                st.session_state.waiting_for_response = True

            st.rerun()

    else:
        st.session_state.messages.append(("user", question))
        track_question(question)
        typing_placeholder.markdown(show_typing_indicator(), unsafe_allow_html=True)

        cached_answer = cache.search(question)
        if cached_answer:
            answer = cached_answer
        else:
            answer = qa_chain.invoke(question)
            cache.add(question, answer)

        typing_placeholder.empty()
        followup = extract_followup_question(answer)
        main_answer = clean_main_answer(answer)
        st.session_state.messages.append(("bot", main_answer))

        if followup:
            st.session_state.messages.append(("followup", followup))
            st.session_state.pending_followup = followup
            st.session_state.waiting_for_response = True
        else:
            st.session_state.pending_followup = None
            st.session_state.waiting_for_response = False

        st.rerun()