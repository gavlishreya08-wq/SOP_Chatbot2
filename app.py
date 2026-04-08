import streamlit as st
from dotenv import load_dotenv
import os
import re
import unicodedata
from datetime import datetime
import pymupdf as fitz
import shutil

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag.vectorstore import create_vectorstore, load_existing_vectorstore
from rag.loader import load_pdfs
from rag.splitter import split_docs
from rag.retriever import get_retriever
from sop_auto_sync_v2 import SOPAutoSync
from sop_metadata_fixed import get_metadata_handler

load_dotenv()

# Initialize metadata handler
metadata_handler = get_metadata_handler()

st.set_page_config(page_title="SOP Chatbot", page_icon="🤖", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');
.stApp { background: linear-gradient(135deg, #0f172a, #1e293b, #020617); }
.main > div { max-width: 900px; margin: auto; }
h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    text-align: center;
    background: linear-gradient(135deg, #e8f0fe 30%, #38bdf8 70%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}
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
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; background-color: #e2e8f0;
    margin: 0 2px; animation: typing 1.4s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes typing {
    0%, 60%, 100% { opacity: 0.3; }
    30% { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)


#  Session State 
for key, default in [
    ("active_sop", None),
    ("last_docs", []),
    ("pending_image", None),
    ("messages", []),
    ("pending_followup", None),
    ("waiting_for_response", False),
    ("quick_question", None),
    ("asked_questions", []),
    ("is_admin", False),
    ("show_admin_login", False),
    ("conversation_context", []),
    ("not_found_count", 0),
    ("asked_followups", set()),
    ("pending_question", None),  # holds question to process after rerun
]:
    if key not in st.session_state:
        st.session_state[key] = default


#  Topic change detection 

def is_topic_change(question: str) -> bool:
    return False


#  Helpers 
def track_question(q: str):
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


def normalize_text(text):
    return unicodedata.normalize("NFKD", text)


def clean_main_answer(answer: str):
    # STEP 1: Normalize unicode (VERY IMPORTANT)
    answer = normalize_text(answer)

    # STEP 2: Replace weird bullets
    bullet_map = {
        '': '•',
        '': '•',
        '': '•',
        '': '•',
        '■': '•',
        '▪': '•',
        '◆': '•',
        '►': '•'
    }

    for k, v in bullet_map.items():
        answer = answer.replace(k, v)

    lines = answer.strip().split('\n')
    cleaned_lines = []

    for line in lines:
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()


def clean_sources_html(raw_html: str) -> str:
    import re as _re
    texts = _re.findall(r'>([^<]+)<', raw_html)
    hrefs = _re.findall(r'href=["\']([^"\']+)["\']', raw_html)
    pieces = [t.strip() for t in texts if t.strip()]
    if not pieces:
        return raw_html
    parts_html = []
    href_idx = 0
    date_html = None
    for p in pieces:
        if _re.search(r'\d{2}\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}', p):
            date_html = f'<span style="font-size:0.82em;color:#64748b;font-style:italic;">{p}</span>'
            continue
        if href_idx < len(hrefs) and ('open' in p.lower() or ('sop' in p.lower() and len(p) < 20)):
            parts_html.append(f'<a href="{hrefs[href_idx]}" target="_blank">{p}</a>')
            href_idx += 1
        else:
            parts_html.append(f'<span>{p}</span>')
    result = ' · '.join(parts_html)
    if date_html:
        result += f'<br>{date_html}'
    return result


def format_answer_html(text: str) -> str:
    import html as _html
    import re as _re
    lines = text.split('\n')
    output = []
    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            output.append('</ul>')
            in_ul = False
        if in_ol:
            output.append('</ol>')
            in_ol = False

    def render_inline(s: str) -> str:
        s = _html.escape(s)
        s = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
        s = _re.sub(r'\*(.+?)\*', r'<em>\1</em>', s)
        return s

    for line in lines:
        stripped = line.strip()
        if not stripped:
            close_lists()
            continue
        if _re.match(r'^[•\-\*–■▪◆►]\s+', stripped):
            if in_ol:
                output.append('</ol>')
                in_ol = False
            if not in_ul:
                output.append('<ul>')
                in_ul = True
            content = _re.sub(r'^[•\-\*–■▪◆►]\s+', '', stripped)
            output.append(f'<li>{render_inline(content)}</li>')
        elif _re.match(r'^\d+[\.\)]\s+', stripped):
            if in_ul:
                output.append('</ul>')
                in_ul = False
            if not in_ol:
                output.append('<ol>')
                in_ol = True
            content = _re.sub(r'^\d+[\.\)]\s+', '', stripped)
            output.append(f'<li>{render_inline(content)}</li>')
        elif len(stripped) < 80 and stripped.endswith(':') and '.' not in stripped[:-1]:
            close_lists()
            output.append(f'<p><strong>{render_inline(stripped)}</strong></p>')
        else:
            close_lists()
            output.append(f'<p>{render_inline(stripped)}</p>')
    close_lists()
    return '\n'.join(output)


def build_chat_text():
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
        elif role in ("bot", "bot_with_sources"):
            answer_text = msg if isinstance(msg, str) else msg.get('answer', '')
            lines.append(f"BOT:  {answer_text}\n")
        elif role == "followup":
            lines.append(f"BOT (follow-up):  {msg}\n")
    lines.append("=" * 50)
    return "\n".join(lines)


def get_last_exchange() -> str:
    if not st.session_state.conversation_context:
        return "No previous conversation."
    last_user, last_bot = st.session_state.conversation_context[-1]
    return f"User: {last_user}\nAssistant: {last_bot}"


def get_full_history() -> str:
    ctx = st.session_state.conversation_context[-3:]
    if not ctx:
        return "No previous conversation."
    lines = []
    for user_q, bot_a in ctx:
        lines.append(f"User: {user_q}")
        lines.append(f"Assistant: {bot_a}")
    return "\n".join(lines)


def update_context(user_q: str, bot_answer: str):
    st.session_state.conversation_context.append((user_q, bot_answer))
    st.session_state.conversation_context = st.session_state.conversation_context[-6:]


def detect_intent(user_input: str):
    lowered = user_input.strip().lower()
    yes_words = sorted(["yes", "yeah", "yep", "yup", "ok", "okay", "sure", "continue", "please", "y"], key=len, reverse=True)
    no_words  = sorted(["nope", "no thanks", "not now", "no", "nah", "skip", "n"], key=len, reverse=True)
    connectors = r'^(and also|also|and then|and|but also|but|please also|please|tell me also|also tell me|tell me)\s*'

    for word in yes_words:
        if lowered == word:
            return ("yes", "")
        if lowered.startswith(word + " ") or lowered.startswith(word + ","):
            extra = re.sub(connectors, '', lowered[len(word):].strip().lstrip(",").strip(), flags=re.IGNORECASE).strip()
            return ("yes", extra) if extra else ("yes", "")

    for word in no_words:
        if lowered == word:
            return ("no", "")
        if lowered.startswith(word + " ") or lowered.startswith(word + ","):
            extra = re.sub(connectors, '', lowered[len(word):].strip().lstrip(",").strip(), flags=re.IGNORECASE).strip()
            return ("no", extra) if extra else ("no", "")

    return ("new_question", "")


def is_image_relevant(question: str, image_path: str) -> bool:
    question_lower = question.lower()
    image_name = re.sub(r'\.(png|jpg|jpeg|gif)$', '', os.path.basename(image_path).lower())
    image_keywords = re.split(r'[_\-\s\.]+', image_name)
    return sum(1 for kw in image_keywords if len(kw) > 2 and kw in question_lower) >= 1


def is_conversational(question: str):
    q = question.strip().lower()
    gibberish = {"k", "kk", "hmm", "hm", "lol", "lmao", "haha", "hehe",
                 "ohh", "ohk", "ohkay", "ooh", "umm", "uh", "err", "wtf", "omg"}
    greetings = {"hi", "hello", "hey", "hii", "helo", "howdy"}
    closings  = {"bye", "goodbye", "see you", "cya", "thanks", "thank you", "thx", "ty"}
    ack       = {"ok", "okay", "cool", "great", "got it", "noted", "alright"}

    if q in gibberish or (len(q) == 1 and q not in {"y", "n"}):
        return "I didn't quite get that. Could you ask a specific SOP question?\n\nFor example: *What is the dress code?* or *How do I apply for leave?*"
    if q in greetings:
        return "Hello! I'm your SOP assistant. Ask me anything about company policies, procedures, or workflows."
    if q in closings:
        return "Goodbye! Feel free to return anytime you have SOP-related questions."
    if q in ack:
        return "Glad to help! Feel free to ask anything else about the SOPs."
    return None


def rebuild_vectorstore():
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
    docs   = load_pdfs()
    chunks = split_docs(docs)
    return create_vectorstore(chunks)


def answer_from_docs(section_key: str, docs: list) -> str:
    # Fallback: search docs directly (no LLM)
    if not docs:
        st.session_state.not_found_count += 1

        if st.session_state.not_found_count >= 2:
            st.session_state.active_sop = None
            st.session_state.not_found_count = 0

        return "This information is not available in the provided SOP.", [], None
    else:
        st.session_state.not_found_count = 0
    full_context = "\n\n".join(d.page_content for d in docs)
    idx = full_context.lower().find(section_key.lower())
    if idx == -1:
        return "This information is not available in the provided SOP."

    snippet = full_context[idx: idx + 800].strip()
    if len(snippet) < 30:
        return "This information is not available in the provided SOP."

    return snippet


def get_k_for_question(question: str):
    q = question.lower()
    if any(word in q for word in [
        "all", "list", "complete", "responsibilities", "steps",
        "process", "guidelines", "sections", "sub", "detail",
        "full", "entire", "everything", "what are", "types"
    ]):
        return 8
    return 4


@st.cache_resource
def setup_system():
    vectorstore = load_existing_vectorstore()
    if vectorstore is None:
        vectorstore = rebuild_vectorstore()

    retriever = get_retriever(vectorstore)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    query_rewrite_prompt = ChatPromptTemplate.from_template(
        """You rewrite a user question into a standalone search query for retrieving relevant SOP passages.

RULES:
- If the current question depends on prior context, incorporate the needed details from the last exchange.
- If it is already standalone, return it unchanged.
- Output ONLY the rewritten search query (no quotes, no extra text).

Last exchange:
{last_exchange}

User question:
{question}"""
    )

    answer_prompt = ChatPromptTemplate.from_template("""

You are a STRICT internal SOP assistant.
    RULES:
    - Answer ONLY from CONTEXT.
    - Do NOT use outside knowledge.
    - Do NOT assume policies.
    - If answer not present → say EXACTLY:
    "This information is not available in the provided SOP."
    - If answer has multiple sections or subsections → include ALL of them completely
    - NEVER truncate or summarize a long answer
    - NEVER say "and more" or "etc." — always give the complete answer
    FOLLOW UP RULES:
    - Suggest ONLY ONE next question from SAME CONTEXT.
    - Do NOT invent topics.
    - Do NOT ask generic questions.
- If no follow-up possible → write FOLLOW_UP: NONE

OUTPUT FORMAT:
ANSWER:
<full answer>

FOLLOW_UP:
<question OR NONE>

CONTEXT:
{context}

QUESTION:
{question}

""")

    query_rewrite_chain = query_rewrite_prompt | llm | StrOutputParser()

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'SOP')}]\n{doc.page_content}"
            for doc in docs
        )

    def rag_with_history(question: str):
        full_history = get_full_history()

        if st.session_state.active_sop:
            last_exchange = get_last_exchange()
            search_query = query_rewrite_chain.invoke({
                "last_exchange": last_exchange,
                "question":      question,
            }).strip()
            if not search_query or len(search_query) < 4:
                search_query = question

            k = get_k_for_question(question)
            locked_docs = [
                d for d in retriever.vectorstore.similarity_search(search_query, k=k)
                if d.metadata.get("source") == st.session_state.active_sop
            ]

            if locked_docs:
                context = format_docs(locked_docs)

                temp_filled = answer_prompt.invoke({
                    "history": full_history,
                    "context": context,
                    "question": question,
                })

                temp_raw = llm.invoke(temp_filled).content
                temp_answer = temp_raw.split("FOLLOW_UP:")[0].replace("ANSWER:", "").strip()

                if "not available" not in temp_answer.lower():
                    docs = locked_docs
                else:
                    st.session_state.active_sop = None
                    docs = retriever.vectorstore.similarity_search(search_query, k=get_k_for_question(question))
            else:
                st.session_state.active_sop = None

                docs = retriever.vectorstore.similarity_search(search_query, k=get_k_for_question(question))

                if not docs:
                    return "This information is not available in the provided SOP.", [], None

                from collections import Counter
                sources = [d.metadata.get("source") for d in docs if d.metadata.get("source")]

                if sources:
                    best_source = Counter(sources).most_common(1)[0][0]
                    docs = [d for d in docs if d.metadata.get("source") == best_source]
                    st.session_state.active_sop = best_source
                    st.session_state.asked_followups = set()
            if st.session_state.active_sop is None and docs:
                from collections import Counter
                sources = [d.metadata.get("source") for d in docs if d.metadata.get("source")]

                if sources:
                    best_source = Counter(sources).most_common(1)[0][0]
                    docs = [d for d in docs if d.metadata.get("source") == best_source]
                    st.session_state.active_sop = best_source
        # SOP not locked yet → detect best SOP and lock
        else:
            if is_topic_change(question):
                search_query = question
            else:
                last_exchange = get_last_exchange()
                search_query = query_rewrite_chain.invoke({
                    "last_exchange": last_exchange,
                    "question":      question,
                }).strip()
                if not search_query or len(search_query) < 4:
                    search_query = question

            docs = retriever.vectorstore.similarity_search(search_query, k=get_k_for_question(question))

            from collections import Counter
            sources = [d.metadata.get("source") for d in docs if d.metadata.get("source")]
            if sources:
                best_source = Counter(sources).most_common(1)[0][0]
                docs = [d for d in docs if d.metadata.get("source") == best_source]
                st.session_state.active_sop = best_source
                st.session_state.asked_followups = set()

        #  HARD GUARD
        if not docs:
            return "This information is not available in the provided SOP.", [], None

        context = format_docs(docs)

        filled = answer_prompt.invoke({
            "history":  full_history,
            "context":  context,
            "question": question,
        })
        raw = llm.invoke(filled).content

        followup = None

        if "FOLLOW_UP:" in raw:
            parts = raw.split("FOLLOW_UP:")
            answer = parts[0].replace("ANSWER:", "").strip()
            followup = parts[1].strip()

            if followup.upper() == "NONE":
                followup = None
        else:
            answer = raw.strip()

        #  Dynamic Grounding Guard (FINAL)
        if "not available" not in answer.lower():
            answer_words = set(answer.lower().split())
            context_words = set(context.lower().split())

            overlap = len(answer_words.intersection(context_words))

            if overlap < max(2, int(len(answer_words) * 0.2)):
                answer = "This information is not available in the provided SOP."
                docs = []

        return answer, docs ,followup

    return rag_with_history, retriever


#  Instantiate 
qa_chain, retriever = setup_system()


#  Sidebar 
with st.sidebar:
    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages             = []
            st.session_state.conversation_context = []
            st.session_state.pending_followup     = None
            st.session_state.waiting_for_response = False
            st.session_state.asked_questions      = []
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
                st.session_state.is_admin         = True
                st.session_state.show_admin_login = False
                st.success("Admin Access Granted")
                st.rerun()
    else:
        st.success("Logged in as Admin")
        if st.button("🔄 Sync SOPs", use_container_width=True):
            with st.spinner("Syncing SOPs and rebuilding index..."):
                syncer = SOPAutoSync(
                    base_url="https://upaygoa.com/geltm/helpndoc",
                    download_dir="./sop_documents"
                )
                new_files, updated_files = syncer.sync()
                if new_files > 0 or updated_files > 0:
                    rebuild_vectorstore()
                    setup_system.clear()
                    st.success("SOPs synced and index rebuilt successfully!")
                else:
                    st.info("No new or updated SOPs found.")

        if st.button("🔁 Rebuild Index", use_container_width=True):
            with st.spinner("Rebuilding vector index..."):
                rebuild_vectorstore()
                setup_system.clear()
                st.success("Index rebuilt! Please refresh the page.")

        if st.button("Logout Admin"):
            st.session_state.is_admin = False
            st.rerun()


#  Main Chat UI 
st.title("Prakriya AI 🤖")
st.markdown("<div style='text-align:center; font-family:Syne,sans-serif; font-size:0.72rem; font-weight:500; letter-spacing:0.22em; text-transform:uppercase; color:#f59e0b; margin-top:-0.5rem; margin-bottom:1rem;'>GEL SOP Chatbot</div>", unsafe_allow_html=True)

for role, msg in st.session_state.messages:
    if role == "user":
        st.markdown(f"<div class='user-msg'>{msg}</div>", unsafe_allow_html=True)
    elif role == "bot":
        if msg.startswith("[IMAGE]"):
            st.image(msg.replace("[IMAGE]", ""))
        else:
            st.markdown(f"<div class='bot-msg'>{msg}</div>", unsafe_allow_html=True)
    elif role == "bot_with_sources":
        st.markdown(f"<div class='bot-msg'>{msg['answer']}</div>", unsafe_allow_html=True)
        if msg.get('sources_html'):
             st.container().markdown(
                 msg['sources_html'].strip(),
                 unsafe_allow_html=True)
typing_placeholder = st.empty()

with st.form("chat_form", clear_on_submit=True):
    question = st.text_input("Ask about SOP...", placeholder="Type your question here...")
    send     = st.form_submit_button("Send")


# Shared helper
def handle_extra(eq: str):

    if not eq:
        return

    typing_placeholder.markdown(show_typing_indicator(), unsafe_allow_html=True)

    ans, docs, followup = qa_chain(eq)
    st.session_state.last_docs = docs

    typing_placeholder.empty()

    main_answer = clean_main_answer(ans)

    negative = (
        "not available" in main_answer.lower()
        or "not covered" in main_answer.lower()
        or "could not find" in main_answer.lower()
    )

    if negative or not docs:
        st.session_state.messages.append(("bot", main_answer))
        st.session_state.pending_followup = None
        st.session_state.waiting_for_response = False
        return

    raw_sources = metadata_handler.format_sources_html(docs)
    sources_html = clean_sources_html(raw_sources) if docs else ""
    combined_answer = format_answer_html(main_answer)

    st.session_state.messages.append(
        ("bot_with_sources", {
            "answer": combined_answer,
            "sources_html": sources_html
        })
    )

    if followup:
        st.session_state.messages.append(("followup", followup))
        st.session_state.pending_followup = followup
        st.session_state.waiting_for_response = True
    else:
        st.session_state.pending_followup = None
        st.session_state.waiting_for_response = False

#  FAQ Handler 
if st.session_state.quick_question:
    question = st.session_state.quick_question
    st.session_state.quick_question = None

    track_question(question)
    st.session_state.messages.append(("user", question))
    typing_placeholder.markdown(show_typing_indicator(), unsafe_allow_html=True)

    answer, docs ,followup= qa_chain(question)
    update_context(question, answer)
    st.session_state.last_docs = docs
    
    image_doc = next(
        (d for d in docs if d.metadata.get("type") == "image"
         and is_image_relevant(question, d.metadata["path"])), None
    )
    if image_doc:
        typing_placeholder.empty()
        st.session_state.messages.append(("bot", f"[IMAGE]{os.path.abspath(image_doc.metadata['path'])}"))
        st.session_state.messages.append(("followup", "Does this answer your question? Feel free to ask anything else!"))
        st.session_state.pending_followup     = "image_shown"
        st.session_state.waiting_for_response = True
        st.rerun()

    typing_placeholder.empty()

    main_answer = clean_main_answer(answer)

    negative = (
    "not available" in main_answer.lower()
    or "not covered" in main_answer.lower()
    or "could not find" in main_answer.lower()
    or not docs
    )

    if negative:
        st.session_state.messages.append(("bot", main_answer))
        update_context(question, main_answer)
    else:
        raw_sources = metadata_handler.format_sources_html(docs)
        sources_html = clean_sources_html(raw_sources) if docs else ""
        combined_answer = format_answer_html(main_answer)
        st.session_state.messages.append(("bot_with_sources", {
            'answer': combined_answer,
            'sources_html': sources_html
        }))
        update_context(question, main_answer)

    st.rerun()


#  Main Input Handler 
if send and question.strip():
    user_input = question.strip().lower()
    intent, extra_question = detect_intent(user_input)

    # CASE 1: yes/no 
    if st.session_state.waiting_for_response and intent in ("yes", "no"):
        st.session_state.messages.append(("user", question))

        #  NEW: handle "no" immediately before any other checks 
        if intent == "no":
            st.session_state.messages.append(("bot", "No problem! Feel free to ask anything else."))
            st.session_state.pending_followup = None
            st.session_state.waiting_for_response = False
            update_context(question, "")
            if extra_question:
                handle_extra(extra_question)
            st.rerun()
        #  END NEW 

        if st.session_state.pending_followup == "show_image":
            if intent == "yes":
                abs_path = os.path.abspath(st.session_state.pending_image)
                st.session_state.messages.append(("bot", f"[IMAGE]{abs_path}"))
                st.session_state.messages.append(("followup", "Hope that helps! Do you have any other questions?"))
                st.session_state.pending_followup     = "image_shown"
                st.session_state.waiting_for_response = True
            else:
                st.session_state.messages.append(("bot", "No problem! Feel free to ask anything else."))
                st.session_state.pending_followup     = None
                st.session_state.waiting_for_response = False
            st.session_state.pending_image = None
            handle_extra(extra_question)
            st.rerun()

        elif st.session_state.pending_followup == "image_shown":
            if intent == "yes" and extra_question:
                handle_extra(extra_question)
            elif intent == "yes":
                st.session_state.messages.append(("bot", "Sure! What specific details would you like to know? Feel free to ask anything."))
                st.session_state.pending_followup     = None
                st.session_state.waiting_for_response = False
            else:
                st.session_state.messages.append(("bot", "No problem! Feel free to ask anything else."))
                st.session_state.pending_followup     = None
                st.session_state.waiting_for_response = False
                handle_extra(extra_question)
            st.rerun()

        else:
            if intent == "yes":

                if not st.session_state.pending_followup:
                    st.session_state.messages.append(("bot", "Please ask a question."))
                    st.rerun()

                question = st.session_state.pending_followup

                typing_placeholder.markdown(show_typing_indicator(), unsafe_allow_html=True)

                answer, docs, followup = qa_chain(question)

                typing_placeholder.empty()

                st.session_state.messages.append(("bot", answer))

                if followup and followup not in st.session_state.asked_followups:
                    st.session_state.asked_followups.add(followup)

                    st.session_state.messages.append(("followup", followup))
                    st.session_state.pending_followup = followup
                    st.session_state.waiting_for_response = True
                else:
                    st.session_state.pending_followup = None
                    st.session_state.waiting_for_response = False

                st.rerun()

    #  CASE 2: New question — PASS 1: show user bubble immediately
    else:
        st.session_state.pending_followup     = None
        st.session_state.waiting_for_response = False
        st.session_state.pending_image        = None
        if is_topic_change(question):
            st.session_state.active_sop = None
        st.session_state.messages.append(("user", question))
        track_question(question)
        st.session_state.pending_question = question
        st.rerun()
        
        
# PASS 2: user bubble is visible, now show typing -> answer
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

    typing_placeholder.markdown(show_typing_indicator(), unsafe_allow_html=True)

    conv_response = is_conversational(question)
    if conv_response:
        typing_placeholder.empty()
        st.session_state.messages.append(("bot", conv_response))
        st.rerun()

    answer, docs, followup = qa_chain(question)
    update_context(question, answer)
    st.session_state.last_docs = docs
    typing_placeholder.empty()

    image_doc = next(
        (d for d in docs if d.metadata.get("type") == "image"
         and is_image_relevant(question, d.metadata["path"])), None
    )
    if image_doc:
        st.session_state.messages.append(("bot", f"[IMAGE]{os.path.abspath(image_doc.metadata['path'])}"))
        st.session_state.messages.append(("followup", "Does this answer your question? Feel free to ask anything else!"))
        st.session_state.pending_followup = "image_shown"
        st.session_state.waiting_for_response = True
        st.rerun()

    if "IRRELEVANT_QUESTION" in answer:
        st.session_state.messages.append(("bot",
            "Sorry, that seems unrelated to our SOPs. I'm here to help with company policies!\n\n"
            "You can ask me about:\n"
            "• **Dress Code** — formal vs casual attire rules\n"
            "• **Leave Policy** — types of leave and procedures\n"
            "• **Hierarchy** — organisational structure\n"
            "• **Jira Workflow** — project and task management\n"
            "• **Social Media Policy** — content approval process\n"
            "• **IT & Security** — access control, data backup\n\n"
            "Just type your question!"
        ))
        st.rerun()

    main_answer = clean_main_answer(answer)
    sop_docs = [d for d in docs if d.metadata.get("source") == st.session_state.active_sop]

    negative = (
        "not available" in main_answer.lower()
        or "not covered" in main_answer.lower()
        or "could not find" in main_answer.lower()
        or not docs
    )

    if negative:
        st.session_state.messages.append(("bot", main_answer))
        update_context(question, main_answer)
    else:
        raw_sources = metadata_handler.format_sources_html(docs)
        sources_html = clean_sources_html(raw_sources) if docs else ""
        combined_answer = format_answer_html(main_answer)
        if followup:
            combined_answer += f"<div class='followup-inline'>💡 {followup}</div>"
        st.session_state.messages.append(("bot_with_sources", {
            'answer': combined_answer,
            'sources_html': sources_html
        }))
        update_context(question, main_answer)

    related_image = next(
        (d for d in docs if d.metadata.get("type") == "image"
         and is_image_relevant(question, d.metadata["path"])), None
    )

    if related_image:
        st.session_state.pending_image = related_image.metadata["path"]
        st.session_state.messages.append(("followup", "There is a related flowchart available. Would you like to see it?"))
        st.session_state.pending_followup = "show_image"
        st.session_state.waiting_for_response = True

    elif followup:
        if followup not in st.session_state.asked_followups:
            st.session_state.asked_followups.add(followup)
            st.session_state.pending_followup = followup
            st.session_state.waiting_for_response = True
        else:
            st.session_state.pending_followup = None
            st.session_state.waiting_for_response = False

    else:
        st.session_state.pending_followup = None
        st.session_state.waiting_for_response = False

    st.rerun()
