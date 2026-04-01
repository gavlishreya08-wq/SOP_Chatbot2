import streamlit as st
from dotenv import load_dotenv
import os
import re
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

st.set_page_config(page_title="Prakriya AI", page_icon="🤖", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root Variables ── */
:root {
    --bg-base:        #060a12;
    --bg-panel:       #0c1220;
    --bg-surface:     #111827;
    --bg-glass:       rgba(17, 24, 39, 0.7);
    --border-subtle:  rgba(99, 179, 237, 0.08);
    --border-glow:    rgba(99, 179, 237, 0.22);
    --accent-primary: #38bdf8;
    --accent-gold:    #f59e0b;
    --accent-glow:    rgba(56, 189, 248, 0.15);
    --text-primary:   #e8f0fe;
    --text-secondary: #94a3b8;
    --text-muted:     #475569;
    --user-bubble:    linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
    --bot-bubble:     #0f1c2e;
    --radius-lg:      16px;
    --radius-md:      10px;
    --radius-sm:      6px;
    --shadow-glow:    0 0 40px rgba(56, 189, 248, 0.08);
    --shadow-card:    0 4px 24px rgba(0,0,0,0.4);
}

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg-base) !important;
    color: var(--text-primary);
}

/* ── Animated mesh background ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(56,189,248,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 60% at 80% 90%, rgba(37,99,235,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 55% 50%, rgba(245,158,11,0.03) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* ── Layout container ── */
.main > div {
    max-width: 860px;
    margin: 0 auto;
    padding-bottom: 2rem;
    position: relative;
    z-index: 1;
}

/* ── Header / Title ── */
h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.4rem !important;
    text-align: center !important;
    letter-spacing: -0.03em !important;
    background: linear-gradient(135deg, #e8f0fe 30%, #38bdf8 70%, #f59e0b 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin-bottom: 0.15rem !important;
    line-height: 1.1 !important;
}

/* Subtitle line injected via markdown below the title */
.gel-subtitle {
    text-align: center;
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--accent-gold);
    opacity: 0.85;
    margin-top: -0.2rem;
    margin-bottom: 1.6rem;
}

/* ── Chat bubbles — USER ── */
.user-msg {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 400;
    background: var(--user-bubble);
    color: #fff;
    padding: 13px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 10px 0 10px auto;
    width: fit-content;
    max-width: 68%;
    line-height: 1.55;
    box-shadow: 0 2px 16px rgba(14,165,233,0.25), 0 1px 4px rgba(0,0,0,0.3);
    animation: slideInRight 0.22s ease-out;
}

/* ── Chat bubbles — BOT ── */
.bot-msg {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 300;
    background: var(--bot-bubble);
    color: var(--text-primary);
    padding: 14px 18px;
    border-radius: 4px 18px 18px 18px;
    margin: 10px auto 10px 0;
    width: fit-content;
    max-width: 72%;
    line-height: 1.6;
    border: 1px solid var(--border-subtle);
    box-shadow: var(--shadow-card), inset 0 1px 0 rgba(255,255,255,0.03);
    animation: slideInLeft 0.22s ease-out;
    position: relative;
}

.bot-msg::before {
    content: '';
    position: absolute;
    left: -1px; top: 0; bottom: 0;
    width: 3px;
    border-radius: 4px 0 0 4px;
    background: linear-gradient(180deg, var(--accent-primary), transparent);
    opacity: 0.6;
}

/* ── Follow-up suggestion ── */
.followup-box {
    margin: 6px 0 0 0;
    padding: 8px 14px;
    border-radius: var(--radius-md);
    background: rgba(56, 189, 248, 0.06);
    border: 1px solid rgba(56, 189, 248, 0.15);
    color: #7dd3fc;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85em;
    font-style: italic;
    max-width: 70%;
    animation: fadeIn 0.3s ease-out;
}

.followup-inline {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid var(--border-subtle);
    font-size: 0.82em;
    color: #7dd3fc;
    font-style: italic;
    opacity: 0.9;
    display: block;
}

/* ── Typing indicator ── */
.typing-indicator {
    background: var(--bot-bubble);
    border: 1px solid var(--border-subtle);
    color: var(--text-primary);
    padding: 14px 20px;
    border-radius: 4px 18px 18px 18px;
    margin: 10px 0;
    width: fit-content;
    max-width: 70%;
    display: flex;
    align-items: center;
    gap: 4px;
}

.typing-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent-primary);
    opacity: 0.4;
    animation: typingPulse 1.4s infinite ease-in-out;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes typingPulse {
    0%, 60%, 100% { opacity: 0.2; transform: scale(0.85); }
    30% { opacity: 1; transform: scale(1.1); }
}

/* ── SOP source metadata strip ── */
.sop-compact {
    margin-top: 10px;
    max-width: 72%;
    background: transparent !important;
    border: none !important;
    padding: 2px 0 !important;
    box-shadow: none !important;
    display: block;
    line-height: 1.8;
}

/* Force the date (last span) onto its own line */
.sop-compact span:last-child {
    display: block !important;
    margin-top: 1px !important;
}

.sop-compact span {
    font-size: 0.82em !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 400 !important;
    color: #94a3b8 !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* SOP name — first span, brighter */
.sop-compact span:first-child {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
}

.meta-dot {
    color: #475569 !important;
    font-size: 0.75em !important;
    vertical-align: middle !important;
    margin: 0 3px !important;
}

.meta-date {
    font-size: 0.82em !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #64748b !important;
    font-style: italic !important;
}

/* Open SOP link */
.sop-compact a {
    font-size: 0.82em !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: #38bdf8 !important;
    text-decoration: none !important;
    border-bottom: 1px solid rgba(56, 189, 248, 0.3) !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
    transition: color 0.15s, border-color 0.15s !important;
}
.sop-compact a:hover {
    color: #bae6fd !important;
    border-bottom-color: rgba(56, 189, 248, 0.7) !important;
}

/* ── Formatted answer content inside bot bubble ── */
.bot-msg p {
    margin: 0 0 6px 0 !important;
    line-height: 1.6 !important;
}
.bot-msg p:last-of-type {
    margin-bottom: 0 !important;
}
.bot-msg ul, .bot-msg ol {
    margin: 4px 0 8px 0 !important;
    padding-left: 20px !important;
}
.bot-msg li {
    margin: 4px 0 !important;
    line-height: 1.55 !important;
    color: #e2e8f0 !important;
}
.bot-msg ul li {
    list-style-type: disc !important;
}
.bot-msg ol li {
    list-style-type: decimal !important;
}
.bot-msg li::marker {
    color: #38bdf8 !important;
}
.bot-msg strong {
    font-weight: 600 !important;
    color: #f1f5f9 !important;
}
.bot-msg em {
    font-style: italic !important;
    color: #cbd5e1 !important;
}
.ans-heading {
    font-weight: 600 !important;
    color: #7dd3fc !important;
    margin: 8px 0 4px 0 !important;
    font-size: 0.92em !important;
    letter-spacing: 0.01em !important;
}
section[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
section[data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif !important;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
}
section[data-testid="stSidebar"] p {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}

/* Sidebar divider */
section[data-testid="stSidebar"] hr {
    border-color: var(--border-subtle) !important;
    margin: 1rem 0 !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    color: var(--accent-primary) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: var(--radius-sm) !important;
    padding: 7px 14px !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
.stButton > button:hover {
    background: var(--accent-glow) !important;
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 16px rgba(56,189,248,0.15) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}
.stButton > button:disabled {
    opacity: 0.3 !important;
    cursor: not-allowed !important;
}

/* FAQ buttons in sidebar */
section[data-testid="stSidebar"] .stButton > button {
    text-transform: none !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 400 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0 !important;
    color: var(--text-secondary) !important;
    border-color: var(--border-subtle) !important;
    text-align: left !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    color: var(--text-primary) !important;
    border-color: var(--border-glow) !important;
}

/* ── Form / Input ── */
.stForm {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: var(--radius-lg) !important;
    padding: 14px 18px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), var(--shadow-glow) !important;
    margin-top: 1.2rem !important;
}

.stTextInput > div > div > input {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 300 !important;
    background: rgba(6,10,18,0.7) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    caret-color: var(--accent-primary) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.12) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder {
    color: var(--text-muted) !important;
    font-style: italic !important;
}

/* Send button — accent filled */
.stForm .stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #2563eb) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 10px 22px !important;
    box-shadow: 0 2px 14px rgba(14,165,233,0.35) !important;
}
.stForm .stButton > button:hover {
    box-shadow: 0 4px 20px rgba(14,165,233,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Success / Info / Warning alerts ── */
.stSuccess, .stInfo {
    background: rgba(56,189,248,0.07) !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-color: var(--accent-primary) transparent transparent !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-glow); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-primary); }

/* ── Animations ── */
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(18px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-18px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Password input ── */
.stTextInput [type="password"] {
    font-family: 'DM Sans', sans-serif !important;
    background: rgba(6,10,18,0.9) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
}

/* ── Hide Streamlit branding ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Download button ── */
.stDownloadButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    color: var(--accent-gold) !important;
    border: 1px solid rgba(245,158,11,0.3) !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    background: rgba(245,158,11,0.08) !important;
    border-color: var(--accent-gold) !important;
    box-shadow: 0 0 16px rgba(245,158,11,0.15) !important;
}

/* ── Admin section ── */
section[data-testid="stSidebar"] .stSubheader {
    font-family: 'Syne', sans-serif !important;
    color: var(--accent-gold) !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.08em !important;
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


def clean_main_answer(answer: str):
    lines = answer.strip().split('\n')
    cleaned_lines = []
    for i, line in enumerate(lines):
        if i > 0 and line.strip().endswith('?') and len(line.strip()) < 150:
            if (i == len(lines) - 1) or (i > 0 and not lines[i - 1].strip()):
                break
        cleaned_lines.append(line)
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    return '\n'.join(cleaned_lines).strip() or answer


def format_answer_html(text: str) -> str:
    """Convert plain-text answer (with bullets, numbers, bold) into clean HTML."""
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
        # Escape HTML first, then restore bold/italic markdown
        s = _html.escape(s)
        # **bold**
        s = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
        # *italic*
        s = _re.sub(r'\*(.+?)\*', r'<em>\1</em>', s)
        return s

    for line in lines:
        stripped = line.strip()
        if not stripped:
            close_lists()
            continue

        # Bullet: lines starting with •, -, *, –
        if _re.match(r'^[•\-\*–]\s+', stripped):
            if in_ol:
                output.append('</ol>')
                in_ol = False
            if not in_ul:
                output.append('<ul>')
                in_ul = True
            content = _re.sub(r'^[•\-\*–]\s+', '', stripped)
            output.append(f'<li>{render_inline(content)}</li>')

        # Numbered list: 1. or 1)
        elif _re.match(r'^\d+[\.\)]\s+', stripped):
            if in_ul:
                output.append('</ul>')
                in_ul = False
            if not in_ol:
                output.append('<ol>')
                in_ol = True
            content = _re.sub(r'^\d+[\.\)]\s+', '', stripped)
            output.append(f'<li>{render_inline(content)}</li>')

        # Heading-like lines (short, end with colon, no period inside)
        elif len(stripped) < 80 and stripped.endswith(':') and '.' not in stripped[:-1]:
            close_lists()
            output.append(f'<p class="ans-heading">{render_inline(stripped)}</p>')

        # Normal paragraph line
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


def clean_sources_html(raw_html: str) -> str:
    """Strip the raw block-level HTML from metadata_handler and rebuild as
    a clean two-line strip: name · Open SOP · version  /  date on next line"""
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
        # Detect date pieces (contain digits that look like a date)
        if _re.search(r'\d{2}\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}', p):
            date_html = f'<span class="meta-date">{p}</span>'
            continue
        if href_idx < len(hrefs) and ('open' in p.lower() or ('sop' in p.lower() and len(p) < 20)):
            parts_html.append(f'<a href="{hrefs[href_idx]}" target="_blank">{p}</a>')
            href_idx += 1
        else:
            parts_html.append(f'<span>{p}</span>')
    result = ' <span class="meta-dot">·</span> '.join(parts_html)
    if date_html:
        result += f'<br>{date_html}'
    return result


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

        return "This information is not defined in the GEL SOP standards. You may check related SOP documents or reach out to your team lead for further guidance..", [], None
    else:
        st.session_state.not_found_count = 0
    full_context = "\n\n".join(d.page_content for d in docs)
    idx = full_context.lower().find(section_key.lower())
    if idx == -1:
        return "This information is not defined in the GEL SOP standards. You may check related SOP documents or reach out to your team lead for further guidance.."

    snippet = full_context[idx: idx + 800].strip()
    if len(snippet) < 30:
        return "This information is not defined in the GEL SOP standards. You may check related SOP documents or reach out to your team lead for further guidance.."

    return snippet


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
"This information is not defined in the GEL SOP standards. You may check related SOP documents or reach out to your team lead for further guidance.."
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

        # If SOP is locked, ALWAYS search within it first regardless of topic change
        if st.session_state.active_sop:
            last_exchange = get_last_exchange()
            search_query = query_rewrite_chain.invoke({
                "last_exchange": last_exchange,
                "question":      question,
            }).strip()
            if not search_query or len(search_query) < 4:
                search_query = question

            # STEP 1 — search inside locked SOP
            locked_docs = [
                d for d in retriever.invoke(search_query)
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
                    docs = retriever.invoke(search_query)
            else:
                # STEP 2 — unlock and search globally
                st.session_state.active_sop = None

                docs = retriever.invoke(search_query)

                if not docs:
                    return "This information is not defined in the GEL SOP standards. You may check related SOP documents or reach out to your team lead for further guidance..", [], None

                # STEP 3 — lock new SOP
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

            docs = retriever.invoke(search_query)

            from collections import Counter
            sources = [d.metadata.get("source") for d in docs if d.metadata.get("source")]
            if sources:
                best_source = Counter(sources).most_common(1)[0][0]
                docs = [d for d in docs if d.metadata.get("source") == best_source]
                st.session_state.active_sop = best_source
                st.session_state.asked_followups = set()

        #  HARD GUARD
        if not docs:
            return "This information is not defined in the GEL SOP standards. You may check related SOP documents or reach out to your team lead for further guidance..", [], None

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
                answer = "This information is not defined in the GEL SOP standards. You may check related SOP documents or reach out to your team lead for further guidance.."
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
st.markdown("<div class='gel-subtitle'>GEL SOP Chatbot</div>", unsafe_allow_html=True)

for role, msg in st.session_state.messages:
    if role == "user":
        import html as _html
        safe_msg = _html.escape(str(msg))
        st.markdown(f"<div class='user-msg'>{safe_msg}</div>", unsafe_allow_html=True)
    elif role == "bot":
        if msg.startswith("[IMAGE]"):
            st.image(msg.replace("[IMAGE]", ""))
        else:
            formatted = format_answer_html(str(msg))
            st.markdown(f"<div class='bot-msg'>{formatted}</div>", unsafe_allow_html=True)
    elif role == "bot_with_sources":
        st.markdown(f"<div class='bot-msg'>{msg['answer']}</div>", unsafe_allow_html=True)
        if msg.get('sources_html'):
            st.markdown(
                f"<div class='sop-compact'>{msg['sources_html']}</div>",
                unsafe_allow_html=True
            )
    elif role == "followup":
        st.markdown(
            f"""
            <div class='followup-box'>
                💡 {msg}
            </div>
            """,
            unsafe_allow_html=True
        )

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

    sources_html = metadata_handler.format_sources_html(docs)

    combined_answer = format_answer_html(main_answer)

    if followup:
        combined_answer += f"<div class='followup-inline'>💡 {followup}</div>"

    st.session_state.messages.append(
        ("bot_with_sources", {
            "answer": combined_answer,
            "sources_html": sources_html
        })
    )

    if followup:
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
        sources_html = metadata_handler.format_sources_html(docs)

        combined_answer = format_answer_html(main_answer)

        if followup:
            combined_answer += f"<div class='followup-inline'>💡 {followup}</div>"

        st.session_state.messages.append(("bot_with_sources", {
            'answer': combined_answer,
            'sources_html': sources_html
        }))

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
        st.session_state.pending_question = question  # queue for PASS 2
        st.rerun()

# PASS 2: user bubble is visible, now show typing → answer
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
        st.session_state.pending_followup     = "image_shown"
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
        sources_html = clean_sources_html(raw_sources)
        combined_answer = format_answer_html(main_answer)
        if followup:
            combined_answer += f"<div class='followup-inline'>💡 {followup}</div>"
        st.session_state.messages.append(("bot_with_sources", {
            'answer': combined_answer,
            'sources_html': sources_html
        }))

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