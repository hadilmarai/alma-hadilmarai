import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from collaboration import run_stage, STAGES
from rag import MiniRAG


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="ALMA – Collaborative Debugging Assistant",
    layout="centered"
)

# --------------------------------------------------
# Chat CSS (LEFT = LLM, RIGHT = USER)
# --------------------------------------------------
st.markdown("""
<style>
.chat-row {
    display: flex;
    margin-bottom: 12px;
}

.chat-left {
    justify-content: flex-start;
}

.chat-right {
    justify-content: flex-end;
}

.chat-bubble-llm {
    background-color: #e8f5ee;
    padding: 12px;
    border-radius: 12px;
    max-width: 70%;
}

.chat-bubble-user {
    background-color: #f1f3f6;
    padding: 12px;
    border-radius: 12px;
    max-width: 70%;
}

.chat-meta {
    font-size: 0.8em;
    color: #666;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🤝 Collaborative Debugging Assistant")
st.caption(
    "A structured discussion with an LLM acting as a co-explorer, "
    "not an answer machine."
)

# --------------------------------------------------
# Session state initialization
# --------------------------------------------------
if "stage_idx" not in st.session_state:
    st.session_state.stage_idx = 0

if "history" not in st.session_state:
    st.session_state.history = []

if "rag" not in st.session_state:
    try:
        st.session_state.rag = MiniRAG.from_folder("data")
    except Exception:
        st.session_state.rag = None
        st.warning("Reference materials unavailable (RAG disabled).")

# --------------------------------------------------
# Current stage
# --------------------------------------------------
current_stage = STAGES[st.session_state.stage_idx]

st.markdown(
    f"<div style='color:#666;font-size:0.9em;'>"
    f"Current collaboration phase: <b>{current_stage}</b>"
    f"</div>",
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# Conversation view (COHERENT DISCUSSION)
# --------------------------------------------------
for turn in st.session_state.history:

    # Learner message (RIGHT)
    st.markdown(
        f"""
        <div class="chat-row chat-right">
            <div>
                <div class="chat-meta">🧑‍🎓 You</div>
                <div class="chat-bubble-user">
                    {turn['learner']}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # LLM message (LEFT)
    st.markdown(
        f"""
        <div class="chat-row chat-left">
            <div>
                <div class="chat-meta">🤖 LLM — {turn['stage']}</div>
                <div class="chat-bubble-llm">
                    {turn['assistant']}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # RAG references
    rag_items = turn.get("rag", [])
    if rag_items:
        with st.expander("📚 Shared reference material used", expanded=False):
            for item in rag_items:
                st.markdown(
                    f"**{item['doc_id']}** (relevance: {item['score']:.2f})"
                )
                st.markdown(item["excerpt"] + "…")
                st.divider()

    st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# Chat input (ONLY INPUT MECHANISM)
# --------------------------------------------------
user_text = st.chat_input(
    "Respond to the LLM, explain your reasoning, or ask a follow-up question…"
)

if user_text:
    current_stage = STAGES[st.session_state.stage_idx]

    answer, retrieved = run_stage(
        stage=current_stage,
        learner_input=user_text,
        history=st.session_state.history,
        rag=st.session_state.rag
    )

    st.session_state.history.append({
        "stage": current_stage,
        "learner": user_text,
        "assistant": answer,
        "rag": retrieved
    })

    # Advance stage
    st.session_state.stage_idx = (
        st.session_state.stage_idx + 1
    ) % len(STAGES)

    st.rerun()
