import streamlit as st
import time
from src.rag.rag_pipeline import ask_question
from src.rag.summarizer import summarize_company
from src.rag.evaluator import compute_faithfulness

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Financial AI Assistant",
    page_icon="📊",
    layout="centered"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>

.stApp { background-color: #0f1117; }

.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.2rem;
}

.main-subtitle {
    font-size: 1rem;
    color: #8b8fa8;
    margin-bottom: 2rem;
}

.mode-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}

.mode-qa { background: #1a3a5c; color: #60a5fa; }
.mode-sum { background: #1a3a2a; color: #34d399; }

.answer-card {
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    color: #e2e8f0;
    font-size: 1rem;
    line-height: 1.7;
}

.metric-card {
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
}

.metric-label {
    font-size: 0.75rem;
    color: #8b8fa8;
    margin-top: 4px;
}

.faith-grounded {
    background: #052e16;
    border: 1px solid #166534;
    color: #4ade80;
    padding: 8px 16px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
}

.faith-partial {
    background: #1c1400;
    border: 1px solid #854d0e;
    color: #fbbf24;
    padding: 8px 16px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
}

.faith-hallucinated {
    background: #1c0505;
    border: 1px solid #991b1b;
    color: #f87171;
    padding: 8px 16px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
}

.section-header {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #8b8fa8;
    text-transform: uppercase;
    margin: 1.5rem 0 0.75rem 0;
    border-bottom: 1px solid #2d3148;
    padding-bottom: 0.5rem;
}

.stTextInput > div > div > input {
    background-color: #1e2130 !important;
    border: 1px solid #2d3148 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

.stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
}

.stSelectbox > div > div {
    background-color: #1e2130 !important;
    border: 1px solid #2d3148 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

hr { border-color: #2d3148 !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

</style>
""", unsafe_allow_html=True)

# ============================================
# PRELOAD RESOURCES
# ============================================

@st.cache_resource(show_spinner="Loading financial intelligence engine...")
def preload_resources():
    from src.rag.retrieve_chunks import load_retrieval_resources
    return load_retrieval_resources()

preload_resources()

# ============================================
# HEADER
# ============================================

st.markdown('<div class="main-title">📊 Financial AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Semantic search and question answering over 18,755 earnings call transcripts</div>', unsafe_allow_html=True)
st.divider()

# ============================================
# INPUTS
# ============================================

mode = st.selectbox("Select Mode", ["Question Answering", "Summarization"], label_visibility="collapsed")

if mode == "Question Answering":
    st.markdown('<span class="mode-badge mode-qa">Question Answering Mode</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="mode-badge mode-sum">Summarization Mode</span>', unsafe_allow_html=True)

query = st.text_input(
    "Query",
    placeholder="e.g. What did NVDA say about data center growth in 2022?",
    label_visibility="collapsed"
)

if mode == "Summarization":
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Ticker", placeholder="e.g. NVDA")
    with col2:
        year = st.text_input("Year", placeholder="e.g. 2022")
    with col3:
        quarter = st.text_input("Quarter", placeholder="e.g. Q3")
else:
    ticker, year, quarter = "", "", ""

run = st.button("Analyze", type="primary")

# ============================================
# RUN
# ============================================

if run:

    if not query.strip():
        st.warning("Please enter a query.")

    elif mode == "Summarization" and not ticker.strip():
        st.warning("Please enter a company ticker for summarization.")

    else:
        with st.spinner("Analyzing financial transcripts..."):

            if mode == "Question Answering":
                answer, sources, timings = ask_question(query)
            else:
                t0 = time.time()
                answer, sources = summarize_company(
                    query=query,
                    ticker=ticker,
                    year=year,
                    quarter=quarter
                )
                timings = {
                    "retrieval_seconds": "—",
                    "generation_seconds": "—",
                    "total_seconds": round(time.time() - t0, 1)
                }

            faith = compute_faithfulness(answer, sources)

        # ── Answer ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">AI Output</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

        # ── Evaluation metrics ────────────────────────────────────────
        st.markdown('<div class="section-header">Evaluation</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            label = faith["label"]
            score = faith["score"]
            if label == "grounded":
                st.markdown(f'<div class="faith-grounded">✓ Grounded ({score:.0%})</div>', unsafe_allow_html=True)
            elif label == "partially grounded":
                st.markdown(f'<div class="faith-partial">~ Partially grounded ({score:.0%})</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="faith-hallucinated">✗ Hallucinated ({score:.0%})</div>', unsafe_allow_html=True)
            st.caption(faith["explanation"])

        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{timings["retrieval_seconds"]}s</div>
                <div class="metric-label">Retrieval time</div>
            </div>''', unsafe_allow_html=True)

        with col3:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{timings["generation_seconds"]}s</div>
                <div class="metric-label">Generation time</div>
            </div>''', unsafe_allow_html=True)

        with st.expander("Faithfulness detail"):
            if faith["matched"]:
                st.markdown("**Verified in context:**")
                st.write(", ".join(faith["matched"]))
            if faith["unmatched"]:
                st.markdown("**Not found in context:**")
                st.write(", ".join(faith["unmatched"]))

        # ── Sources ───────────────────────────────────────────────────
        st.markdown('<div class="section-header">Sources Used</div>', unsafe_allow_html=True)

        if not sources:
            st.info("No sources returned.")
        else:
            for s in sources:
                rel_score = s.get("relevance_score", None)

                if rel_score is not None:
                    if rel_score >= 0.7:
                        score_label = f"🟢 {rel_score}"
                    elif rel_score >= 0.5:
                        score_label = f"🟡 {rel_score}"
                    else:
                        score_label = f"🔴 {rel_score}"
                    expander_label = f"{s['ticker']} — {s['year']} {s['quarter']}  |  Relevance: {score_label}"
                else:
                    expander_label = f"{s['ticker']} — {s['year']} {s['quarter']}"

                with st.expander(expander_label):
                    st.write(s["text"][:400] + "...")
