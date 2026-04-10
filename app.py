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
# PRELOAD RESOURCES AT STARTUP
# ============================================

@st.cache_resource(show_spinner="Loading financial intelligence engine...")
def preload_resources():
    from src.rag.retrieve_chunks import load_retrieval_resources
    return load_retrieval_resources()

preload_resources()

# ============================================
# UI
# ============================================

st.title("📊 Enterprise Financial AI Assistant")
st.write("Ask questions or generate executive summaries from earnings call transcripts.")

mode = st.selectbox("Select Mode", ["Question Answering", "Summarization"])

st.divider()

query = st.text_input(
    "Enter your query",
    placeholder="e.g. What did NVDA say about data center growth in 2023?"
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

# ============================================
# RUN
# ============================================

if st.button("Run", type="primary"):

    if not query.strip():
        st.warning("Please enter a query.")

    elif mode == "Summarization" and not ticker.strip():
        st.warning("Please enter a company ticker for summarization.")

    else:
        with st.spinner("Analyzing financial transcripts..."):

            start = time.time()

            if mode == "Question Answering":
                answer, sources = ask_question(query)
            else:
                answer, sources = summarize_company(
                    query=query,
                    ticker=ticker,
                    year=year,
                    quarter=quarter
                )

            total_time = time.time() - start

            faith = compute_faithfulness(answer, sources)

        # ============================================
        # OUTPUT
        # ============================================

        st.subheader("AI Output")
        st.write(answer)

        # ── Faithfulness + latency row ────────────────────────────────
        st.divider()
        col_f, col_t = st.columns([2, 1])

        with col_f:
            label = faith["label"]
            score = faith["score"]

            if label == "grounded":
                st.success(f"Faithfulness: {label} ({score:.0%})")
            elif label == "partially grounded":
                st.warning(f"Faithfulness: {label} ({score:.0%})")
            else:
                st.error(f"Faithfulness: {label} ({score:.0%})")

            st.caption(faith["explanation"])

        with col_t:
            st.metric("Total time", f"{total_time:.1f}s")
            st.caption("FAISS retrieval: ~0.1s | Remainder: LLM generation")

        with st.expander("Faithfulness detail"):
            if faith["matched"]:
                st.markdown("**Verified in context:**")
                st.write(", ".join(faith["matched"]))
            if faith["unmatched"]:
                st.markdown("**Not found in context:**")
                st.write(", ".join(faith["unmatched"]))

        # ── Sources with relevance scores ─────────────────────────────
        st.subheader("Sources Used")

        if not sources:
            st.info("No sources returned.")
        else:
            for s in sources:
                # Get relevance score — default to None if not present
                rel_score = s.get("relevance_score", None)

                # Build expander label with score if available
                # Score color: green >= 0.7, yellow >= 0.5, red < 0.5
                if rel_score is not None:
                    if rel_score >= 0.7:
                        score_label = f"🟢 {rel_score}"
                    elif rel_score >= 0.5:
                        score_label = f"🟡 {rel_score}"
                    else:
                        score_label = f"🔴 {rel_score}"

                    label = f"{s['ticker']} — {s['year']} {s['quarter']}  |  Relevance: {score_label}"
                else:
                    label = f"{s['ticker']} — {s['year']} {s['quarter']}"

                with st.expander(label):
                    st.write(s["text"][:400] + "...")
