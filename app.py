import streamlit as st
from src.rag.rag_pipeline import ask_question

st.title("Enterprise Financial AI Assistant")

st.write("Ask questions about financial earnings call transcripts.")

query = st.text_input("Enter your question")

if st.button("Ask"):

    if query.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching financial transcripts..."):

            answer, sources = ask_question(query)

        st.subheader("AI Answer")
        st.write(answer)

        st.subheader("Sources")

        for s in sources:
            st.markdown(f"### {s['ticker']} — {s['year']} {s['quarter']}")
            st.write(s["text"][:400] + "...")
            st.divider()