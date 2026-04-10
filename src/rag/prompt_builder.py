def build_prompt(query, retrieved_chunks):

    context = "\n\n".join(chunk["text"][:600] for chunk in retrieved_chunks)

    prompt = f"""
You are a financial research assistant analyzing earnings call transcripts.

Use the provided transcript context to answer the question as specifically as possible.
Include relevant figures, names, and quotes from the context where available.
If only partial information is available, answer with what is provided and note any gaps.
Only say information is unavailable if the context contains absolutely no relevant details.

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt
