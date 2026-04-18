def build_prompt(query, retrieved_chunks):

    # Include source metadata (ticker, year, quarter) with each chunk
    # so the LLM knows which company each piece of context belongs to
    # and can reference the company name in its answer.
    context = "\n\n".join(
        f"[Source: {chunk['ticker']} | {chunk['year']} {chunk['quarter']}]\n{chunk['text'][:600]}"
        for chunk in retrieved_chunks
    )

    prompt = f"""
You are a financial research assistant analyzing earnings call transcripts.

Use the provided transcript context to answer the question as specifically as possible.
Each source is labeled with the company ticker, year and quarter.
Always mention the company ticker or name when referencing specific information.
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
