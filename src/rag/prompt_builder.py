def build_prompt(query, retrieved_chunks):

    context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)

    prompt = f"""
You are a financial research assistant.

Use ONLY the provided earnings call transcript context to answer the question.

Do not use outside knowledge.

If the answer cannot be found in the context, say:
"The transcripts do not contain enough information to answer this question."

Provide a clear and concise summary.

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt