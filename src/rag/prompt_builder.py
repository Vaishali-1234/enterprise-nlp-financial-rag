def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a financial research assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt