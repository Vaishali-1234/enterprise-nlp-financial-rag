import ollama


def rewrite_query(query):

    prompt = f"""
You are a financial search assistant.

Rewrite the user's question into a detailed semantic search query
that can be used to retrieve relevant earnings call transcript excerpts.

Focus on:
- financial strategies
- risks
- investments
- growth
- market trends
- business performance

User Question:
{query}

Rewritten Search Query:
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 60}
    )

    better_query = response["message"]["content"].strip()

    return better_query