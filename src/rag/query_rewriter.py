import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def rewrite_query(query):
    prompt = f"""You are a financial search assistant.

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

Rewritten Search Query (one line only, no explanation):
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return query
