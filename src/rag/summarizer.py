from src.rag.retrieve_chunks import retrieve_top_chunks
from src.rag.generate_answer import generate_answer


def summarize_company(query, ticker=None, year=None, quarter=None):

    chunks = retrieve_top_chunks(query, top_k=8)

    # filtering logic
    filtered = []

    for c in chunks:

        if ticker and c["ticker"] != ticker:
            continue

        if year and str(c["year"]) != str(year):
            continue

        if quarter and c["quarter"] != quarter:
            continue

        filtered.append(c)

    if len(filtered) == 0:
        filtered = chunks

    context = "\n\n".join(c["text"][:700] for c in filtered)

    prompt = f"""
You are a senior financial research analyst.

Create an executive summary of the earnings discussion.

Focus on:
- Growth Strategy
- Financial Performance
- Risks
- Investments
- Future Outlook

Transcript Context:
{context}

Executive Summary:
"""

    summary = generate_answer(prompt)

    return summary, filtered