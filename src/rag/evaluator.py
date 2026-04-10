"""
src/rag/evaluator.py
--------------------
Faithfulness checker — evaluates whether the generated answer
is grounded in the retrieved context chunks.

How it works:
1. Extract key phrases (noun chunks) from the answer using simple NLP
2. Check what fraction of those phrases appear in the retrieved context
3. Return a score (0-1) and a label: grounded / partially grounded / hallucinated

This is a lightweight, dependency-free approach — no external eval
frameworks needed. Good enough to demonstrate the concept in interviews
and catches obvious hallucinations reliably.
"""

import re


# ============================================
# KEY PHRASE EXTRACTION
# ============================================

def extract_key_phrases(text: str) -> list[str]:
    """Extract meaningful phrases from text for overlap checking.

    Approach: extract capitalized words/phrases and numbers,
    which carry the most factual content in financial text.
    Also extracts multi-word sequences of 2-3 words.
    """
    text_clean = text.strip()

    phrases = []

    # 1. Numbers and percentages (most important in financial context)
    numbers = re.findall(r'\$?[\d,]+\.?\d*\s*(?:billion|million|percent|%)?', text_clean)
    phrases.extend([n.strip() for n in numbers if len(n.strip()) > 1])

    # 2. Capitalized proper nouns / named entities (company names, people, products)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text_clean)
    phrases.extend([p.strip() for p in proper_nouns if len(p) > 3])

    # 3. Key financial terms
    financial_terms = [
        "revenue", "growth", "profit", "margin", "earnings", "forecast",
        "guidance", "acquisition", "investment", "capital", "cash flow",
        "operating", "net income", "gross margin", "data center", "cloud",
        "artificial intelligence", "machine learning", "supply chain"
    ]
    text_lower = text_clean.lower()
    for term in financial_terms:
        if term in text_lower:
            phrases.append(term)

    return list(set(phrases))  # deduplicate


# ============================================
# FAITHFULNESS SCORER
# ============================================

def compute_faithfulness(answer: str, retrieved_chunks: list[dict]) -> dict:
    """Score how grounded the answer is in the retrieved context.

    Args:
        answer: The LLM-generated answer string
        retrieved_chunks: List of dicts with 'text' key (from retrieve_top_chunks)

    Returns:
        dict with:
            score (float): 0.0 to 1.0
            label (str): 'grounded' | 'partially grounded' | 'hallucinated'
            matched (list): phrases found in context
            unmatched (list): phrases NOT found in context
            explanation (str): human-readable summary
    """

    if not answer or not retrieved_chunks:
        return {
            "score": 0.0,
            "label": "hallucinated",
            "matched": [],
            "unmatched": [],
            "explanation": "No answer or context to evaluate."
        }

    # Combine all retrieved chunk text into one searchable string
    full_context = " ".join(chunk["text"].lower() for chunk in retrieved_chunks)

    # Extract key phrases from the answer
    key_phrases = extract_key_phrases(answer)

    if not key_phrases:
        # No extractable phrases — answer is probably too vague to evaluate
        return {
            "score": 0.5,
            "label": "partially grounded",
            "matched": [],
            "unmatched": [],
            "explanation": "Could not extract key claims to verify."
        }

    # Check which phrases appear in the context
    matched = []
    unmatched = []

    for phrase in key_phrases:
        if phrase.lower() in full_context:
            matched.append(phrase)
        else:
            unmatched.append(phrase)

    # Compute score as fraction of matched phrases
    score = len(matched) / len(key_phrases)

    # Assign label based on score thresholds
    if score >= 0.7:
        label = "grounded"
        explanation = f"{len(matched)}/{len(key_phrases)} key claims found in source transcripts."
    elif score >= 0.4:
        label = "partially grounded"
        explanation = f"{len(matched)}/{len(key_phrases)} key claims verified. Some details may not be in the retrieved context."
    else:
        label = "hallucinated"
        explanation = f"Only {len(matched)}/{len(key_phrases)} claims found in context. Answer may contain fabricated details."

    return {
        "score": round(score, 2),
        "label": label,
        "matched": matched,
        "unmatched": unmatched,
        "explanation": explanation
    }
