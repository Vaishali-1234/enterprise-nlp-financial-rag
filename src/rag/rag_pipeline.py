import time
from src.rag.retrieve_chunks import retrieve_top_chunks
from src.rag.prompt_builder import build_prompt
from src.rag.generate_answer import generate_answer
from src.rag.query_rewriter import rewrite_query


def ask_question(query):

    # ── Query rewriting ───────────────────────────────────────────────
    if len(query.split()) <= 3:
        better_query = rewrite_query(query)
    else:
        better_query = query

    # ── Retrieval timing ──────────────────────────────────────────────
    # Measures ONLY the FAISS search + filtering step.
    # This is what proves IVF optimization worked — should be ~0.1s
    retrieval_start = time.time()
    chunks = retrieve_top_chunks(better_query)
    retrieval_time = round(time.time() - retrieval_start, 3)

    # ── Generation timing ─────────────────────────────────────────────
    # Measures ONLY the LLM generation step.
    # This is the actual bottleneck — proves GPU/API would help most.
    prompt = build_prompt(query, chunks)

    generation_start = time.time()
    answer = generate_answer(prompt)
    generation_time = round(time.time() - generation_start, 3)

    # Return timings alongside answer and chunks
    timings = {
        "retrieval_seconds": retrieval_time,
        "generation_seconds": generation_time,
        "total_seconds": round(retrieval_time + generation_time, 3)
    }

    return answer, chunks, timings


if __name__ == "__main__":
    query = input("Ask a financial question: ")
    answer, _, timings = ask_question(query)
    print("\nAI Answer:\n")
    print(answer)
    print(f"\nRetrieval: {timings['retrieval_seconds']}s | Generation: {timings['generation_seconds']}s")
