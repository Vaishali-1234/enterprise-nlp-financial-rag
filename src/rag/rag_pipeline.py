from src.rag.retrieve_chunks import retrieve_top_chunks
from src.rag.prompt_builder import build_prompt
from src.rag.generate_answer import generate_answer
from src.rag.query_rewriter import rewrite_query

def ask_question(query):

    if len(query.split()) <= 3:
        better_query = rewrite_query(query)
    else:
        better_query = query

    chunks = retrieve_top_chunks(better_query)

    prompt = build_prompt(query, chunks)

    answer = generate_answer(prompt)

    return answer, chunks


if __name__ == "__main__":
    query = input("Ask a financial question: ")

    answer, _ = ask_question(query)

    print("\nAI Answer:\n")
    print(answer)