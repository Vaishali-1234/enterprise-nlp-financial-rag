from retrieve_chunks import retrieve_top_chunks
from prompt_builder import build_prompt
from generate_answer import generate_answer


def ask_question(query):
    chunks = retrieve_top_chunks(query)

    prompt = build_prompt(query, chunks)

    answer = generate_answer(prompt)

    return answer


if __name__ == "__main__":
    query = input("Ask a financial question: ")

    answer = ask_question(query)

    print("\nAI Answer:\n")
    print(answer)