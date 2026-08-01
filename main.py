import rag.config  # noqa: F401  (sets HF offline mode before heavy imports)

from rag.rag_chain import get_rag_chain


def main():
    """
    Run the RAG application.
    """

    print("=" * 50)
    print("Healthcare RAG Chatbot")
    print("Type 'exit' to quit.")
    print("=" * 50)

    rag_chain = get_rag_chain()

    while True:
        question = input("\nAsk a question: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        response = rag_chain.invoke(
            {
                "input": question
            }
        )

        print("\nAnswer:")
        print(response["answer"])


if __name__ == "__main__":
    main()