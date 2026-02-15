from rag import RAGChain


def main() -> None:
    print("Research Paper Assistant")
    print("Ask questions about your ingested papers.")
    print("Commands: /reset (clear history), /quit or /exit (quit)\n")

    chain = RAGChain()

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question in ("/quit", "/exit"):
            print("Goodbye!")
            break
        if question == "/reset":
            chain.reset()
            print("Conversation history cleared.\n")
            continue

        result = chain.ask(question)
        print(f"\nAssistant: {result['answer']}")
        print(f"\nSources:\n{result['sources']}\n")


if __name__ == "__main__":
    main()
