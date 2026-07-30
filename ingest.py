from rag.loader import load_documents
from rag.splitter import split_documents
from rag.vector_store import create_vector_store


def main():
    """
    Build the ChromaDB vector database.
    """

    print("Loading documents...")

    documents = load_documents()

    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents...")

    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    print("Creating vector database...")

    create_vector_store(chunks)

    print("Vector database created successfully!")


if __name__ == "__main__":
    main()