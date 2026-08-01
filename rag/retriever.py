from rag.vector_store import ensure_vector_store


def get_retriever():
    """
    Load the Chroma vector store and create a retriever.
    """

    # Load the existing Chroma vector database (building it from data/ PDFs
    # first if it is missing or empty, e.g. on a fresh Streamlit Cloud deploy)
    vector_store = ensure_vector_store()

    # Create a retriever that returns the top 3 matching documents
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever


def retrieve_documents(query):
    """
    Retrieve the most relevant documents for the user's query.
    """

    # Get the retriever
    retriever = get_retriever()

    # Search the vector database
    documents = retriever.invoke(query)

    # Debug output
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Retrieved {len(documents)} document(s)")
    print("=" * 60)

    if not documents:
        print("No matching documents found.")

    for i, doc in enumerate(documents, start=1):
        print(f"\nDocument {i}")
        print("-" * 40)
        print(doc.page_content)
        print("-" * 40)

        # Print metadata if available
        if hasattr(doc, "metadata"):
            print("Metadata:", doc.metadata)

    return documents

