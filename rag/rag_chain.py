from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from rag.llm import get_llm
from rag.prompt import RAG_PROMPT
from rag.retriever import get_retriever


def get_rag_chain():
    """
    Create and return the RAG chain.
    """

    llm = get_llm()

    retriever = get_retriever()

    document_chain = create_stuff_documents_chain(
        llm,
        RAG_PROMPT
    )

    rag_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return rag_chain


def generate_response(query, documents):
    """
    Generate an answer using the given query and documents.
    """

    llm = get_llm()

    document_chain = create_stuff_documents_chain(
        llm,
        RAG_PROMPT
    )

    response = document_chain.invoke(
        {
             "context": documents,
             "input": query 
        }
    )

    return response