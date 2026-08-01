from unittest.mock import patch

from langchain_core.documents import Document

from rag.retriever import get_retriever, retrieve_documents


@patch("rag.retriever.load_vector_store")
def test_get_retriever_top_k_is_three(mock_load):
    mock_store = mock_load.return_value

    get_retriever()

    mock_store.as_retriever.assert_called_once_with(search_kwargs={"k": 3})


@patch("rag.retriever.load_vector_store")
def test_get_retriever_returns_retriever(mock_load):
    mock_store = mock_load.return_value

    result = get_retriever()

    assert result is mock_store.as_retriever.return_value


@patch("rag.retriever.get_retriever")
def test_retrieve_documents_returns_results(mock_get_retriever):
    docs = [Document(page_content="d1", metadata={"source": "s"})]
    mock_get_retriever.return_value.invoke.return_value = docs

    result = retrieve_documents("query")

    mock_get_retriever.return_value.invoke.assert_called_once_with("query")
    assert result == docs


@patch("rag.retriever.get_retriever")
def test_retrieve_documents_no_results(mock_get_retriever):
    mock_get_retriever.return_value.invoke.return_value = []

    result = retrieve_documents("nothing")

    assert result == []
