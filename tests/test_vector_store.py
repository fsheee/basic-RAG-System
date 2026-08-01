from unittest.mock import patch

from rag.vector_store import create_vector_store, load_vector_store


@patch("rag.vector_store.Chroma.from_documents")
@patch("rag.vector_store.get_embedding_model")
@patch("rag.vector_store.CHROMA_DB_DIR", "test_db")
def test_create_vector_store_calls_chroma(mock_emb, mock_from_documents):
    chunks = ["chunk1", "chunk2"]

    result = create_vector_store(chunks)

    mock_from_documents.assert_called_once_with(
        documents=chunks,
        embedding=mock_emb.return_value,
        persist_directory="test_db",
    )
    assert result is mock_from_documents.return_value


@patch("rag.vector_store.Chroma")
@patch("rag.vector_store.get_embedding_model")
@patch("rag.vector_store.CHROMA_DB_DIR", "test_db")
def test_load_vector_store_calls_chroma(mock_emb, mock_chroma):
    result = load_vector_store()

    mock_chroma.assert_called_once_with(
        persist_directory="test_db",
        embedding_function=mock_emb.return_value,
    )
    assert result is mock_chroma.return_value
