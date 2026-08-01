from unittest.mock import Mock, patch

from rag.loader import load_documents


@patch("rag.loader.PyPDFLoader")
@patch("rag.loader.Path.glob")
def test_load_documents_loads_all_pdfs(mock_glob, mock_loader_cls):
    from pathlib import Path

    mock_glob.return_value = [
        Path("data/a.pdf"),
        Path("data/b.pdf"),
    ]

    loader_a = Mock()
    loader_b = Mock()
    mock_loader_cls.side_effect = [loader_a, loader_b]

    loader_a.load.return_value = ["doc_a_1", "doc_a_2"]
    loader_b.load.return_value = ["doc_b_1"]

    docs = load_documents()

    assert docs == ["doc_a_1", "doc_a_2", "doc_b_1"]
    assert mock_loader_cls.call_count == 2


@patch("rag.loader.PyPDFLoader")
@patch("rag.loader.Path.glob")
def test_load_documents_no_pdfs(mock_glob, mock_loader_cls):
    mock_glob.return_value = []

    docs = load_documents()

    assert docs == []
    mock_loader_cls.assert_not_called()
