from langchain_core.documents import Document

from rag.splitter import split_documents


def test_split_documents_returns_chunks():
    docs = [
        Document(page_content="word " * 2000, metadata={"source": "test.pdf"}),
    ]

    chunks = split_documents(docs)

    assert len(chunks) > 1
    assert all(isinstance(c, Document) for c in chunks)
    assert all(c.metadata == {"source": "test.pdf"} for c in chunks)


def test_split_documents_short_text_single_chunk():
    docs = [Document(page_content="short text", metadata={})]

    chunks = split_documents(docs)

    assert len(chunks) == 1
    assert chunks[0].page_content == "short text"


def test_split_documents_empty_input():
    chunks = split_documents([])

    assert chunks == []


def test_split_chunks_respect_max_size():
    docs = [Document(page_content="word " * 1000, metadata={})]

    chunks = split_documents(docs)

    assert all(len(c.page_content) <= 500 for c in chunks)
