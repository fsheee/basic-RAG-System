from unittest.mock import patch

from rag.llm import get_llm


@patch("rag.llm.ChatGroq")
@patch("rag.llm.MODEL_NAME", "test-model")
@patch("rag.llm.GROQ_API_KEY", "test-key")
def test_get_llm_creates_chat_groq(mock_chat_groq):
    get_llm()

    mock_chat_groq.assert_called_once_with(
        api_key="test-key",
        model="test-model",
        temperature=0,
    )


@patch("rag.llm.ChatGroq")
@patch("rag.llm.MODEL_NAME", "test-model")
@patch("rag.llm.GROQ_API_KEY", "test-key")
def test_get_llm_returns_instance(mock_chat_groq):
    result = get_llm()

    assert result is mock_chat_groq.return_value
