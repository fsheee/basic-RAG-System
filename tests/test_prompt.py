from rag.prompt import RAG_PROMPT


def test_prompt_contains_context_placeholder():
    assert "context" in RAG_PROMPT.input_variables


def test_prompt_contains_question_placeholder():
    assert "input" in RAG_PROMPT.input_variables


def test_prompt_formats_message():
    messages = RAG_PROMPT.format_messages(
        context="Hospitals open 24/7.",
        input="What are visiting hours?",
    )

    assert len(messages) == 1
    assert "Hospitals open 24/7." in messages[0].content
    assert "What are visiting hours?" in messages[0].content
