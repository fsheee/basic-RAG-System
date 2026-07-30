from langchain_groq import ChatGroq

from rag.config import GROQ_API_KEY, MODEL_NAME


def get_llm():
    """
    Create and return the Groq LLM.
    """

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL_NAME,
        temperature=0
    )

    return llm