# app.py

import streamlit as st

from rag.retriever import retrieve_documents
from rag.rag_chain import generate_response


# -------------------------------
# Streamlit Page Configuration
# -------------------------------

st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide"
)


# -------------------------------
# Application Title
# -------------------------------

st.title("🤖 RAG Document Assistant")

st.write(
    "Ask questions from your uploaded documents."
)


# -------------------------------
# Store Chat History
# -------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------------
# Display Previous Chat
# -------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])



# -------------------------------
# User Input
# -------------------------------

query = st.chat_input(
    "Ask something about your documents..."
)



# -------------------------------
# RAG Pipeline
# -------------------------------

if query:

    # Display user message
    with st.chat_message("user"):
        st.write(query)


    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )


    # 1. Retrieve relevant documents
    documents = retrieve_documents(query)


    # 2. Generate answer using LLM
    answer = generate_response(
        query=query,
        documents=documents
    )


    # Display AI answer
    with st.chat_message("assistant"):
        st.write(answer)


    # Save AI response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )