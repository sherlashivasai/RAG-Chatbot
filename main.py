import streamlit as st
from src.herlper import get_conversational_chain, get_vector_store,handle_user_query
from dotenv import load_dotenv
import os

load_dotenv()

def user_input(query):
    response = st.session_state.conversational_chain({'question':query})
    st.session_state.chat_history = response['chat_history']
    for i,message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.markdown(f"**User:** {message['content']}")
        else:
            st.markdown(f"**Bot:** {message['content']}")

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
    st.title("RAG Chatbot Application")
    st.write("Welcome to the RAG Chatbot! Ask me anything.")

    query = st.text_input("Enter your question here:")
    if 'conversational_chain' not in st.session_state:
        st.session_state.conversational_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

   

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF documents", type=["pdf"], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Placeholder for processing logic
                vector_store = get_vector_store(pdf_docs)
                conversational_chain = get_conversational_chain(vector_store)
                st.session_state.conversational_chain = conversational_chain

                st.success("Documents processed successfully!")

if __name__ == "__main__":
    main()
