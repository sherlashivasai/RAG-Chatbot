from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

load_dotenv()
Google_key = os.getenv("GOOGLE_API_KEY")
embeddings = GooglePalmEmbeddings(api_key=Google_key)

def get_pdf_texts(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(chunks):
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def get_vector_store(pdf_docs):
    text = get_pdf_texts(pdf_docs)
    chunks = get_text_chunks(text)
    vector_store = create_vector_store(chunks)
    return vector_store

def get_conversational_chain(vector_store):
    llm = GooglePalm(api_key=Google_key, model_name="gemini-pro")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversational_chain

def handle_user_query(user_query, conversational_chain):
    response = conversational_chain.run(input=user_query)
    return response

