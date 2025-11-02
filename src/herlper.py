from typing import Iterable, List, Optional
import os

from dotenv import load_dotenv


load_dotenv()


def _get_env_key(name: str) -> Optional[str]:
    """Return environment variable or None if missing."""
    return os.getenv(name)


def build_embeddings(api_key: Optional[str] = None):
    """Lazily create embeddings instance. Raises ValueError if key missing.

    Keeping creation inside a function avoids errors at import time when the
    environment isn't configured yet.
    """
    key = api_key or _get_env_key("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY is not set. Set it in the environment.")
    try:
        from langchain.embeddings import GooglePalmEmbeddings
    except Exception as exc:  # ImportError or resolution issues
        raise ImportError(
            "langchain.embeddings.GooglePalmEmbeddings is required but not available. "
            "Install 'langchain' and its optional dependencies."
        ) from exc
    return GooglePalmEmbeddings(api_key=key)


def get_pdf_texts(pdf_docs: Iterable[str]) -> str:
    """Read a list of PDF file paths and return concatenated text.

    Non-text pages are skipped. Any PDF read errors bubble up to the caller.
    """
    try:
        from PyPDF2 import PdfReader
    except Exception as exc:
        raise ImportError(
            "PyPDF2 is required to read PDF files. Install it with 'pip install PyPDF2'."
        ) from exc

    parts: List[str] = []
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                parts.append(page_text)
    return "\n".join(parts)


def get_text_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into chunks suitable for embedding and indexing."""
    if not text:
        return []
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception as exc:
        raise ImportError(
            "langchain.text_splitter.RecursiveCharacterTextSplitter is required. "
            "Install 'langchain' to enable text splitting."
        ) from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "],
    )
    return splitter.split_text(text)


def create_vector_store(chunks: List[str], embeddings=None):
    """Create a FAISS vector store from text chunks.

    If embeddings is None the function will attempt to build them from
    environment configuration.
    """
    if not chunks:
        raise ValueError("No text chunks provided to index.")
    emb = embeddings or build_embeddings()
    try:
        from langchain.vectorstores import FAISS
    except Exception as exc:
        raise ImportError(
            "langchain.vectorstores.FAISS is required to build the vector store. "
            "Install 'faiss-cpu' (or 'faiss-gpu') and 'langchain' optional dependencies."
        ) from exc
    return FAISS.from_texts(chunks, emb)


def get_vector_store(pdf_docs: Iterable[str], api_key: Optional[str] = None):
    """Complete pipeline: read PDFs, split text, and build a vector store."""
    text = get_pdf_texts(pdf_docs)
    chunks = get_text_chunks(text)
    return create_vector_store(chunks, embeddings=(build_embeddings(api_key) if api_key else None))


def get_conversational_chain(vector_store, api_key: Optional[str] = None, model_name: str = "gemini-pro"):
    """Create a ConversationalRetrievalChain wired to the provided vector store.

    Raises ValueError if an API key is required but missing.
    """
    key = api_key or _get_env_key("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY is not set. Set it in the environment to use GooglePalm.")

    try:
        from langchain.llms import GooglePalm
        from langchain.memory import ConversationBufferMemory
        from langchain.chains import ConversationalRetrievalChain
    except Exception as exc:
        raise ImportError(
            "langchain llm/memory/chains are required to build the conversational chain. "
            "Install 'langchain' and its optional dependencies."
        ) from exc

    llm = GooglePalm(api_key=key, model_name=model_name)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )
    return chain


def handle_user_query(user_query: str, conversational_chain) -> str:
    """Execute the conversational chain with a user query and return the answer.

    The function is defensive about the chain's API surface: some chain
    implementations accept `run(input=...)`, others expect a dict-like call
    returning a mapping. We try the common call patterns and raise a clear
    error if nothing works.
    """
    # Try the simplest "run" interface first
    try:
        return conversational_chain.run(input=user_query)
    except TypeError:
        # Some chains expect a dict/kwargs and return a dict with an answer key
        try:
            result = conversational_chain({"question": user_query})
            if isinstance(result, dict):
                # Common keys: 'answer', 'output_text'
                return result.get("answer") or result.get("output_text") or str(result)
            return str(result)
        except Exception as exc:
            raise RuntimeError(f"Conversational chain failed: {exc}") from exc


