# test_adapter.py

from functools import lru_cache
import os
from pathlib import Path

from app import PDF_DIR, RAGSystem, answer_question, load_pdf_chunks


@lru_cache(maxsize=1)
def get_rag() -> RAGSystem:
    """
    Build the RAG index once and reuse it for all tests.
    """
    chunks, _errors = load_pdf_chunks(Path(PDF_DIR))
    return RAGSystem(chunks)


def retrieve(question: str, top_k: int = 5):
    """
    Test only the retrieval component of the RAG system.
    """

    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    rag = get_rag()

    return rag.retrieve(
        question,
        top_k=top_k
    )


def ask_chatbot(question: str, top_k: int = 5) -> str:
    """
    Run the complete RAG pipeline:
    retrieval + answer generation.
    """

    if isinstance(question, str) and not question.strip():
        return "Please enter a question."

    results = retrieve(
        question,
        top_k=top_k
    )

    answer = answer_question(
        question,
        results,
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    return answer