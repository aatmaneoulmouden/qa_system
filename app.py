from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader


ROOT_DIR = Path(__file__).resolve().parent
PDF_DIR = ROOT_DIR / "PDF Files"
DEFAULT_MODEL = "gpt-4o-mini"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*", re.IGNORECASE)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "how", "i", "in", "is", "it", "its", "of",
    "on", "or", "our", "that", "the", "their", "them", "there", "these", "this",
    "to", "was", "we", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "you", "your",
}


@dataclass(frozen=True)
class Chunk:
    source: str
    page: int
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source}, p. {self.page}"


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token.lower() not in STOP_WORDS
    ]


def chunk_text(
    text: str,
    source: str,
    page: int,
    chunk_size: int = 180,
    overlap: int = 35,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    words = text.split()
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(Chunk(source=source, page=page, text=chunk))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def load_pdf_chunks(pdf_dir: Path) -> tuple[list[Chunk], list[str]]:
    chunks: list[Chunk] = []
    errors: list[str] = []
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(str(pdf_path))
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                chunks.extend(
                    chunk_text(page_text, pdf_path.name, page_number)
                )
        except Exception as error:
            errors.append(f"{pdf_path.name}: {error}")

    return chunks, errors


class RAGSystem:
    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks = list(chunks)
        self.document_frequencies: Counter[str] = Counter()
        self.document_vectors: list[dict[str, float]] = []

        tokenized_documents = [tokenize(chunk.text) for chunk in self.chunks]
        for tokens in tokenized_documents:
            self.document_frequencies.update(set(tokens))

        document_count = len(self.chunks)
        for tokens in tokenized_documents:
            counts = Counter(tokens)
            vector = {
                token: (1 + math.log(frequency))
                * (math.log((document_count + 1) / (self.document_frequencies[token] + 1)) + 1)
                for token, frequency in counts.items()
            }
            self.document_vectors.append(self._normalize(vector))

    @staticmethod
    def _normalize(vector: dict[str, float]) -> dict[str, float]:
        magnitude = math.sqrt(sum(weight * weight for weight in vector.values()))
        if magnitude == 0:
            return vector
        return {token: weight / magnitude for token, weight in vector.items()}

    def retrieve(self, question: str, top_k: int = 4) -> list[tuple[Chunk, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_tokens = set(tokenize(question))
        if not query_tokens or not self.chunks:
            return []

        document_count = len(self.chunks)
        query_vector = {
            token: math.log((document_count + 1) / (self.document_frequencies[token] + 1)) + 1
            for token in query_tokens
            if token in self.document_frequencies
        }
        query_vector = self._normalize(query_vector)
        if not query_vector:
            return []

        scored = []
        for chunk, document_vector in zip(self.chunks, self.document_vectors):
            score = sum(
                query_weight * document_vector.get(token, 0.0)
                for token, query_weight in query_vector.items()
            )
            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda result: result[1], reverse=True)
        return scored[:top_k]


class LocalRAG:
    def __init__(self, pdf_dir: str | Path = PDF_DIR):
        self.pdf_dir = Path(pdf_dir)
        self.chunks: list[Chunk] = []
        self.errors: list[str] = []
        self.system: RAGSystem | None = None

    def build(self) -> int:
        self.chunks, self.errors = load_pdf_chunks(self.pdf_dir)
        self.system = RAGSystem(self.chunks)
        return len(self.chunks)

    def search(self, question: str, top_k: int = 4) -> list[tuple[Chunk, float]]:
        if self.system is None:
            self.build()
        return self.system.retrieve(question, top_k=top_k)


def extractive_answer(matches: Sequence[tuple[Chunk, float]], question: str) -> str:
    if not matches:
        return "I couldn't find relevant information in the loaded PDFs. Try rephrasing your question or adding a relevant document."

    question_terms = set(tokenize(question))
    selected: list[tuple[str, Chunk, int]] = []
    seen_sentences: set[str] = set()
    for chunk, _score in matches:
        for sentence in SENTENCE_PATTERN.split(chunk.text):
            sentence = sentence.strip()
            if not sentence:
                continue
            normalized = sentence.lower()
            if normalized in seen_sentences:
                continue
            seen_sentences.add(normalized)
            overlap_count = len(question_terms.intersection(tokenize(sentence)))
            if overlap_count >= 3:
                selected.append((sentence, chunk, overlap_count))

    if not selected:
        return "I couldn't find relevant information in the loaded PDFs. Try rephrasing your question or adding a relevant document."

    selected.sort(key=lambda item: item[2], reverse=True)
    bullets = [
        f"- {sentence} [{chunk.citation}]"
        for sentence, chunk, _overlap in selected[:3]
    ]
    return "Based on the retrieved passages:\n\n" + "\n".join(bullets)


def answer_question(
    question: str,
    matches: Sequence[tuple[Chunk, float]],
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    if not matches:
        return extractive_answer(matches, question)
    if not api_key:
        return extractive_answer(matches, question)

    context = "\n\n".join(
        f"[{chunk.citation}]\n{chunk.text}" for chunk, _score in matches
    )
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question using only the supplied document passages. "
                        "If the passages do not contain the answer, say so. Cite factual claims "
                        "with the source labels exactly as provided, for example [report.pdf, p. 2]."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nDocument passages:\n{context}",
                },
            ],
        )
        content = response.choices[0].message.content
        if content:
            return content.strip()
    except Exception:
        pass

    return extractive_answer(matches, question)


@st.cache_resource(show_spinner="Reading and indexing PDFs...")
def get_rag_system(pdf_dir: str) -> tuple[RAGSystem, list[str]]:
    chunks, errors = load_pdf_chunks(Path(pdf_dir))
    return RAGSystem(chunks), errors


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    st.set_page_config(page_title="Document QA", page_icon="⌕", layout="wide")
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    rag, errors = get_rag_system(str(PDF_DIR))
    st.markdown(
        """
        <style>
        :root {
            --qa-paper: #f6f5ef;
            --qa-panel: #e8eee8;
            --qa-ink: #202923;
            --qa-muted: #66736b;
            --qa-line: #d6ded6;
            --qa-accent: #a84f3c;
        }
        .stApp { background: var(--qa-paper); color: var(--qa-ink); }
        header[data-testid="stHeader"] { background: rgba(246, 245, 239, 0.94); }
        section[data-testid="stSidebar"] { background: var(--qa-panel); border-right: 1px solid var(--qa-line); }
        .block-container { max-width: 920px; padding-top: 2.6rem; padding-bottom: 6rem; }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: var(--qa-ink); }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stMetricLabel"],
        [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: var(--qa-ink); }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--qa-muted); }
        [data-testid="stMetric"] { padding: 0.25rem 0 0.7rem; }
        [data-testid="stMetricValue"] { color: var(--qa-ink); }
        [data-testid="stChatMessage"] {
            background: #fff;
            border: 1px solid var(--qa-line);
            border-radius: 8px;
            padding: 1rem 1.15rem;
            margin-bottom: 0.8rem;
        }
        [data-testid="stChatMessage"] p { color: var(--qa-ink); }
        [data-testid="stChatInput"] {
            background: #fff;
            border: 1px solid #cbd5cb;
            border-radius: 8px;
            box-shadow: 0 4px 18px rgba(32, 41, 35, 0.08);
        }
        [data-testid="stChatInput"] textarea { color: var(--qa-ink); }
        .qa-kicker { color: var(--qa-accent); font-size: 0.76rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
        h1 { color: var(--qa-ink); font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif; font-size: 2.55rem; font-weight: 600; margin: 0.2rem 0 0.1rem; }
        .qa-deck { color: var(--qa-muted); margin: 0 0 1.8rem; }
        .qa-empty { border-top: 1px solid var(--qa-line); margin-top: 2.4rem; padding-top: 1.6rem; }
        .qa-empty-label { color: var(--qa-accent); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
        .qa-empty-title { color: var(--qa-ink); font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif; font-size: 1.55rem; font-weight: 600; margin: 0.35rem 0 0.35rem; }
        .qa-empty-copy { color: var(--qa-muted); margin: 0 0 1.25rem; }
        [data-testid="stMainBlockContainer"] [data-testid="stButton"] button {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--qa-line);
            border-radius: 8px;
            color: var(--qa-ink);
            min-height: 3.5rem;
            height: auto;
            text-align: left;
            white-space: normal;
        }
        [data-testid="stMainBlockContainer"] [data-testid="stButton"] button p {
            display: block !important;
            overflow: visible !important;
            text-overflow: clip !important;
            white-space: normal !important;
            -webkit-line-clamp: unset !important;
        }
        [data-testid="stMainBlockContainer"] [data-testid="stButton"] button:hover { border-color: var(--qa-accent); color: var(--qa-accent); }
        @media (max-width: 900px) {
            section[data-testid="stSidebar"] { width: 15rem !important; min-width: 15rem !important; }
        }
        @media (max-width: 700px) {
            .block-container { padding: 1.6rem 1rem 5rem; }
            h1 { font-size: 2rem; }
            .qa-empty { margin-top: 1.6rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="qa-kicker">PDF knowledge base</div>', unsafe_allow_html=True)
    st.title("Ask your documents")
    st.markdown(
        f'<p class="qa-deck">{len(pdf_files)} reports · {len(rag.chunks):,} passages indexed</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown('<div class="qa-kicker">Library</div>', unsafe_allow_html=True)
        st.subheader("Knowledge base")
        st.metric("Indexed passages", f"{len(rag.chunks):,}")
        st.caption(f"{len(pdf_files)} PDF reports in {PDF_DIR.name}/")
        with st.expander("View source documents"):
            for pdf_path in pdf_files:
                st.markdown(f"- {pdf_path.name}")
        st.divider()
        st.markdown('<div class="qa-kicker">Answer settings</div>', unsafe_allow_html=True)
        top_k = st.slider("Retrieved passages", min_value=1, max_value=8, value=4)
        if st.button("Reindex PDFs", icon=":material/refresh:", use_container_width=True):
            get_rag_system.clear()
            st.rerun()
        if os.getenv("OPENAI_API_KEY"):
            st.caption("Generation · OpenAI")
        else:
            st.caption("Generation · local extractive")

    for error in errors:
        st.warning(f"Could not read a PDF: {error}")
    if not rag.chunks:
        st.info("No readable PDF text found. Add text-based PDFs to the PDF Files folder and reindex.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown(
            '<section class="qa-empty">'
            '<div class="qa-empty-label">Start with a question</div>'
            '<div class="qa-empty-title">What would you like to know?</div>'
            '<p class="qa-empty-copy">Explore findings across the reports in your library.</p>'
            '</section>',
            unsafe_allow_html=True,
        )
        suggested_questions = [
            "How does AI affect developer productivity?",
            "How are teams using AI coding tools?",
            "What challenges are reported for enterprise AI adoption?",
        ]
        for index, prompt in enumerate(suggested_questions):
            if st.button(prompt, key=f"suggested-question-{index}", use_container_width=True):
                st.session_state.pending_question = prompt
                st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Retrieved passages"):
                    for source in message["sources"]:
                        st.markdown(f"**{source.citation}**")
                        st.write(source.text)

    question = st.chat_input(
        "Ask a question about the PDFs",
        disabled=not rag.chunks,
    )
    if question is None:
        question = st.session_state.pop("pending_question", None)
    else:
        st.session_state.pop("pending_question", None)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        matches = rag.retrieve(question, top_k=top_k)
        response = answer_question(
            question,
            matches,
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "sources": [chunk for chunk, _score in matches],
            }
        )
        st.rerun()


if __name__ == "__main__":
    main()