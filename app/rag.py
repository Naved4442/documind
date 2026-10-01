"""The retrieval and grounded answer flow."""

from collections.abc import Callable
from uuid import UUID

from app.config import SIMILARITY_THRESHOLD, TOP_K
from app.db import search_chunks
from app.gemini import embed_text, generate_answer

FALLBACK_ANSWER = "I couldn't find this in the document"


def answer_question(
    document_id: UUID,
    question: str,
    *,
    embedder: Callable[[str], list[float]] = embed_text,
    answer_generator: Callable[[str], str] = generate_answer,
    retriever: Callable[[str, list[float], int], list[dict]] = search_chunks,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    question_embedding = embedder(question)
    matches = retriever(str(document_id), question_embedding, TOP_K)
    if not matches or float(matches[0]["similarity"]) < threshold:
        return {"answer": FALLBACK_ANSWER, "sources": []}

    context = "\n\n".join(
        f"[Page {match['page_number']}]\n{match['chunk_text']}" for match in matches
    )
    prompt = (
        "Answer the question using ONLY the document context below. "
        "If the context does not contain the answer, say you cannot find it in the document. "
        "Do not use outside knowledge.\n\n"
        f"Document context:\n{context}\n\nQuestion: {question}"
    )
    answer = answer_generator(prompt)
    sources = [
        {"page_number": match["page_number"], "snippet": match["chunk_text"][:200]}
        for match in matches
    ]
    return {"answer": answer, "sources": sources}
