from uuid import uuid4

from app.rag import FALLBACK_ANSWER, answer_question


def test_low_similarity_does_not_generate_answer() -> None:
    generated = False

    def generator(prompt: str) -> str:
        nonlocal generated
        generated = True
        return "should not be used"

    result = answer_question(
        uuid4(),
        "What is this?",
        embedder=lambda text: [0.1, 0.2],
        retriever=lambda document_id, embedding, limit: [
            {"page_number": 2, "chunk_text": "unrelated", "similarity": 0.2}
        ],
        answer_generator=generator,
        threshold=0.35,
    )

    assert result == {"answer": FALLBACK_ANSWER, "sources": []}
    assert generated is False


def test_high_similarity_returns_answer_and_sources() -> None:
    result = answer_question(
        uuid4(),
        "What is this?",
        embedder=lambda text: [0.1, 0.2],
        retriever=lambda document_id, embedding, limit: [
            {"page_number": 3, "chunk_text": "The answer is here.", "similarity": 0.9}
        ],
        answer_generator=lambda prompt: "The answer is here.",
    )
    assert result["answer"] == "The answer is here."
    assert result["sources"] == [{"page_number": 3, "snippet": "The answer is here."}]
