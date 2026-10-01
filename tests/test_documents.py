from io import BytesIO

import pytest

from app.documents import DocumentValidationError, extract_pages, split_pages, validate_stream_size


def test_chunking_has_overlap_and_target_size() -> None:
    chunks = split_pages([(1, "abcdefghij" * 120)])
    assert len(chunks) > 1
    assert all(len(chunk["chunk_text"]) <= 800 for chunk in chunks)
    assert chunks[0]["chunk_text"][-100:] == chunks[1]["chunk_text"][:100]


def test_unsupported_file_type() -> None:
    with pytest.raises(DocumentValidationError, match="Only PDF and TXT"):
        extract_pages("notes.docx", b"text")


def test_file_larger_than_limit() -> None:
    with pytest.raises(DocumentValidationError, match="10 MB"):
        validate_stream_size(BytesIO(b"x" * (10 * 1024 * 1024 + 1)))


def test_txt_is_one_logical_page() -> None:
    assert extract_pages("notes.txt", b"hello\nworld") == [(1, "hello\nworld")]


def test_empty_txt_is_rejected() -> None:
    with pytest.raises(DocumentValidationError, match="no readable text"):
        extract_pages("notes.txt", b"   ")
