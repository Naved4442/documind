"""Document validation, text extraction, and chunk preparation."""

from io import BytesIO
from pathlib import PurePath
from typing import BinaryIO

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_FILE_SIZE


class DocumentValidationError(ValueError):
    """Raised when an uploaded document cannot be accepted."""


def validate_upload(filename: str | None, content: bytes) -> str:
    suffix = PurePath(filename or "").suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise DocumentValidationError("Only PDF and TXT files are supported")
    if len(content) > MAX_FILE_SIZE:
        raise DocumentValidationError("File size must not exceed 10 MB")
    if not content:
        raise DocumentValidationError("The uploaded file is empty")
    return suffix


def extract_pages(filename: str | None, content: bytes) -> list[tuple[int, str]]:
    suffix = validate_upload(filename, content)
    if suffix == ".txt":
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise DocumentValidationError("TXT files must be valid UTF-8") from error
        if not text.strip():
            raise DocumentValidationError("The uploaded file has no readable text")
        return [(1, text)]

    try:
        reader = PdfReader(BytesIO(content))
        pages = [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, 1)]
    except Exception as error:
        raise DocumentValidationError("The PDF file is invalid or unreadable") from error
    if not any(text.strip() for _, text in pages):
        raise DocumentValidationError("The PDF has no readable text")
    return pages


def split_pages(pages: list[tuple[int, str]]) -> list[dict[str, object]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks: list[dict[str, object]] = []
    for page_number, text in pages:
        for chunk_text in splitter.split_text(text):
            if chunk_text.strip():
                chunks.append({"page_number": page_number, "chunk_text": chunk_text})
    return chunks


def validate_stream_size(file: BinaryIO) -> bytes:
    content = file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise DocumentValidationError("File size must not exceed 10 MB")
    return content
