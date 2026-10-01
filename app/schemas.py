"""Pydantic request and response models."""

from uuid import UUID

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    document_id: UUID
    filename: str
    chunk_count: int


class DocumentListItem(BaseModel):
    document_id: UUID
    filename: str
    uploaded_at: str
    chunk_count: int


class ChatRequest(BaseModel):
    document_id: UUID
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    page_number: int
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
