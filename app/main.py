"""FastAPI application and its four public endpoints."""

from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, status

from app import db
from app.documents import DocumentValidationError, extract_pages, split_pages, validate_stream_size
from app.gemini import embed_text
from app.rag import answer_question
from app.schemas import ChatRequest, ChatResponse, DocumentListItem, DocumentResponse

@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(
    title="DocuMind",
    description="A simple RAG document Q&A API",
    lifespan=lifespan,
)


@app.post("/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def upload_document(file: UploadFile = File(...)) -> DocumentResponse:
    try:
        content = validate_stream_size(file.file)
        pages = extract_pages(file.filename, content)
        chunks = split_pages(pages)
        if not chunks:
            raise DocumentValidationError("The uploaded file has no readable text")
        for chunk in chunks:
            chunk["embedding"] = embed_text(str(chunk["chunk_text"]))
        document_id = uuid4()
        db.save_document(document_id, file.filename or "document", chunks)
    except DocumentValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return DocumentResponse(
        document_id=document_id,
        filename=file.filename or "document",
        chunk_count=len(chunks),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not db.document_exists(str(request.document_id)):
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        result = answer_question(request.document_id, request.question)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return ChatResponse(**result)


@app.get("/documents", response_model=list[DocumentListItem])
def get_documents() -> list[DocumentListItem]:
    return db.list_documents()


@app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: UUID) -> None:
    if not db.delete_document(str(document_id)):
        raise HTTPException(status_code=404, detail="Document not found")
