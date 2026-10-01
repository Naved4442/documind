# DocuMind

DocuMind is a small document question-answering API built to demonstrate Retrieval-Augmented Generation (RAG). Upload a PDF or UTF-8 TXT file, then ask questions whose answers should come from that document.

## What It Does

- Accepts PDF and TXT uploads up to 10 MB.
- Extracts PDF text page by page with `pypdf`.
- Splits text into roughly 800-character chunks with 100-character overlap.
- Creates Gemini embeddings and stores them in PostgreSQL with pgvector.
- Retrieves the four most similar chunks for a question.
- Uses a similarity threshold before calling Gemini for an answer.
- Returns answer sources with page numbers and snippets.
- Lists and deletes uploaded documents.

## What It Intentionally Does Not Do

DocuMind does not have a frontend, user accounts, OCR for scanned PDFs, conversation history, streaming answers, or multi-document search. It is intentionally a focused learning and interview project.

## Tech Stack

- Python and FastAPI
- PostgreSQL with the `pgvector/pgvector` Docker image
- Gemini for embeddings and answer generation
- `pypdf` for PDFs
- `langchain-text-splitters` only for `RecursiveCharacterTextSplitter`
- pytest for tests
- Docker Compose for local services

## Architecture

The application uses short-lived PostgreSQL connections and plain SQL. The database schema is created when the API starts. Documents and chunks are connected by a foreign key with cascade deletion.

### Upload flow

```text
Client -> POST /documents
       -> validate file type and size
       -> extract pages with pypdf or decode TXT
       -> split each page into chunks
       -> create one Gemini embedding per chunk
       -> save document and chunks in PostgreSQL/pgvector
       -> return document_id and chunk_count
```

### Question-answering flow

```text
Client -> POST /chat
       -> verify document exists
       -> embed the question with Gemini
       -> pgvector cosine-distance search for top 4 chunks
       -> compare best similarity with SIMILARITY_THRESHOLD
       -> below threshold: return fallback without answer generation
       -> otherwise: send only retrieved chunks to Gemini
       -> return grounded answer and page/snippet sources
```

## Project Structure

```text
app/
  config.py       Environment settings and RAG constants
  db.py           Schema creation and parameterized PostgreSQL queries
  documents.py    Upload validation, extraction, and chunking
  gemini.py       Gemini embedding and generation adapter
  main.py         FastAPI app and public routes
  rag.py          Retrieval threshold and grounded prompt flow
  schemas.py      Pydantic request and response models
tests/
  test_api.py     FastAPI validation test
  test_documents.py  Chunking and file validation tests
  test_rag.py     Similarity threshold and source tests
Dockerfile
docker-compose.yml
requirements.txt
.env.example
```

## Environment Setup

Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. The `.env` file is ignored by Git.

```bash
cp .env.example .env
```

Variables:

- `GEMINI_API_KEY`: required for upload embeddings and answer generation.
- `DATABASE_URL`: used by the application; Compose sets the service URL automatically.
- `SIMILARITY_THRESHOLD`: default `0.35`; controls whether answer generation is allowed.
- `EMBEDDING_MODEL`: default `gemini-embedding-001`.
- `GENERATION_MODEL`: default `gemini-2.5-flash`.

The embedding output is configured to 768 dimensions, and the database vector column uses the same dimension.

## Run With Docker

Docker Compose starts both the API and pgvector database:

```bash
docker compose up --build
```

Open Swagger UI at <http://localhost:8000/docs>. PostgreSQL is available to the API inside the Compose network and is not published to the host. Stop the services with `Ctrl+C`.

## Using `/docs`

1. Open `/docs`.
2. Use `POST /documents` and upload a PDF or TXT file.
3. Copy the returned `document_id`.
4. Use `POST /chat` with that ID and a question.
5. Use `GET /documents` to list uploads.
6. Use `DELETE /documents/{document_id}` to remove a document and its chunks.

## Example Requests

Upload a text file:

```bash
curl -X POST http://localhost:8000/documents \
  -F 'file=@notes.txt'
```

Ask a question:

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"document_id":"11111111-1111-1111-1111-111111111111","question":"What is the main topic?"}'
```

List documents:

```bash
curl http://localhost:8000/documents
```

Delete a document:

```bash
curl -X DELETE http://localhost:8000/documents/11111111-1111-1111-1111-111111111111
```

### Example Responses

Upload:

```json
{"document_id":"11111111-1111-1111-1111-111111111111","filename":"notes.txt","chunk_count":2}
```

Chat with an answer:

```json
{"answer":"The main topic is document retrieval.","sources":[{"page_number":1,"snippet":"This document explains document retrieval."}]}
```

Chat below the threshold:

```json
{"answer":"I couldn't find this in the document","sources":[]}
```

## How RAG Works Here

RAG means Retrieval-Augmented Generation. Instead of asking a language model to answer from memory, DocuMind first retrieves relevant pieces of the uploaded document and gives only those pieces to the model.

An embedding is a list of numbers representing the meaning of text. Text with similar meaning tends to have vectors that point in similar directions. pgvector stores those vectors and supports similarity search inside PostgreSQL.

The query uses cosine distance (`<=>`). DocuMind converts it to similarity with `1 - distance`, orders the chunks by nearest vector, and keeps the top four. Top-k retrieval gives the answer model enough relevant context without sending an entire document.

The similarity threshold is a safety gate. If the best retrieved chunk is below `SIMILARITY_THRESHOLD`, DocuMind returns `I couldn't find this in the document` and does not call Gemini to generate an answer. The prompt also explicitly says to use only the retrieved context. These controls reduce, but do not eliminate, hallucinations.

Chunking makes long documents searchable in manageable pieces. The 100-character overlap helps preserve meaning when a sentence or idea crosses a chunk boundary.

## Testing

Install the dependencies and run:

```bash
python -m pytest -q
```

Tests cover chunk size and overlap, unsupported files, the 10 MB limit, TXT extraction, and the low-similarity path where answer generation must not run. Gemini and retrieval are passed as functions in the RAG service, so tests can use fakes and never need a real API key.

## Known Limitations

- Scanned PDFs without an embedded text layer are rejected as having no readable text.
- There is no authentication or per-user document ownership.
- Each request opens a short-lived database connection; a production service would normally use pooling.
- Embeddings and answer generation add network latency and cost.
- A fixed chunk size, top-k value, and threshold may need tuning for different document types.
- The answer model can still make mistakes even with grounded context.
- Compose uses PostgreSQL trust authentication for this local-only setup; production deployments should use managed credentials and restricted networking.
