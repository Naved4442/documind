"""Small PostgreSQL access layer using parameterized SQL."""

from collections.abc import Sequence
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.config import DATABASE_URL, EMBEDDING_DIMENSION

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    filename TEXT NOT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector({EMBEDDING_DIMENSION}) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);
"""


def get_connection() -> psycopg.Connection:
    """Open one short-lived connection; simple enough for this project."""
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(SCHEMA_SQL)
        connection.commit()


def document_exists(document_id: str) -> bool:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM documents WHERE id = %s", (document_id,)
        ).fetchone()
    return row is not None


def save_document(document_id: str, filename: str, chunks: Sequence[dict[str, Any]]) -> None:
    with get_connection() as connection:
        with connection.transaction():
            connection.execute(
                "INSERT INTO documents (id, filename) VALUES (%s, %s)",
                (document_id, filename),
            )
            for chunk in chunks:
                connection.execute(
                    """
                    INSERT INTO chunks (document_id, page_number, chunk_text, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    """,
                    (
                        document_id,
                        chunk["page_number"],
                        chunk["chunk_text"],
                        vector_literal(chunk["embedding"]),
                    ),
                )


def list_documents() -> list[dict[str, Any]]:
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT d.id AS document_id, d.filename, d.uploaded_at::text,
                   COUNT(c.id)::int AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.uploaded_at DESC
            """
        ).fetchall()


def delete_document(document_id: str) -> bool:
    with get_connection() as connection:
        cursor = connection.execute("DELETE FROM documents WHERE id = %s", (document_id,))
        connection.commit()
    return cursor.rowcount > 0


def search_chunks(document_id: str, embedding: list[float], limit: int) -> list[dict[str, Any]]:
    literal = vector_literal(embedding)
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT page_number, chunk_text,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            WHERE document_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (literal, document_id, literal, limit),
        ).fetchall()


def vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"
