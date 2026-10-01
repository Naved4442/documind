"""Application settings loaded from environment variables."""

import os


MAX_FILE_SIZE = 10 * 1024 * 1024
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4
EMBEDDING_DIMENSION = 768
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://documind@db:5432/documind")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")
