"""Gemini calls kept in one module so they are easy to mock in tests."""

from google import genai
from google.genai import types

from app.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL, GENERATION_MODEL, GEMINI_API_KEY


def get_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    return genai.Client(api_key=GEMINI_API_KEY)


def embed_text(text: str) -> list[float]:
    response = get_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION),
    )
    return list(response.embeddings[0].values)


def generate_answer(prompt: str) -> str:
    response = get_client().models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    return response.text.strip()
