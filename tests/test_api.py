from fastapi.testclient import TestClient

from app.main import app


class FakeDatabase:
    def init_db(self):
        pass


# Validation happens before Gemini and database calls, so this test needs no services.
def test_api_rejects_unsupported_upload(monkeypatch):
    monkeypatch.setattr("app.main.db", FakeDatabase())
    with TestClient(app) as client:
        response = client.post(
            "/documents", files={"file": ("notes.docx", b"text", "application/octet-stream")}
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF and TXT files are supported"
