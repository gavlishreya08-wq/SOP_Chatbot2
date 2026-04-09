import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.chat import router as chat_router


class ChatApiTests(unittest.TestCase):
    def build_client(self, rag_chain, rag_chains=None):
        app = FastAPI()
        app.include_router(chat_router)
        app.state.rag_chain = rag_chain
        app.state.vectorstore = None
        if rag_chains is not None:
            app.state.rag_chains = rag_chains
        return TestClient(app)

    def test_non_stream_chat_returns_json(self):
        class FakeChain:
            async def query(self, message, history, active_sop, **kwargs):
                return {
                    "answer": "Answer text",
                    "sources": None,
                    "followup": None,
                    "active_sop": None,
                    "image": None,
                    "confidence": "high",
                    "suggestions": None,
                }

        client = self.build_client(FakeChain())
        response = client.post(
            "/api/chat",
            json={"message": "hello", "history": [], "active_sop": None, "stream": False},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Answer text")
        self.assertEqual(response.json()["confidence"], "high")

    def test_stream_chat_returns_sse_events(self):
        class FakeChain:
            async def stream_query(self, message, history, active_sop, **kwargs):
                yield {"type": "token", "content": "Hello"}
                yield {"type": "done", "sources": None, "followup": None, "active_sop": None, "image": None}

        client = self.build_client(FakeChain())
        response = client.post(
            "/api/chat",
            json={"message": "hello", "history": [], "active_sop": None, "stream": True},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn('"type": "token"', response.text)
        self.assertIn('"type": "done"', response.text)

    def test_stream_chat_converts_exception_to_error_event(self):
        class FakeChain:
            async def stream_query(self, message, history, active_sop, **kwargs):
                raise RuntimeError("boom")
                yield  # pragma: no cover

        client = self.build_client(FakeChain())
        response = client.post(
            "/api/chat",
            json={"message": "hello", "history": [], "active_sop": None, "stream": True},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn('"type": "error"', response.text)
        self.assertIn("boom", response.text)

    def test_non_stream_chat_uses_requested_model_chain(self):
        class DefaultChain:
            async def query(self, message, history, active_sop, **kwargs):
                return {
                    "answer": "default",
                    "sources": None,
                    "followup": None,
                    "active_sop": None,
                    "image": None,
                    "confidence": "medium",
                    "suggestions": None,
                }

        class GroqChain:
            async def query(self, message, history, active_sop, **kwargs):
                return {
                    "answer": "groq-selected",
                    "sources": None,
                    "followup": None,
                    "active_sop": None,
                    "image": None,
                    "confidence": "high",
                    "suggestions": None,
                }

        client = self.build_client(
            DefaultChain(),
            rag_chains={"groq": GroqChain()},
        )
        response = client.post(
            "/api/chat",
            json={
                "message": "hello",
                "history": [],
                "active_sop": None,
                "stream": False,
                "llm_provider": "groq",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "groq-selected")

    def test_answer_mode_passed_through(self):
        class FakeChain:
            def __init__(self):
                self.received_mode = None

            async def query(self, message, history, active_sop, **kwargs):
                self.received_mode = kwargs.get("answer_mode")
                return {
                    "answer": f"Mode: {self.received_mode}",
                    "sources": None,
                    "followup": None,
                    "active_sop": None,
                    "image": None,
                    "confidence": "high",
                    "suggestions": None,
                }

        chain = FakeChain()
        client = self.build_client(chain)
        response = client.post(
            "/api/chat",
            json={
                "message": "hello",
                "history": [],
                "active_sop": None,
                "stream": False,
                "answer_mode": "brief",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Mode: brief")

    def test_source_locked_passed_through(self):
        class FakeChain:
            def __init__(self):
                self.received_locked = None

            async def query(self, message, history, active_sop, **kwargs):
                self.received_locked = kwargs.get("source_locked")
                return {
                    "answer": f"Locked: {self.received_locked}",
                    "sources": None,
                    "followup": None,
                    "active_sop": None,
                    "image": None,
                    "confidence": "high",
                    "suggestions": None,
                }

        chain = FakeChain()
        client = self.build_client(chain)
        response = client.post(
            "/api/chat",
            json={
                "message": "hello",
                "history": [],
                "active_sop": None,
                "stream": False,
                "source_locked": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Locked: True")

    def test_feedback_endpoint(self):
        app = FastAPI()
        app.include_router(chat_router)
        app.state.rag_chain = None
        client = TestClient(app)

        response = client.post(
            "/api/feedback",
            json={
                "question": "test question",
                "answer": "test answer",
                "rating": "up",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])

    def test_sops_endpoint_when_not_ready(self):
        app = FastAPI()
        app.include_router(chat_router)
        app.state.rag_chain = None
        client = TestClient(app)

        response = client.get("/api/sops")
        self.assertEqual(response.status_code, 503)

    def test_provider_status_endpoint(self):
        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)

        response = client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("gemini", data)
        self.assertIn("groq", data)


if __name__ == "__main__":
    unittest.main()
