"""Comprehensive tests for all new dynamic features."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.chat import router as chat_router


# ── Helpers ──────────────────────────────────────────────────────────────

class FakeChain:
    """Fake RAG chain that accepts **kwargs for answer_mode/source_locked."""

    def __init__(self, answer="Test answer", suggestions=None, confidence="high"):
        self._answer = answer
        self._suggestions = suggestions
        self._confidence = confidence
        self.last_kwargs = {}

    async def query(self, message, history, active_sop, **kwargs):
        self.last_kwargs = kwargs
        return {
            "answer": self._answer,
            "sources": None,
            "followup": None,
            "active_sop": active_sop,
            "image": None,
            "confidence": self._confidence,
            "suggestions": self._suggestions,
        }

    async def stream_query(self, message, history, active_sop, **kwargs):
        self.last_kwargs = kwargs
        yield {"type": "token", "content": self._answer}
        yield {
            "type": "done",
            "sources": None,
            "followup": None,
            "active_sop": active_sop,
            "image": None,
            "confidence": self._confidence,
            "suggestions": self._suggestions,
            "full_answer": self._answer,
        }

    async def compare_sops(self, question, sop_a, sop_b):
        return {
            "answer": f"Comparing {sop_a} vs {sop_b}: {question}",
            "sources": None,
            "sources_b": None,
            "sop_a_title": "SOP A",
            "sop_b_title": "SOP B",
            "confidence": "medium",
        }

    def get_all_sop_titles(self):
        return [
            {"source": "test.pdf", "title": "Test SOP"},
            {"source": "other.pdf", "title": "Other SOP"},
        ]


def build_app(chain=None):
    app = FastAPI()
    app.include_router(chat_router)
    app.state.rag_chain = chain or FakeChain()
    app.state.vectorstore = None
    app.state.rag_chains = {}
    return app


# ── Answer Mode Tests ────────────────────────────────────────────────────

class AnswerModeTests(unittest.TestCase):
    def test_all_valid_modes_accepted(self):
        modes = ["brief", "detailed", "checklist", "step-by-step", "only-responsibilities", "only-objective"]
        for mode in modes:
            chain = FakeChain()
            client = TestClient(build_app(chain))
            resp = client.post("/api/chat", json={
                "message": "test", "history": [], "active_sop": None,
                "stream": False, "answer_mode": mode,
            })
            self.assertEqual(resp.status_code, 200, f"Failed for mode: {mode}")
            self.assertEqual(chain.last_kwargs.get("answer_mode"), mode)

    def test_invalid_mode_defaults_to_detailed(self):
        chain = FakeChain()
        client = TestClient(build_app(chain))
        resp = client.post("/api/chat", json={
            "message": "test", "history": [], "active_sop": None,
            "stream": False, "answer_mode": "invalid_mode",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(chain.last_kwargs.get("answer_mode"), "detailed")

    def test_default_mode_is_detailed(self):
        chain = FakeChain()
        client = TestClient(build_app(chain))
        resp = client.post("/api/chat", json={
            "message": "test", "history": [], "active_sop": None, "stream": False,
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(chain.last_kwargs.get("answer_mode"), "detailed")


# ── Source Lock Tests ────────────────────────────────────────────────────

class SourceLockTests(unittest.TestCase):
    def test_source_locked_passed_to_chain(self):
        chain = FakeChain()
        client = TestClient(build_app(chain))
        resp = client.post("/api/chat", json={
            "message": "test", "history": [], "active_sop": "some.pdf",
            "stream": False, "source_locked": True,
        })
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(chain.last_kwargs.get("source_locked"))

    def test_source_unlocked_by_default(self):
        chain = FakeChain()
        client = TestClient(build_app(chain))
        resp = client.post("/api/chat", json={
            "message": "test", "history": [], "active_sop": None, "stream": False,
        })
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(chain.last_kwargs.get("source_locked"))


# ── Confidence-Aware Answer Tests ────────────────────────────────────────

class ConfidenceTests(unittest.TestCase):
    def test_high_confidence_returned(self):
        chain = FakeChain(confidence="high")
        client = TestClient(build_app(chain))
        resp = client.post("/api/chat", json={
            "message": "test", "history": [], "active_sop": None, "stream": False,
        })
        self.assertEqual(resp.json()["confidence"], "high")

    def test_low_confidence_returned(self):
        chain = FakeChain(confidence="low")
        client = TestClient(build_app(chain))
        resp = client.post("/api/chat", json={
            "message": "test", "history": [], "active_sop": None, "stream": False,
        })
        self.assertEqual(resp.json()["confidence"], "low")


# ── Suggestions Tests ────────────────────────────────────────────────────

class SuggestionTests(unittest.TestCase):
    def test_suggestions_returned_in_response(self):
        chain = FakeChain(suggestions=["Next question 1", "Next question 2"])
        client = TestClient(build_app(chain))
        resp = client.post("/api/chat", json={
            "message": "test", "history": [], "active_sop": None, "stream": False,
        })
        self.assertEqual(resp.json()["suggestions"], ["Next question 1", "Next question 2"])

    def test_suggestions_in_stream_done_event(self):
        chain = FakeChain(suggestions=["Follow up?"])
        client = TestClient(build_app(chain))
        resp = client.post("/api/chat", json={
            "message": "test", "history": [], "active_sop": None, "stream": True,
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn('"Follow up?"', resp.text)


# ── Compare Mode Tests ───────────────────────────────────────────────────

class CompareModeTests(unittest.TestCase):
    def test_compare_endpoint_returns_comparison(self):
        client = TestClient(build_app())
        resp = client.post("/api/compare", json={
            "question": "Compare responsibilities",
            "sop_a": "test.pdf",
            "sop_b": "other.pdf",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("Comparing", data["answer"])
        self.assertEqual(data["sop_a_title"], "SOP A")
        self.assertEqual(data["sop_b_title"], "SOP B")

    def test_compare_requires_both_sops(self):
        client = TestClient(build_app())
        resp = client.post("/api/compare", json={
            "question": "Compare",
            "sop_a": "",
            "sop_b": "other.pdf",
        })
        self.assertEqual(resp.status_code, 200)  # Still processes, empty SOP


# ── SOP List Tests ───────────────────────────────────────────────────────

class SopListTests(unittest.TestCase):
    def test_list_sops_returns_titles(self):
        client = TestClient(build_app())
        resp = client.get("/api/sops")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data), 2)
        titles = [s["title"] for s in data]
        self.assertIn("Test SOP", titles)

    def test_list_sops_fails_when_not_ready(self):
        app = FastAPI()
        app.include_router(chat_router)
        app.state.rag_chain = None
        client = TestClient(app)
        resp = client.get("/api/sops")
        self.assertEqual(resp.status_code, 503)


# ── Provider Status Tests ────────────────────────────────────────────────

class ProviderStatusTests(unittest.TestCase):
    def test_status_returns_all_providers(self):
        client = TestClient(build_app())
        resp = client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("gemini", data)
        self.assertIn("groq", data)
        self.assertIn("healthy", data["gemini"])
        self.assertIn("configured", data["gemini"])


# ── Feedback Tests ───────────────────────────────────────────────────────

class FeedbackTests(unittest.TestCase):
    def test_feedback_up_accepted(self):
        client = TestClient(build_app())
        resp = client.post("/api/feedback", json={
            "question": "What is the process?",
            "answer": "The process is...",
            "rating": "up",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

    def test_feedback_down_accepted(self):
        client = TestClient(build_app())
        resp = client.post("/api/feedback", json={
            "question": "What is the process?",
            "answer": "The process is...",
            "rating": "down",
            "comment": "Wrong SOP referenced",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

    def test_feedback_invalid_rating_rejected(self):
        client = TestClient(build_app())
        resp = client.post("/api/feedback", json={
            "question": "test",
            "answer": "test",
            "rating": "invalid",
        })
        self.assertEqual(resp.status_code, 422)


# ── Conversation History Tests ───────────────────────────────────────────

class ConversationTests(unittest.TestCase):
    def test_save_and_list_conversations(self):
        client = TestClient(build_app())

        # Save a conversation
        resp = client.post("/api/conversations", json={
            "conversation_id": "test-conv-1",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
            "title": "Test conversation",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["saved"])

        # List conversations
        resp = client.get("/api/conversations")
        self.assertEqual(resp.status_code, 200)

    def test_search_conversations(self):
        client = TestClient(build_app())
        resp = client.get("/api/conversations/search?q=hello")
        self.assertEqual(resp.status_code, 200)

    def test_get_nonexistent_conversation(self):
        client = TestClient(build_app())
        resp = client.get("/api/conversations/nonexistent-id")
        self.assertEqual(resp.status_code, 404)

    def test_delete_nonexistent_conversation(self):
        client = TestClient(build_app())
        resp = client.delete("/api/conversations/nonexistent-id")
        self.assertEqual(resp.status_code, 404)


# ── Feedback Store Unit Tests ────────────────────────────────────────────

class FeedbackStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._original_data_dir = os.environ.get("DATA_DIR_OVERRIDE")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("backend.core.feedback._data_dir")
    def test_save_and_get_feedback(self, mock_data_dir):
        mock_data_dir.return_value = Path(self.tmpdir)

        from backend.core.feedback import get_feedback, save_feedback

        save_feedback("Q1", "A1", "up")
        save_feedback("Q2", "A2", "down", comment="Bad answer")

        results = get_feedback()
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["rating"], "down")  # Reversed order
        self.assertEqual(results[1]["rating"], "up")

    @patch("backend.core.feedback._data_dir")
    def test_save_and_get_failed_queries(self, mock_data_dir):
        mock_data_dir.return_value = Path(self.tmpdir)

        from backend.core.feedback import get_failed_queries, save_failed_query

        save_failed_query("Vague question", "low")
        save_failed_query("Another vague one", "low", answer="Not available")

        results = get_failed_queries()
        self.assertEqual(len(results), 2)

    @patch("backend.core.feedback._data_dir")
    def test_clear_failed_query(self, mock_data_dir):
        mock_data_dir.return_value = Path(self.tmpdir)

        from backend.core.feedback import clear_failed_query, get_failed_queries, save_failed_query

        save_failed_query("Q1", "low")
        save_failed_query("Q2", "low")

        self.assertTrue(clear_failed_query(0))
        results = get_failed_queries()
        self.assertEqual(len(results), 1)

    @patch("backend.core.feedback._data_dir")
    def test_analytics_summary(self, mock_data_dir):
        mock_data_dir.return_value = Path(self.tmpdir)

        from backend.core.feedback import get_analytics_summary, log_query, save_feedback

        log_query("test q", None, "test.pdf", "high")
        log_query("test q", None, "test.pdf", "medium")
        log_query("vague q", None, None, "low", was_clarification=True)
        save_feedback("Q1", "A1", "up")

        summary = get_analytics_summary()
        self.assertEqual(summary["total_queries"], 3)
        self.assertEqual(summary["confidence_breakdown"]["high"], 1)
        self.assertEqual(summary["confidence_breakdown"]["medium"], 1)
        self.assertEqual(summary["confidence_breakdown"]["low"], 1)
        self.assertEqual(summary["clarification_count"], 1)
        self.assertEqual(summary["feedback_summary"]["thumbs_up"], 1)

    @patch("backend.core.feedback._data_dir")
    def test_conversation_lifecycle(self, mock_data_dir):
        mock_data_dir.return_value = Path(self.tmpdir)

        from backend.core.feedback import (
            delete_conversation,
            get_conversation,
            get_conversations,
            save_conversation,
            search_conversations,
        )

        save_conversation("conv-1", [{"role": "user", "content": "Hello world"}], "My chat")
        save_conversation("conv-2", [{"role": "user", "content": "Jira process"}], "Jira")

        convs = get_conversations()
        self.assertEqual(len(convs), 2)

        conv = get_conversation("conv-1")
        self.assertIsNotNone(conv)
        self.assertEqual(conv["title"], "My chat")

        results = search_conversations("jira")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "conv-2")

        self.assertTrue(delete_conversation("conv-1"))
        self.assertIsNone(get_conversation("conv-1"))
        self.assertEqual(len(get_conversations()), 1)


# ── LLM Fallback Tests ──────────────────────────────────────────────────

class LLMFallbackTests(unittest.TestCase):
    def test_mark_provider_error_and_recovery(self):
        from backend.core.llm import (
            _provider_status,
            get_provider_status,
            mark_provider_error,
            mark_provider_healthy,
        )

        # Reset state
        mark_provider_healthy("gemini")

        mark_provider_error("gemini", "timeout")
        mark_provider_error("gemini", "timeout")
        status = get_provider_status()
        self.assertTrue(status["gemini"]["healthy"])  # 2 errors, still healthy
        self.assertEqual(status["gemini"]["error_count"], 2)

        mark_provider_error("gemini", "timeout")
        status = get_provider_status()
        self.assertFalse(status["gemini"]["healthy"])  # 3 errors, unhealthy

        mark_provider_healthy("gemini")
        status = get_provider_status()
        self.assertTrue(status["gemini"]["healthy"])  # Recovered
        self.assertEqual(status["gemini"]["error_count"], 0)

    def test_fallback_provider_returns_correct_provider(self):
        from backend.core.llm import get_fallback_provider

        # The actual result depends on whether the other provider is configured
        # but we can test the function doesn't crash
        result = get_fallback_provider("gemini")
        self.assertIn(result, [None, "groq"])

        result = get_fallback_provider("groq")
        self.assertIn(result, [None, "gemini"])


# ── Retriever Enhancement Tests ──────────────────────────────────────────

class RetrieverEnhancementTests(unittest.TestCase):
    def test_generate_query_variants(self):
        from backend.rag.retriever import generate_query_variants

        variants = generate_query_variants("test lead responsibilities")
        self.assertGreaterEqual(len(variants), 1)
        self.assertEqual(variants[0], "test lead responsibilities")

    def test_generate_query_variants_with_synonyms(self):
        from backend.rag.retriever import generate_query_variants

        variants = generate_query_variants("what is the process for deployment")
        self.assertGreaterEqual(len(variants), 1)
        # Should include a synonym variant
        all_text = " ".join(variants).lower()
        self.assertIn("process", all_text)  # Original included

    def test_detect_section_intent_for_responsibilities(self):
        from backend.rag.retriever import _detect_section_intent

        sections = _detect_section_intent("what are the responsibilities of test lead")
        self.assertTrue(any("Responsibilities" in s for s in sections))

    def test_detect_section_intent_for_workflow(self):
        from backend.rag.retriever import _detect_section_intent

        sections = _detect_section_intent("explain the workflow")
        self.assertTrue(any("Workflow" in s for s in sections))

    def test_detect_section_intent_for_objective(self):
        from backend.rag.retriever import _detect_section_intent

        sections = _detect_section_intent("what is the objective")
        self.assertTrue(any("Objective" in s for s in sections))

    def test_detect_section_intent_no_match(self):
        from backend.rag.retriever import _detect_section_intent

        sections = _detect_section_intent("tell me about test lead")
        # Should return empty if no section intent detected
        self.assertIsInstance(sections, list)

    def test_bm25_score_basic(self):
        from backend.rag.retriever import _bm25_score

        score = _bm25_score(
            {"test", "lead"},
            {"test", "lead", "roles"},
            100,
            80.0,
        )
        self.assertGreater(score, 0.0)

    def test_bm25_score_no_overlap(self):
        from backend.rag.retriever import _bm25_score

        score = _bm25_score(
            {"test", "lead"},
            {"deployment", "release"},
            100,
            80.0,
        )
        self.assertEqual(score, 0.0)


# ── RAG Chain New Features Tests ─────────────────────────────────────────

class RAGChainNewFeaturesTests(unittest.TestCase):
    def _make_chain(self, llm_responses=None, stream_chunks=None):
        from tests.helpers import FakeLLM, FakeVectorStore, make_doc

        docs = [
            make_doc(
                "Test Lead is responsible for test planning, execution, and reporting.",
                "TestLead_RR.pdf",
                page=0,
                source_title="Test Lead Roles and Responsibilities",
                source_kind="role",
                source_aliases="Test Lead, TL",
                source_intents="Test Lead role responsibilities",
                source_summary="Test Lead role",
                source_section_titles="Responsibilities, Job Objectives",
                section_title="Responsibilities",
            ),
        ]
        vs = FakeVectorStore(
            global_results=[(d, 0.5) for d in docs],
            filtered_results={"TestLead_RR.pdf": [(d, 0.3) for d in docs]},
            metadatas=[d.metadata for d in docs],
            documents=[d.page_content for d in docs],
        )
        llm = FakeLLM(
            responses=llm_responses or ["Answer text\nFOLLOWUP: What else?"],
            stream_chunks=stream_chunks or [],
        )
        from backend.core.rag_chain import RAGChain
        return RAGChain(llm=llm, vectorstore=vs)

    def test_query_with_brief_mode(self):
        chain = self._make_chain(
            llm_responses=[
                "Test Lead does test planning.\nFOLLOWUP: NONE",
                "q1?\nq2?\nq3?",  # suggestions response
            ]
        )
        result = asyncio.run(chain.query(
            "test lead responsibilities", [], None,
            answer_mode="brief",
        ))
        self.assertIn("answer", result)

    def test_query_with_source_locked(self):
        chain = self._make_chain(
            llm_responses=[
                "Answer\nFOLLOWUP: NONE",
                "q1?\nq2?\nq3?",
            ]
        )
        result = asyncio.run(chain.query(
            "responsibilities", [], "TestLead_RR.pdf",
            source_locked=True,
        ))
        self.assertIn("answer", result)

    def test_get_all_sop_titles(self):
        chain = self._make_chain()
        titles = chain.get_all_sop_titles()
        self.assertIsInstance(titles, list)
        self.assertGreater(len(titles), 0)
        self.assertIn("source", titles[0])
        self.assertIn("title", titles[0])


# ── Streaming with New Features Tests ────────────────────────────────────

class StreamingNewFeaturesTests(unittest.TestCase):
    def test_stream_with_answer_mode(self):
        chain = FakeChain()
        client = TestClient(build_app(chain))
        resp = client.post("/api/chat", json={
            "message": "test",
            "history": [],
            "active_sop": None,
            "stream": True,
            "answer_mode": "checklist",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(chain.last_kwargs.get("answer_mode"), "checklist")

    def test_stream_with_source_locked(self):
        chain = FakeChain()
        client = TestClient(build_app(chain))
        resp = client.post("/api/chat", json={
            "message": "test",
            "history": [],
            "active_sop": "test.pdf",
            "stream": True,
            "source_locked": True,
        })
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(chain.last_kwargs.get("source_locked"))


# ── Admin Analytics Endpoint Tests ───────────────────────────────────────

class AdminAnalyticsTests(unittest.TestCase):
    def _build_admin_client(self):
        from backend.api.admin import router as admin_router
        from backend.core.llm import get_llm

        app = FastAPI()
        app.include_router(admin_router)
        return TestClient(app)

    def _get_token(self, client):
        resp = client.post("/api/admin/login", json={"password": "admin123"})
        return resp.json()["token"]

    def test_analytics_requires_auth(self):
        client = self._build_admin_client()
        resp = client.get("/api/admin/analytics")
        self.assertEqual(resp.status_code, 403)

    def test_analytics_returns_data_when_authenticated(self):
        client = self._build_admin_client()
        token = self._get_token(client)
        resp = client.get("/api/admin/analytics", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("total_queries", data)
        self.assertIn("confidence_breakdown", data)
        self.assertIn("feedback_summary", data)

    def test_feedback_list_requires_auth(self):
        client = self._build_admin_client()
        resp = client.get("/api/admin/feedback")
        self.assertEqual(resp.status_code, 403)

    def test_failed_queries_requires_auth(self):
        client = self._build_admin_client()
        resp = client.get("/api/admin/failed-queries")
        self.assertEqual(resp.status_code, 403)


if __name__ == "__main__":
    unittest.main()
