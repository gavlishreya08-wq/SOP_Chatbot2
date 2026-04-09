import asyncio
import unittest

from backend.config import settings
from backend.core.rag_chain import RAGChain
from backend.rag.retriever import build_source_catalog, retrieve
from backend.rag.vectorstore import load_existing_vectorstore
from tests.chatbot_scenarios import (
    ACTIVE_SOP_SCENARIOS,
    CLARIFICATION_SCENARIOS,
    NEGATIVE_SCENARIOS,
    POSITIVE_SCENARIOS,
)
from tests.helpers import FakeLLM


class ChatbotScenarioMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vectorstore = load_existing_vectorstore()
        if cls.vectorstore is None:
            raise unittest.SkipTest(f"No vectorstore found in {settings.chroma_db_dir}")
        cls.source_catalog = build_source_catalog(cls.vectorstore)
        cls.rag_chain = RAGChain(FakeLLM(), cls.vectorstore)

    def test_positive_scenario_matrix_routes_to_expected_source(self):
        for group in POSITIVE_SCENARIOS:
            for question in group.variants:
                with self.subTest(group=group.name, question=question):
                    docs, source = retrieve(
                        self.vectorstore,
                        question,
                        source_catalog=self.source_catalog,
                    )
                    self.assertEqual(source, group.expected_source)
                    self.assertTrue(docs)

    def test_negative_scenario_matrix_returns_no_documents(self):
        for group in NEGATIVE_SCENARIOS:
            for question in group.variants:
                with self.subTest(group=group.name, question=question):
                    docs, source = retrieve(
                        self.vectorstore,
                        question,
                        source_catalog=self.source_catalog,
                    )
                    self.assertEqual(docs, [])
                    self.assertIsNone(source)

    def test_active_sop_followup_matrix_stays_on_same_source(self):
        for group in ACTIVE_SOP_SCENARIOS:
            for question in group.variants:
                with self.subTest(group=group.name, question=question):
                    docs, source = retrieve(
                        self.vectorstore,
                        question,
                        active_sop=group.active_sop,
                        source_catalog=self.source_catalog,
                    )
                    self.assertEqual(source, group.expected_source)
                    self.assertTrue(docs)

    def test_clarification_matrix_prompts_for_disambiguation(self):
        for group in CLARIFICATION_SCENARIOS:
            for question in group.variants:
                with self.subTest(group=group.name, question=question):
                    result = asyncio.run(self.rag_chain.query(question, [], None))
                    self.assertIn("Which", result["answer"])
                    self.assertEqual(result["confidence"], "low")
                    self.assertIsNotNone(result["suggestions"])


if __name__ == "__main__":
    unittest.main()
