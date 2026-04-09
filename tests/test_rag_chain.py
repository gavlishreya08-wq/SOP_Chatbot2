import asyncio
import unittest
from unittest.mock import patch

from backend.core.rag_chain import RAGChain
from tests.helpers import FakeLLM, FakeVectorStore, make_doc


class RAGChainTests(unittest.TestCase):
    def test_query_returns_conversational_response(self):
        chain = RAGChain(FakeLLM(), FakeVectorStore())

        result = asyncio.run(chain.query("hi", [], None))

        self.assertIn("Prakriya AI", result["answer"])
        self.assertIsNone(result["sources"])

    def test_query_rejects_ungrounded_answer(self):
        vectorstore = FakeVectorStore(metadatas=[{"source": "ChangeManagementWorkflow_version2.pdf"}])
        llm = FakeLLM(responses=["Completely unrelated answer.\nFOLLOWUP: NONE"])
        chain = RAGChain(llm, vectorstore)
        docs = [
            make_doc(
                "Change management flow requires PM review and impact analysis.",
                "ChangeManagementWorkflow_version2.pdf",
                page=0,
            )
        ]

        with patch.object(chain, "retrieve_docs", return_value=(docs, "ChangeManagementWorkflow_version2.pdf")):
            result = asyncio.run(chain.query("Explain change workflow", [], None))

        self.assertEqual(result["answer"], "This information is not available in the provided SOP.")
        self.assertIsNone(result["sources"])
        self.assertIsNone(result["followup"])

    def test_stream_query_emits_done_event_with_followup(self):
        vectorstore = FakeVectorStore(metadatas=[{"source": "ChangeManagementWorkflow_version2.pdf"}])
        llm = FakeLLM(
            stream_chunks=[
                "1. PM reviews the request [Page 1]\n",
                "FOLLOW",
                "UP: Who prepares the impact analysis?",
            ]
        )
        chain = RAGChain(llm, vectorstore)
        docs = [
            make_doc(
                "PM reviews the request and the TL prepares the impact analysis.",
                "ChangeManagementWorkflow_version2.pdf",
                page=0,
                page_label="1",
                pdf_link="https://example.com/workflow",
                version="v2.0",
                created_date="28 Mar 2024",
            )
        ]

        async def collect_events():
            with patch.object(chain, "retrieve_docs", return_value=(docs, "ChangeManagementWorkflow_version2.pdf")):
                return [event async for event in chain.stream_query("Explain workflow", [], None)]

        events = asyncio.run(collect_events())

        self.assertEqual(events[0]["type"], "token")
        self.assertIn("PM reviews", events[0]["content"])
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["followup"], "Who prepares the impact analysis?")
        self.assertEqual(events[-1]["active_sop"], "ChangeManagementWorkflow_version2.pdf")
        self.assertEqual(events[-1]["sources"]["pages"], ["1"])

    def test_query_returns_clarification_for_ambiguous_prompt(self):
        vectorstore = FakeVectorStore(
            metadatas=[
                {
                    "source": "RR_TestLead_V1.pdf",
                    "source_title": "TEST-LEAD",
                    "source_summary": "Responsibilities and job objectives.",
                    "source_aliases": "Test Lead | Test Lead role",
                    "source_intents": "Test Lead responsibilities",
                    "source_section_titles": "Job Objectives | Responsibilities",
                },
                {
                    "source": "3_RR_TechnicalLead_version2.pdf",
                    "source_title": "TECHNICAL LEAD",
                    "source_summary": "Technical lead responsibilities and objectives.",
                    "source_aliases": "Technical Lead | Technical Lead role",
                    "source_intents": "Technical Lead responsibilities",
                    "source_section_titles": "Job Objectives | Responsibilities",
                },
            ],
            documents=[
                "Test lead job objectives and responsibilities.",
                "Technical lead job objectives and responsibilities.",
            ],
        )
        chain = RAGChain(FakeLLM(), vectorstore)

        result = asyncio.run(chain.query("What is the role?", [], None))

        self.assertIn("Which role do you mean?", result["answer"])
        self.assertEqual(result["confidence"], "low")
        self.assertTrue(result["suggestions"])

    def test_query_marks_weak_retrieval_as_low_confidence(self):
        vectorstore = FakeVectorStore(
            metadatas=[
                {"source": "SOP_TestAutomation_V1.0.pdf", "source_title": "TEST AUTOMATION"},
            ]
        )
        llm = FakeLLM(responses=["Test automation objective is to streamline testing.\nFOLLOWUP: NONE"])
        chain = RAGChain(llm, vectorstore)
        docs = [
            make_doc(
                "Test automation objective is to streamline testing.",
                "SOP_TestAutomation_V1.0.pdf",
                page=0,
                page_label="1",
                section_title="Objective",
            )
        ]

        with patch.object(chain, "_should_clarify", return_value=(False, [("SOP_TestAutomation_V1.0.pdf", 0.8)], [])):
            with patch.object(chain, "retrieve_docs", return_value=(docs, "SOP_TestAutomation_V1.0.pdf")):
                result = asyncio.run(chain.query("Explain the automation objective", [], None))

        self.assertEqual(result["confidence"], "low")
        self.assertTrue(result["answer"].startswith("Low confidence:"))


if __name__ == "__main__":
    unittest.main()
