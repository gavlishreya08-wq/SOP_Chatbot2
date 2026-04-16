import asyncio
import unittest
from unittest.mock import patch

from backend.core.rag_chain import RAGChain
from tests.helpers import FakeLLM, FakeVectorStore, make_doc


class RAGChainTests(unittest.TestCase):
    def test_effective_active_sop_drops_stale_source_for_specific_question(self):
        chain = RAGChain(FakeLLM(), FakeVectorStore())

        effective = chain._effective_active_sop(
            "Who is the CTO?",
            [{"role": "assistant", "content": "Previous SOP answer"}],
            "GELJira_IssueCreation(Annexure2).pdf",
            source_locked=False,
        )

        self.assertIsNone(effective)

    def test_effective_active_sop_keeps_source_for_generic_followup(self):
        chain = RAGChain(FakeLLM(), FakeVectorStore())

        effective = chain._effective_active_sop(
            "job objective",
            [{"role": "assistant", "content": "Previous SOP answer"}],
            "3_RR_TechnicalLead_version2.pdf",
            source_locked=False,
        )

        self.assertEqual(effective, "3_RR_TechnicalLead_version2.pdf")

    def test_effective_active_sop_keeps_source_when_locked(self):
        chain = RAGChain(FakeLLM(), FakeVectorStore())

        effective = chain._effective_active_sop(
            "Who is the CTO?",
            [{"role": "assistant", "content": "Previous SOP answer"}],
            "GELJira_IssueCreation(Annexure2).pdf",
            source_locked=True,
        )

        self.assertEqual(effective, "GELJira_IssueCreation(Annexure2).pdf")

    def test_needs_rewrite_skips_short_standalone_topic_change(self):
        llm = FakeLLM(responses=["Should not be used"])
        chain = RAGChain(llm, FakeVectorStore())

        rewritten = asyncio.run(
            chain.rewrite_query(
                "CEO role",
                [{"role": "assistant", "content": "Previous SOP answer"}],
                "GELJira_IssueCreation(Annexure2).pdf",
            )
        )

        self.assertEqual(rewritten, "CEO role")
        self.assertEqual(llm.calls, [])

    def test_needs_rewrite_keeps_generic_followup_context(self):
        llm = FakeLLM(responses=["Technical Lead job objective"])
        chain = RAGChain(llm, FakeVectorStore())

        rewritten = asyncio.run(
            chain.rewrite_query(
                "job objective",
                [{"role": "assistant", "content": "Previous SOP answer"}],
                "3_RR_TechnicalLead_version2.pdf",
            )
        )

        self.assertEqual(rewritten, "Technical Lead job objective")
        self.assertEqual(len(llm.calls), 1)

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

    def test_query_normalizes_mixed_not_available_answer(self):
        vectorstore = FakeVectorStore(metadatas=[{"source": "ReleaseProcess.pdf"}])
        llm = FakeLLM(
            responses=[
                "The purpose of the planned software release is to ensure planned software release.\n\n"
                "This information is not available in the provided SOP.\nFOLLOWUP: NONE"
            ]
        )
        chain = RAGChain(llm, vectorstore)
        docs = [
            make_doc(
                "Release planning includes impact analysis and deployment approvals.",
                "ReleaseProcess.pdf",
                page=0,
            )
        ]

        with patch.object(chain, "retrieve_docs", return_value=(docs, "ReleaseProcess.pdf")):
            result = asyncio.run(chain.query("What is the purpose of the planned software release?", [], None))

        self.assertEqual(result["answer"], "This information is not available in the provided SOP.")
        self.assertIsNone(result["sources"])
        self.assertIsNone(result["followup"])
        self.assertIsNone(result["active_sop"])
        self.assertIsNone(result["suggestions"])

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

    def test_stream_query_normalizes_mixed_not_available_answer(self):
        vectorstore = FakeVectorStore(metadatas=[{"source": "ReleaseProcess.pdf"}])
        llm = FakeLLM(
            stream_chunks=[
                "The purpose of the planned software release is to ensure planned software release.\n\n",
                "This information is not available in the provided SOP.\n",
                "FOLLOWUP: NONE",
            ]
        )
        chain = RAGChain(llm, vectorstore)
        docs = [
            make_doc(
                "Release planning includes impact analysis and deployment approvals.",
                "ReleaseProcess.pdf",
                page=0,
            )
        ]

        async def collect_events():
            with patch.object(chain, "retrieve_docs", return_value=(docs, "ReleaseProcess.pdf")):
                return [event async for event in chain.stream_query("What is the purpose of the planned software release?", [], None)]

        events = asyncio.run(collect_events())

        self.assertEqual(events[-1]["full_answer"], "This information is not available in the provided SOP.")
        self.assertIsNone(events[-1]["sources"])
        self.assertIsNone(events[-1]["followup"])
        self.assertIsNone(events[-1]["active_sop"])
        self.assertIsNone(events[-1]["suggestions"])

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
