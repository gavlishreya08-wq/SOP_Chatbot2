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

    def test_query_answers_identity_and_capability_questions_without_rag(self):
        chain = RAGChain(FakeLLM(responses=["Should not be used"]), FakeVectorStore())

        questions = [
            "Who are you?",
            "Who are you,",
            "How can you help me?",
            "how can u help me",
            "What can you do?",
            "Do u know what u can do?",
            "What is your purpose?",
        ]

        for question in questions:
            with self.subTest(question=question):
                result = asyncio.run(chain.query(question, [], None))

                self.assertIn("Prakriya AI", result["answer"])
                self.assertIn("SOP", result["answer"])
                self.assertIn("Development", result["answer"])
                self.assertIn("Testing", result["answer"])
                self.assertIn("Database", result["answer"])
                self.assertIn("Deployment", result["answer"])
                self.assertIn("Reports", result["answer"])
                self.assertEqual(result["confidence"], "high")
                self.assertIsNone(result["sources"])
                self.assertEqual(chain.llm.calls, [])

    def test_stream_query_answers_identity_without_rag(self):
        chain = RAGChain(FakeLLM(responses=["Should not be used"]), FakeVectorStore())

        async def collect_events():
            return [event async for event in chain.stream_query("What can u do?", [], None)]

        events = asyncio.run(collect_events())

        self.assertEqual(events[0]["type"], "token")
        self.assertIn("Prakriya AI", events[0]["content"])
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["confidence"], "high")
        self.assertIsNone(events[-1]["sources"])
        self.assertEqual(chain.llm.calls, [])

    def test_query_paginates_extractive_sop_section_by_cursor(self):
        source = "RoleSOP.pdf"
        content = "\n".join(f"{index}) Responsibility {index}." for index in range(1, 21))
        doc = make_doc(
            "SOURCE TITLE: Role SOP\n"
            "SECTION: Responsibilities\n"
            "CONTENT:\n"
            f"{content}",
            source,
            page=1,
            source_title="Role SOP",
            section_title="Responsibilities",
            section_index=2,
            chunk_in_section=1,
            content_type="section",
        )
        chain = RAGChain(FakeLLM(responses=["Should not be used"]), FakeVectorStore(metadatas=[doc.metadata]))

        with patch.object(chain, "_should_clarify", return_value=(False, [(source, 2.1)], [])):
            with patch.object(chain, "retrieve_docs", return_value=([doc], source)):
                first = asyncio.run(chain.query("responsibilities", [], None))
                second = asyncio.run(
                    chain.query(
                        "responsibilities",
                        [{"role": "user", "content": "responsibilities"}],
                        source,
                        cursor_offset=15,
                    )
                )

        self.assertTrue(first["has_more"])
        self.assertEqual(first["next_offset"], 15)
        self.assertIn("15. Responsibility 15.", first["answer"])
        self.assertNotIn("16. Responsibility 16.", first["answer"])
        self.assertFalse(second["has_more"])
        self.assertIn("16. Responsibility 16.", second["answer"])
        self.assertIn("20. Responsibility 20.", second["answer"])
        self.assertNotIn("1. Responsibility 1.", second["answer"])
        self.assertEqual(chain.llm.calls, [])

    def test_query_returns_all_points_for_complete_section_request(self):
        source = "RoleSOP.pdf"
        content = "\n".join(f"{index}) Responsibility {index}." for index in range(1, 41))
        doc = make_doc(
            "SOURCE TITLE: Role SOP\n"
            "SECTION: Responsibilities\n"
            "CONTENT:\n"
            f"{content}",
            source,
            page=1,
            source_title="Role SOP",
            section_title="Responsibilities",
            section_index=2,
            chunk_in_section=1,
            content_type="section",
        )
        chain = RAGChain(FakeLLM(responses=["Should not be used"]), FakeVectorStore(metadatas=[doc.metadata]))

        with patch.object(chain, "_should_clarify", return_value=(False, [(source, 2.1)], [])):
            with patch.object(chain, "retrieve_docs", return_value=([doc], source)):
                result = asyncio.run(chain.query("List all responsibilities", [], None))

        self.assertFalse(result["has_more"])
        self.assertIsNone(result["next_offset"])
        self.assertIn("1. Responsibility 1.", result["answer"])
        self.assertIn("40. Responsibility 40.", result["answer"])
        self.assertEqual(result["answer"].count("Responsibility "), 40)
        self.assertEqual(chain.llm.calls, [])

    def test_query_returns_specific_numbered_point_only(self):
        source = "RoleSOP.pdf"
        content = "\n".join(f"{index}) Responsibility {index}." for index in range(1, 41))
        doc = make_doc(
            "SOURCE TITLE: Role SOP\n"
            "SECTION: Responsibilities\n"
            "CONTENT:\n"
            f"{content}",
            source,
            page=1,
            source_title="Role SOP",
            section_title="Responsibilities",
            section_index=2,
            chunk_in_section=1,
            content_type="section",
        )
        chain = RAGChain(FakeLLM(responses=["Should not be used"]), FakeVectorStore(metadatas=[doc.metadata]))

        with patch.object(chain, "_should_clarify", return_value=(False, [(source, 2.1)], [])):
            with patch.object(chain, "retrieve_docs", return_value=([doc], source)):
                result = asyncio.run(chain.query("What is responsibility 5?", [], None))

        self.assertFalse(result["has_more"])
        self.assertIn("5. Responsibility 5.", result["answer"])
        self.assertNotIn("4. Responsibility 4.", result["answer"])
        self.assertNotIn("6. Responsibility 6.", result["answer"])
        self.assertEqual(result["answer"].count("Responsibility "), 1)
        self.assertEqual(chain.llm.calls, [])

    def test_complete_generic_points_request_refetches_all_chunks_in_selected_section(self):
        source = "LongSection.pdf"
        docs = []
        for chunk_index, start in enumerate(range(1, 41, 10), start=1):
            end = start + 9
            content = "\n".join(f"{index}) Point {index}." for index in range(start, end + 1))
            docs.append(
                make_doc(
                    "SOURCE TITLE: Long Section SOP\n"
                    "SECTION: Responsibilities\n"
                    "CONTENT:\n"
                    f"{content}",
                    source,
                    page=chunk_index,
                    source_title="Long Section SOP",
                    section_title="Responsibilities",
                    section_index=2,
                    chunk_in_section=chunk_index,
                    content_type="section",
                    chunk_id=f"{source}-s2-c{chunk_index}",
                )
            )
        chain = RAGChain(
            FakeLLM(responses=["Should not be used"]),
            FakeVectorStore(
                metadatas=[doc.metadata for doc in docs],
                documents=[doc.page_content for doc in docs],
            ),
        )

        with patch.object(chain, "_should_clarify", return_value=(False, [(source, 2.1)], [])):
            with patch.object(chain, "retrieve_docs", return_value=([docs[0]], source)):
                result = asyncio.run(chain.query("Give all points", [], None))

        self.assertFalse(result["has_more"])
        self.assertIn("1. Point 1.", result["answer"])
        self.assertIn("40. Point 40.", result["answer"])
        self.assertEqual(result["answer"].count("Point "), 40)
        self.assertEqual(chain.llm.calls, [])

    def test_complete_request_includes_numbered_continuation_chunks_with_different_titles(self):
        source = "TechnicalLead.pdf"
        chunks = [
            (
                "Responsibilities",
                2,
                1,
                "\n".join(f"{index}) Responsibility {index}." for index in range(1, 20)),
            ),
            (
                "Jira.",
                3,
                2,
                "\n".join(f"{index}) Responsibility {index}." for index in range(20, 34)),
            ),
            (
                "Training Requirement",
                4,
                3,
                "\n".join(f"{index}) Responsibility {index}." for index in range(34, 47)),
            ),
        ]
        docs = [
            make_doc(
                "SOURCE TITLE: Technical Lead\n"
                f"SECTION: {title}\n"
                "CONTENT:\n"
                f"{content}",
                source,
                page=chunk_index,
                source_title="Technical Lead",
                section_title=title,
                section_index=section_index,
                chunk_in_section=chunk_index,
                content_type="section",
                chunk_id=f"{source}-s{section_index}-c{chunk_index}",
            )
            for title, section_index, chunk_index, content in chunks
        ]
        chain = RAGChain(
            FakeLLM(responses=["Should not be used"]),
            FakeVectorStore(
                metadatas=[doc.metadata for doc in docs],
                documents=[doc.page_content for doc in docs],
            ),
        )

        with patch.object(chain, "_should_clarify", return_value=(False, [(source, 2.1)], [])):
            with patch.object(chain, "retrieve_docs", return_value=([docs[0]], source)):
                result = asyncio.run(chain.query("Give all responsibilities of technical lead", [], None))

        self.assertFalse(result["has_more"])
        self.assertIn("1. Responsibility 1.", result["answer"])
        self.assertIn("46. Responsibility 46.", result["answer"])
        self.assertEqual(result["answer"].count("Responsibility "), 46)
        self.assertEqual(chain.llm.calls, [])

    def test_specific_generic_point_request_refetches_section_but_returns_only_that_point(self):
        source = "LongSection.pdf"
        docs = []
        for chunk_index, start in enumerate(range(1, 41, 10), start=1):
            end = start + 9
            content = "\n".join(f"{index}) Point {index}." for index in range(start, end + 1))
            docs.append(
                make_doc(
                    "SOURCE TITLE: Long Section SOP\n"
                    "SECTION: Responsibilities\n"
                    "CONTENT:\n"
                    f"{content}",
                    source,
                    page=chunk_index,
                    source_title="Long Section SOP",
                    section_title="Responsibilities",
                    section_index=2,
                    chunk_in_section=chunk_index,
                    content_type="section",
                    chunk_id=f"{source}-s2-c{chunk_index}",
                )
            )
        chain = RAGChain(
            FakeLLM(responses=["Should not be used"]),
            FakeVectorStore(
                metadatas=[doc.metadata for doc in docs],
                documents=[doc.page_content for doc in docs],
            ),
        )

        with patch.object(chain, "_should_clarify", return_value=(False, [(source, 2.1)], [])):
            with patch.object(chain, "retrieve_docs", return_value=([docs[0]], source)):
                result = asyncio.run(chain.query("What is point 25?", [], None))

        self.assertFalse(result["has_more"])
        self.assertIn("25. Point 25.", result["answer"])
        self.assertNotIn("24. Point 24.", result["answer"])
        self.assertNotIn("26. Point 26.", result["answer"])
        self.assertEqual(result["answer"].count("Point "), 1)
        self.assertEqual(chain.llm.calls, [])

    def test_specific_item_uses_literal_number_after_subpoints_and_continuation_chunks(self):
        source = "TechnicalLead.pdf"
        docs = []
        first_content = "\n".join(
            [
                *[f"{index}) Responsibility {index}." for index in range(1, 20)],
                "a) Review progress.",
                "b) Resolve issues.",
                "c) Send minutes.",
            ]
        )
        later_content = "\n".join(f"{index}) Responsibility {index}." for index in range(20, 47))
        for chunk_index, (title, section_index, content) in enumerate(
            [
                ("Responsibilities", 2, first_content),
                ("Jira.", 3, later_content),
            ],
            start=1,
        ):
            docs.append(
                make_doc(
                    "SOURCE TITLE: Technical Lead\n"
                    f"SECTION: {title}\n"
                    "CONTENT:\n"
                    f"{content}",
                    source,
                    page=chunk_index,
                    source_title="Technical Lead",
                    section_title=title,
                    section_index=section_index,
                    chunk_in_section=chunk_index,
                    content_type="section",
                    chunk_id=f"{source}-s{section_index}-c{chunk_index}",
                )
            )
        chain = RAGChain(
            FakeLLM(responses=["Should not be used"]),
            FakeVectorStore(
                metadatas=[doc.metadata for doc in docs],
                documents=[doc.page_content for doc in docs],
            ),
        )

        with patch.object(chain, "_should_clarify", return_value=(False, [(source, 2.1)], [])):
            with patch.object(chain, "retrieve_docs", return_value=([docs[0]], source)):
                result = asyncio.run(chain.query("What is responsibility 35?", [], None))

        self.assertFalse(result["has_more"])
        self.assertIn("35. Responsibility 35.", result["answer"])
        self.assertNotIn("34. Responsibility 34.", result["answer"])
        self.assertNotIn("36. Responsibility 36.", result["answer"])
        self.assertEqual(result["answer"].count("Responsibility "), 1)
        self.assertEqual(chain.llm.calls, [])

    def test_manual_more_continues_from_previous_numbered_answer(self):
        source = "RoleSOP.pdf"
        lines = [f"{index}) Responsibility {index}." for index in range(1, 47)]
        lines.insert(19, "a) Review progress of each engineer.")
        lines.insert(20, "b) Identify issues and resolve them.")
        lines.insert(21, "c) Ensure every review is minuted.")
        content = "\n".join(lines)
        doc = make_doc(
            "SOURCE TITLE: Role SOP\n"
            "SECTION: Responsibilities\n"
            "CONTENT:\n"
            f"{content}",
            source,
            page=1,
            source_title="Role SOP",
            section_title="Responsibilities",
            section_index=2,
            chunk_in_section=1,
            content_type="section",
        )
        chain = RAGChain(FakeLLM(responses=["Should not be used"]), FakeVectorStore(metadatas=[doc.metadata]))
        history = [
            {"role": "user", "content": "What are the responsibilities of Technical Lead?"},
            {
                "role": "assistant",
                "content": "\n".join(f"{index}. Responsibility {index}." for index in range(1, 16)),
            },
            {"role": "user", "content": "more"},
            {
                "role": "assistant",
                "content": (
                    "16. Responsibility 16.\n"
                    "17. Responsibility 17.\n"
                    "18. Responsibility 18.\n"
                    "19. Responsibility 19.\n"
                    "a) Review progress of each engineer.\n"
                    "b) Identify issues and resolve them.\n"
                    "c) Ensure every review is minuted."
                ),
            },
            {"role": "user", "content": "more"},
        ]

        with patch.object(chain, "_should_clarify", return_value=(False, [(source, 2.1)], [])):
            with patch.object(chain, "retrieve_docs", return_value=([doc], source)):
                result = asyncio.run(chain.query("more", history, source))

        self.assertNotEqual(result["answer"], "This information is not available in the provided SOP.")
        self.assertIn("20. Responsibility 20.", result["answer"])
        self.assertIn("34. Responsibility 34.", result["answer"])
        self.assertNotIn("19. Responsibility 19.", result["answer"])
        self.assertTrue(result["has_more"])
        self.assertEqual(result["next_offset"], 34)
        self.assertEqual(chain.llm.calls, [])

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

    def test_query_removes_fallback_line_when_valid_answer_exists(self):
        vectorstore = FakeVectorStore(metadatas=[{"source": "ReleaseProcess.pdf"}])
        llm = FakeLLM(
            responses=[
                "Release planning includes impact analysis and deployment approvals.\n\n"
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

        with patch.object(chain, "_confidence_level", return_value="high"):
            with patch.object(chain, "retrieve_docs", return_value=(docs, "ReleaseProcess.pdf")):
                result = asyncio.run(chain.query("What is included in release planning?", [], None))

        self.assertEqual(result["answer"], "Release planning includes impact analysis and deployment approvals.")
        self.assertIsNotNone(result["sources"])
        self.assertIsNone(result["followup"])
        self.assertEqual(result["active_sop"], "ReleaseProcess.pdf")
        self.assertNotIn("This information is not available", result["answer"])

    def test_query_removes_generic_disclaimer_sentence_when_answer_exists(self):
        vectorstore = FakeVectorStore(metadatas=[{"source": "RoleSOP.pdf"}])
        llm = FakeLLM(
            responses=[
                "The Business Analyst reviews requirement details and creates Jira tickets. "
                "Approval timing is not mentioned in the provided context.\nFOLLOWUP: NONE"
            ]
        )
        chain = RAGChain(llm, vectorstore)
        docs = [
            make_doc(
                "The Business Analyst reviews requirement details and creates Jira tickets.",
                "RoleSOP.pdf",
                page=0,
            )
        ]

        with patch.object(chain, "_confidence_level", return_value="high"):
            with patch.object(chain, "retrieve_docs", return_value=(docs, "RoleSOP.pdf")):
                result = asyncio.run(chain.query("What is the BA role in creating tickets?", [], None))

        self.assertEqual(
            result["answer"],
            "The Business Analyst reviews requirement details and creates Jira tickets.",
        )
        self.assertNotIn("not mentioned", result["answer"].lower())
        self.assertEqual(result["active_sop"], "RoleSOP.pdf")

    def test_query_keeps_related_grounded_answer_for_partial_intent(self):
        vectorstore = FakeVectorStore(metadatas=[{"source": "ProductionIssue.pdf"}])
        llm = FakeLLM(
            responses=[
                "Production issues are logged in Jira and reviewed by the Technical Lead.\n"
                "FOLLOWUP: NONE"
            ]
        )
        chain = RAGChain(llm, vectorstore)
        docs = [
            make_doc(
                "Production issues are logged in Jira and reviewed by the Technical Lead.",
                "ProductionIssue.pdf",
                page=0,
            )
        ]

        with patch.object(chain, "_confidence_level", return_value="high"):
            with patch.object(chain, "retrieve_docs", return_value=(docs, "ProductionIssue.pdf")):
                result = asyncio.run(chain.query("What is the escalation SLA?", [], None))

        self.assertEqual(
            result["answer"],
            "Production issues are logged in Jira and reviewed by the Technical Lead.",
        )
        self.assertIsNotNone(result["sources"])
        self.assertEqual(result["active_sop"], "ProductionIssue.pdf")

    def test_response_validation_classifies_full_partial_and_no_answer(self):
        chain = RAGChain(FakeLLM(), FakeVectorStore())
        context = "Developers review code and update Jira tickets."

        full = chain._validate_response(
            "What do developers do?",
            "Developers review code and update Jira tickets.",
            context,
        )
        partial = chain._validate_response(
            "What do developers do and who approves it?",
            "Developers review code. Approval details are not mentioned in the provided context.",
            context,
        )
        no_answer = chain._validate_response(
            "Who approves it?",
            "Information not available.",
            context,
        )

        self.assertEqual(full.quality, "full")
        self.assertEqual(partial.quality, "partial")
        self.assertEqual(partial.answer, "Developers review code.")
        self.assertEqual(no_answer.quality, "no_answer")
        self.assertEqual(no_answer.answer, "This information is not available in the provided SOP.")

    def test_normalize_answer_renumbers_repeated_numbered_items(self):
        chain = RAGChain(FakeLLM(), FakeVectorStore())

        answer = chain._normalize_answer(
            "1. Code Structure\n"
            "1. Naming Conventions\n"
            "1. Code Documentation"
        )

        self.assertEqual(
            answer,
            "1. Code Structure\n"
            "2. Naming Conventions\n"
            "3. Code Documentation",
        )

    def test_normalize_answer_preserves_nested_numbering(self):
        chain = RAGChain(FakeLLM(), FakeVectorStore())

        answer = chain._normalize_answer(
            "1. Parent item\n"
            "   1. Nested item\n"
            "   1. Another nested item\n"
            "1. Second parent"
        )

        self.assertEqual(
            answer,
            "1. Parent item\n"
            "   1. Nested item\n"
            "   2. Another nested item\n"
            "2. Second parent",
        )

    def test_format_context_strips_numeric_prefixes_for_llm_context(self):
        doc = make_doc(
            "SOURCE TITLE: Coding SOP\n"
            "SECTION: Standards\n"
            "CONTENT:\n"
            "1. Code Structure\n"
            "1. Naming Conventions",
            "CodingSOP.pdf",
            section_title="Standards",
            section_index=1,
            chunk_in_section=1,
            content_type="section",
        )
        chain = RAGChain(FakeLLM(), FakeVectorStore(metadatas=[doc.metadata]))

        context = chain._format_context([doc])
        raw_context = chain._format_context([doc], strip_numbering=False)

        self.assertIn("Code Structure", context)
        self.assertNotIn("1. Code Structure", context)
        self.assertIn("1. Code Structure", raw_context)

    def test_query_formats_time_based_table_from_sop_chunks(self):
        source = "DailyPlan.pdf"
        doc = make_doc(
            "SOURCE TITLE: Daily Plan\n"
            "SECTION: Agenda\n"
            "CONTENT:\n"
            "Start Time | End Time | Duration | Activity\n"
            "| --- | --- | --- | --- |\n"
            "9:30 AM | 10:00 AM | 0:30 | Production issues\n"
            "- Development plan tasks\n"
            "10:00 AM | 10:30 AM | 0:30 | Code review",
            source,
            page=0,
            source_title="Daily Plan",
            section_title="Agenda",
            section_index=1,
            chunk_in_section=1,
            content_type="section",
        )
        chain = RAGChain(FakeLLM(responses=["Should not be used"]), FakeVectorStore(metadatas=[doc.metadata]))

        with patch.object(chain, "_should_clarify", return_value=(False, [(source, 2.1)], [])):
            with patch.object(chain, "retrieve_docs", return_value=([doc], source)):
                result = asyncio.run(chain.query("Show agenda", [], None))

        self.assertIn("1. 9:30 AM - 10:00 AM (0:30)", result["answer"])
        self.assertIn("   - Production issues", result["answer"])
        self.assertIn("   - Development plan tasks", result["answer"])
        self.assertIn("2. 10:00 AM - 10:30 AM (0:30)", result["answer"])
        self.assertNotIn("Start Time", result["answer"])
        self.assertEqual(chain.llm.calls, [])

    def test_query_groups_duplicate_time_slot_table_rows(self):
        source = "ReviewMeeting.pdf"
        doc = make_doc(
            "SOURCE TITLE: Review Meeting\n"
            "SECTION: Agenda\n"
            "CONTENT:\n"
            "Start Time | End Time | Duration | Activity\n"
            "11:00 AM | 12:00 PM | 1:00 | Review project plan\n"
            "11:00 AM | 12:00 PM | 1:00 | Conduct issue resolution\n"
            "12:00 PM | 12:30 PM | 0:30 | Share action items",
            source,
            page=0,
            source_title="Review Meeting",
            section_title="Agenda",
            section_index=1,
            chunk_in_section=1,
            content_type="section",
        )
        chain = RAGChain(FakeLLM(responses=["Should not be used"]), FakeVectorStore(metadatas=[doc.metadata]))

        with patch.object(chain, "_should_clarify", return_value=(False, [(source, 2.1)], [])):
            with patch.object(chain, "retrieve_docs", return_value=([doc], source)):
                result = asyncio.run(chain.query("Show agenda", [], None))

        self.assertEqual(result["answer"].count("11:00 AM - 12:00 PM"), 1)
        self.assertIn("1. 11:00 AM - 12:00 PM (1:00)", result["answer"])
        self.assertIn("   - Review project plan", result["answer"])
        self.assertIn("   - Conduct issue resolution", result["answer"])
        self.assertIn("2. 12:00 PM - 12:30 PM (0:30)", result["answer"])
        self.assertLess(
            result["answer"].find("Review project plan"),
            result["answer"].find("Conduct issue resolution"),
        )
        self.assertLess(
            result["answer"].find("Conduct issue resolution"),
            result["answer"].find("12:00 PM - 12:30 PM"),
        )
        self.assertEqual(chain.llm.calls, [])

    def test_query_keeps_time_rows_with_empty_cells_separate(self):
        source = "DailyPlan.pdf"
        doc = make_doc(
            "SOURCE TITLE: Daily Plan\n"
            "SECTION: Extracted Table:\n"
            "CONTENT:\n"
            "# | Start Time | End Time | Duration hh:mm | Activity\n"
            "| --- | --- | --- | --- | --- |\n"
            "8 | 3:00 PM | 4:00 PM | 1:00 | Conduct issue resolution meets\n"
            "9 | 4:00 PM | | | Spare time for planning",
            source,
            page=0,
            source_title="Daily Plan",
            section_title="Extracted Table:",
            section_index=1,
            chunk_in_section=1,
            content_type="section",
        )
        chain = RAGChain(FakeLLM(responses=["Should not be used"]), FakeVectorStore(metadatas=[doc.metadata]))

        with patch.object(chain, "_should_clarify", return_value=(False, [(source, 2.1)], [])):
            with patch.object(chain, "retrieve_docs", return_value=([doc], source)):
                result = asyncio.run(chain.query("Show agenda", [], None))

        self.assertIn("1. 3:00 PM - 4:00 PM (1:00)", result["answer"])
        self.assertIn("   Conduct issue resolution meets", result["answer"])
        self.assertIn("2. 4:00 PM", result["answer"])
        self.assertIn("   Spare time for planning", result["answer"])
        self.assertNotIn("- 8", result["answer"])
        self.assertNotIn("- 9", result["answer"])
        self.assertEqual(chain.llm.calls, [])

    def test_query_formats_generic_table_from_sop_chunks(self):
        source = "Roles.pdf"
        doc = make_doc(
            "SOURCE TITLE: Roles\n"
            "SECTION: Responsibilities\n"
            "CONTENT:\n"
            "Role    Responsibility    Frequency\n"
            "Developer    Code review    Daily\n"
            "QA    Test execution    Daily",
            source,
            page=0,
            source_title="Roles",
            section_title="Responsibilities",
            section_index=1,
            chunk_in_section=1,
            content_type="section",
        )
        chain = RAGChain(FakeLLM(responses=["Should not be used"]), FakeVectorStore(metadatas=[doc.metadata]))

        with patch.object(chain, "_should_clarify", return_value=(False, [(source, 2.1)], [])):
            with patch.object(chain, "retrieve_docs", return_value=([doc], source)):
                result = asyncio.run(chain.query("List responsibilities", [], None))

        self.assertEqual(
            result["answer"],
            "1. Developer\n"
            "   - Responsibility: Code review\n"
            "   - Frequency: Daily\n"
            "2. QA\n"
            "   - Responsibility: Test execution\n"
            "   - Frequency: Daily",
        )
        self.assertEqual(chain.llm.calls, [])

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

    def test_stream_query_removes_fallback_line_when_valid_answer_exists(self):
        vectorstore = FakeVectorStore(metadatas=[{"source": "ReleaseProcess.pdf"}])
        llm = FakeLLM(
            stream_chunks=[
                "Release planning includes impact analysis and deployment approvals.\n\n",
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
            with patch.object(chain, "_confidence_level", return_value="high"):
                with patch.object(chain, "retrieve_docs", return_value=(docs, "ReleaseProcess.pdf")):
                    return [event async for event in chain.stream_query("What is included in release planning?", [], None)]

        events = asyncio.run(collect_events())

        self.assertEqual(events[-1]["full_answer"], "Release planning includes impact analysis and deployment approvals.")
        self.assertIsNotNone(events[-1]["sources"])
        self.assertIsNone(events[-1]["followup"])
        self.assertEqual(events[-1]["active_sop"], "ReleaseProcess.pdf")
        self.assertNotIn("This information is not available", events[-1]["full_answer"])

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
