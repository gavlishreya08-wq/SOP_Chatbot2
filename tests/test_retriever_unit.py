import unittest

from backend.rag.retriever import build_source_catalog, infer_source_candidates, retrieve
from tests.helpers import FakeVectorStore, make_doc


class RetrieverUnitTests(unittest.TestCase):
    def test_infer_source_candidates_matches_jira_annexure(self):
        store = FakeVectorStore(
            metadatas=[
                {"source": "GELJira_IssueCreation(Annexure2).pdf"},
                {"source": "SOP_Root Cause Fixture.pdf"},
                {"source": "ChangeManagementWorkflow_version2.pdf"},
            ],
            documents=[
                "Create tickets as per requirement within respective project.",
                "Production issue root cause fixture in GEL JIRA.",
                "Change management flow and Jira workflow.",
            ],
        )

        catalog = build_source_catalog(store)
        candidates = infer_source_candidates("How to create a Jira issue?", catalog, limit=5)

        self.assertEqual(candidates[0][0], "GELJira_IssueCreation(Annexure2).pdf")
        self.assertGreater(candidates[0][1], 0.7)

    def test_infer_source_candidates_uses_first_page_snippet_for_role_docs(self):
        store = FakeVectorStore(
            metadatas=[
                {"source": "Final_v2_Roles and Responsibilities_Lead_DB.pdf"},
                {"source": "RR_TestLead_V1.pdf"},
            ],
            documents=[
                "LEAD DATABASE responsibilities for database design and management.",
                "TEST-LEAD responsibilities and job objectives for leading the testing team.",
            ],
        )

        catalog = build_source_catalog(store)
        candidates = infer_source_candidates(
            "Tell me roles and responsibility of Test Lead",
            catalog,
            limit=5,
        )

        self.assertEqual(candidates[0][0], "RR_TestLead_V1.pdf")

    def test_retrieve_prefers_strong_title_matched_source(self):
        wrong_doc = make_doc(
            'Attach this documentation to the "Production Issue Root Cause Fixture" ticket in GEL JIRA.',
            "SOP_Root Cause Fixture.pdf",
            page=1,
        )
        correct_doc = make_doc(
            "Create tickets as per requirement within respective project. Only these issue types will be visible while creation.",
            "GELJira_IssueCreation(Annexure2).pdf",
            page=0,
        )
        store = FakeVectorStore(
            global_results=[
                (wrong_doc, 1.01),
                (correct_doc, 1.32),
            ],
            filtered_results={
                "GELJira_IssueCreation(Annexure2).pdf": [
                    (correct_doc, 0.85),
                ]
            },
            metadatas=[
                {"source": "SOP_Root Cause Fixture.pdf"},
                {"source": "GELJira_IssueCreation(Annexure2).pdf"},
            ],
            documents=[
                'Attach this documentation to the "Production Issue Root Cause Fixture" ticket in GEL JIRA.',
                "Create tickets as per requirement within respective project.",
            ],
        )

        docs, source = retrieve(
            store,
            "How to create a Jira issue?",
            source_catalog=build_source_catalog(store),
        )

        self.assertEqual(source, "GELJira_IssueCreation(Annexure2).pdf")
        self.assertEqual([doc.metadata["source"] for doc in docs], ["GELJira_IssueCreation(Annexure2).pdf"])

    def test_retrieve_rejects_weak_irrelevant_match(self):
        unrelated_doc = make_doc(
            "Perform the pilot migration in a controlled environment and monitor performance.",
            "SOP_Database Migration and Porting.pdf",
            page=2,
        )
        store = FakeVectorStore(
            global_results=[(unrelated_doc, 1.33)],
            metadatas=[{"source": "SOP_Database Migration and Porting.pdf"}],
        )

        docs, source = retrieve(
            store,
            "Leave policy overview",
            source_catalog=build_source_catalog(store),
        )

        self.assertEqual(docs, [])
        self.assertIsNone(source)

    def test_retrieve_switches_from_active_source_for_explicit_workflow_change(self):
        active_doc = make_doc(
            "Review the system architecture and capture the design decisions.",
            "SOP How to design system architecture.pdf",
            page=1,
            source_title="SYSTEM ARCHITECTURE DESIGN",
            source_aliases="System architecture SOP",
        )
        target_doc = make_doc(
            "Production issue workflow begins when the PM raises a GELJIRA production issue ticket.",
            "SOP_Production Issue.pdf",
            page=0,
            source_title="HOW TO MANAGE PRODUCTION ISSUE",
            source_aliases="Production issue SOP | How to manage production issue",
        )
        store = FakeVectorStore(
            global_results=[
                (active_doc, 0.88),
                (target_doc, 1.09),
            ],
            filtered_results={
                "SOP How to design system architecture.pdf": [(active_doc, 0.84)],
                "SOP_Production Issue.pdf": [(target_doc, 0.62)],
            },
            metadatas=[
                active_doc.metadata,
                target_doc.metadata,
            ],
            documents=[
                active_doc.page_content,
                target_doc.page_content,
            ],
        )

        docs, source = retrieve(
            store,
            "How to manage a production issue?",
            active_sop="SOP How to design system architecture.pdf",
            source_catalog=build_source_catalog(store),
        )

        self.assertEqual(source, "SOP_Production Issue.pdf")
        self.assertEqual([doc.metadata["source"] for doc in docs], ["SOP_Production Issue.pdf"])

    def test_retrieve_rejects_fallback_match_based_only_on_generic_terms(self):
        generic_doc = make_doc(
            "Yearly training calendar and company policy overview for role development.",
            "Final_v2_Roles and Responsibilities_Lead_Report.pdf",
            page=0,
            source_title="REPORT LEAD",
            section_title="Document Guide",
        )
        store = FakeVectorStore(
            global_results=[(generic_doc, 1.41)],
            metadatas=[generic_doc.metadata],
            documents=[generic_doc.page_content],
        )

        docs, source = retrieve(
            store,
            "Holiday calendar for this year",
            source_catalog=build_source_catalog(store),
        )

        self.assertEqual(docs, [])
        self.assertIsNone(source)

    def test_retrieve_rejects_fallback_match_from_prefix_only_content_overlap(self):
        fuzzy_doc = make_doc(
            "Workflow allowlist updates for server access approvals.",
            "SOP_ServerWhitelistingg.pdf",
            page=0,
            source_title="Server Whitelisting",
            source_aliases="Workflow allowlist access approvals",
        )
        store = FakeVectorStore(
            global_results=[(fuzzy_doc, 1.02)],
            metadatas=[fuzzy_doc.metadata],
            documents=[fuzzy_doc.page_content],
        )

        docs, source = retrieve(
            store,
            "Remote work laptop allowance policy",
            source_catalog=build_source_catalog(store),
        )

        self.assertEqual(docs, [])
        self.assertIsNone(source)


if __name__ == "__main__":
    unittest.main()
