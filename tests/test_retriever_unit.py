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


if __name__ == "__main__":
    unittest.main()
